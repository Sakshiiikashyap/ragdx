"""LLM client: Groq-backed chat wrapper with on-disk caching and cost logging.

Rule 3 of CLAUDE.md: ALL LLM calls go through this module.
Cache key = sha256(model + params + messages). Every call appends a record
to results/_llm_calls.jsonl with token usage (cost is logged when the API
returns a cost field; otherwise tokens are reported and unit prices are left
for the config layer).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # reads .env if present


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    latency_s: float = 0.0
    meta: dict = field(default_factory=dict)


class LLMClient:
    """Disk-cached chat client. temperature defaults to 0 per CLAUDE.md rule 4."""

    def __init__(self, model: str = "openai/gpt-oss-20b", cache_dir: str | Path = ".llm_cache",
                 log_path: str | Path = "results/_llm_calls.jsonl", temperature: float = 0.0,
                 max_retries: int = 3):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._client = None

    # -- lazy Groq client so tests can run without a key --------------------
    @property
    def client(self):
        if self._client is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY not set (put it in .env or the environment)")
            from groq import Groq
            self._client = Groq(api_key=api_key)
        return self._client

    def _cache_path(self, messages: list[dict], **params: Any) -> Path:
        payload = json.dumps({"model": self.model, "messages": messages,
                              "temperature": self.temperature, **params},
                             sort_keys=True, ensure_ascii=False)
        key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def chat(self, messages: list[dict], **params: Any) -> LLMResponse:
        path = self._cache_path(messages, **params)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return LLMResponse(text=data["text"], model=self.model,
                               prompt_tokens=data.get("prompt_tokens", 0),
                               completion_tokens=data.get("completion_tokens", 0),
                               cached=True, meta=data.get("meta", {}))

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=self.temperature, **params)
                latency = time.time() - t0
                text = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                pt = getattr(usage, "prompt_tokens", 0) or 0
                ct = getattr(usage, "completion_tokens", 0) or 0
                path.write_text(json.dumps({"text": text, "prompt_tokens": pt,
                                            "completion_tokens": ct,
                                            "meta": {"latency_s": latency}},
                                           ensure_ascii=False), encoding="utf-8")
                self._log({"ts": time.time(), "model": self.model, "cached": False,
                           "prompt_tokens": pt, "completion_tokens": ct,
                           "latency_s": round(latency, 3),
                           "cache_key": path.stem})
                return LLMResponse(text=text, model=self.model, prompt_tokens=pt,
                                   completion_tokens=ct, latency_s=latency)
            except Exception as e:  # transient API errors
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")

    def _log(self, record: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
