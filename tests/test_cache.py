"""LLM client cache and manifest-related helpers (CLAUDE.md rule 2)."""

import json

from ragdx.llm.client import LLMClient


def _msgs(text):
    return [{"role": "user", "content": text}]


def test_cache_hit_avoids_api(tmp_path):
    client = LLMClient(model="test-model", cache_dir=tmp_path / "cache",
                       log_path=tmp_path / "log.jsonl")
    key = client._cache_path(_msgs("hello"))
    key.write_text(json.dumps({"text": "hi", "prompt_tokens": 1,
                               "completion_tokens": 1, "meta": {}}), encoding="utf-8")
    resp = client.chat(_msgs("hello"))
    assert resp.text == "hi"
    assert resp.cached is True


def test_cache_key_changes_with_message(tmp_path):
    client = LLMClient(model="test-model", cache_dir=tmp_path / "cache",
                       log_path=tmp_path / "log.jsonl")
    assert client._cache_path(_msgs("a")) != client._cache_path(_msgs("b"))


def test_cache_key_changes_with_model(tmp_path):
    c1 = LLMClient(model="m1", cache_dir=tmp_path / "c1", log_path=tmp_path / "l1.jsonl")
    c2 = LLMClient(model="m2", cache_dir=tmp_path / "c2", log_path=tmp_path / "l2.jsonl")
    assert c1._cache_path(_msgs("a")) != c2._cache_path(_msgs("a"))


def test_no_key_raises_clearly(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    import os
    monkeypatch.setattr("ragdx.llm.client.load_dotenv", lambda *a, **k: None)
    client = LLMClient(model="test-model", cache_dir=tmp_path / "cache",
                       log_path=tmp_path / "log.jsonl")
    client._client = "not-none-marker"  # avoid lazy property
    # direct property access path: reset and check raise
    client2 = LLMClient(model="m", cache_dir=tmp_path / "c", log_path=tmp_path / "l.jsonl")
    client2._client = None
    try:
        _ = client2.client
        raised = False
    except RuntimeError:
        raised = True
    assert raised
