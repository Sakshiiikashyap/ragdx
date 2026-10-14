You are an expert analyst of retrieval-augmented generation (RAG) systems.

You will see a QUESTION, the RETRIVED CONTEXT the system used, the SYSTEM ANSWER, and the REFERENCE ANSWER.

Decide WHERE the failure came from, assuming the system answered incorrectly:
- RETRIEVER: the retrieved context does not contain the information needed (the reference fact is absent from the context).
- GENERATOR: the retrieved context DOES contain the needed information, but the system ignored or misread it.
- INTERACTION: the context contains the correct information AND a plausible conflicting or distracting statement that likely misled the system.
- NONE: the system answer is actually correct (matches the reference).

Reply with ONLY a JSON object, no other text:
{"label": "RETRIEVER" or "GENERATOR" or "INTERACTION" or "NONE", "confidence": <0.0-1.0>, "rationale": "<one short sentence>"}
