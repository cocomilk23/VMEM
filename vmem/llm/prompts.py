FACT_EXTRACTION_SYSTEM = (
    "You are a memory extraction engine. Extract concise, standalone factual memories. "
    "Return only JSON."
)

FACT_EXTRACTION_USER = (
    "Text:\n{input_text}\n\n"
    "Extract key factual memories as a list of short sentences. "
    "Output JSON with this schema:\n"
    "{\n"
    '  "facts": ["fact1", "fact2"]\n'
    "}\n"
)

SCORE_AND_TRIPLE_SYSTEM = (
    "You are a value-aware memory analyzer. Score each memory and output a triple. "
    "Return only JSON."
)

SCORE_AND_TRIPLE_USER = (
    "Memory fact:\n{fact}\n\n"
    "Score its value from 0 to 1 based on significance, rarity, and emotional impact. "
    "Also produce a semantic triple (subject, predicate, object). "
    "Output JSON with this schema:\n"
    "{\n"
    '  "value_score": 0.0,\n'
    '  "subject": "entity",\n'
    '  "predicate": "relation",\n'
    '  "object": "entity"\n'
    "}\n"
)

ANSWER_SYSTEM = (
    "You are a concise assistant. Answer the question using only the provided memories. "
    "If the memories do not contain enough information, reply with \"insufficient\"."
)

ANSWER_USER = (
    "Question:\n{question}\n\n"
    "Memories:\n{memories}\n\n"
    "Answer in one or two sentences."
)
