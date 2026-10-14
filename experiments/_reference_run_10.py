import os
import json
import re

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq


# -----------------------------
# 1. Groq setup
# -----------------------------

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "openai/gpt-oss-20b"


# -----------------------------
# 2. Load MIRAGE dataset
# -----------------------------

dataset = load_dataset("nlpai-lab/mirage")["train"]

# First 10 questions only
data = dataset.select(range(10))


# -----------------------------
# 3. Helper: ask LLM
# -----------------------------

def ask_llm(question, context=None):


    if context:
        prompt = f"""
Answer the question strictly using the provided context.

Do not use outside knowledge.
If the answer is stated in the context, use that information.
Give only the answer, without explanation.

Question:
{question}

Context:
{context}

Answer:
"""
    else:
        prompt = f"""
Answer the question.

Question:
{question}

Give only the answer, without explanation.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


# -----------------------------
# 4. Helper: check correctness
# -----------------------------

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_correct(prediction, expected_answers):

    prediction = normalize(prediction)

    for answer in expected_answers:
        answer = normalize(answer)

        if answer in prediction:
            return True

    return False


# -----------------------------
# 5. Diagnose failure
# -----------------------------

def diagnose(base_correct, oracle_correct, retrieved_correct, retrieved_support):

    # Gold context itself could not produce the correct answer -> LLM can't use good context
    if not oracle_correct:
        if base_correct:
            # Knew it without context, but broke when given (even gold) context
            return "LLM/Generator Failure (Context Misinterpretation)"
        if retrieved_support == 0:
            return "LLM/Generator + Retrieval Failure"
        return "LLM/Generator Failure"

    # Gold context works, but retrieved context does not -> retrieval is the bottleneck
    if oracle_correct and not retrieved_correct:
        return "Retriever Failure"

    # Both oracle and retrieved work
    if oracle_correct and retrieved_correct:
        if not base_correct:
            return "No Significant Failure"
        # Model already knew the answer without any context at all
        return "No Significant Failure (Context Insensitive Success)"

    return "Both/Interaction"


# -----------------------------
# 6. Run experiment
# -----------------------------

results = []


for i, row in enumerate(data):

    question = row["query"]
    expected = row["answer"]

    # -------------------------
    # Oracle context
    # -------------------------

    oracle_context = row["oracle"]["doc_chunk"]

    # -------------------------
    # Real retrieval
    # -------------------------

    chunks = row["doc_pool"]["doc_chunk"]

    vectorizer = TfidfVectorizer()

    chunk_vectors = vectorizer.fit_transform(chunks)
    query_vector = vectorizer.transform([question])

    similarities = cosine_similarity(
        query_vector,
        chunk_vectors
    )[0]

    top_index = similarities.argmax()

    retrieved_context = chunks[top_index]

    retrieved_support = row["doc_pool"]["support"][top_index]
    print("Retrieved Context:", retrieved_context)
    print("Retrieved Support:", retrieved_support)
    # -------------------------
    # Three conditions
    # -------------------------

    base_answer = ask_llm(question)

    oracle_answer = ask_llm(
        question,
        oracle_context
    )

    retrieved_answer = ask_llm(
        question,
        retrieved_context
    )

    # -------------------------
    # Correctness
    # -------------------------

    base_correct = is_correct(
        base_answer,
        expected
    )

    oracle_correct = is_correct(
        oracle_answer,
        expected
    )

    retrieved_correct = is_correct(
        retrieved_answer,
        expected
    )

    # -------------------------
    # Diagnosis
    # -------------------------

    diagnosis = diagnose(
    base_correct,
    oracle_correct,
    retrieved_correct,
    retrieved_support
    )
    result = {
        "question": question,
        "expected_answer": expected,

        "base_answer": base_answer,
        "base_correct": base_correct,

        "oracle_answer": oracle_answer,
        "oracle_correct": oracle_correct,

        "retrieved_answer": retrieved_answer,
        "retrieved_correct": retrieved_correct,

        "retrieved_support": retrieved_support,

        "diagnosis": diagnosis
    }

    results.append(result)

    print(f"\nQuestion {i + 1}/10")
    print("Question:", question)
    print("Base:", base_correct)
    print("Oracle:", oracle_correct)
    print("Retrieved:", retrieved_correct)
    print("Diagnosis:", diagnosis)
    print("Expected:", expected)


# -----------------------------
# 7. Save results
# -----------------------------

with open("results_10.json", "w", encoding="utf-8") as f:

    json.dump(
        results,
        f,
        indent=4,
        ensure_ascii=False
    )
from collections import Counter

diagnosis_counts = Counter(
    result["diagnosis"] for result in results
)

print("\n========== DIAGNOSIS SUMMARY ==========")

for diagnosis, count in diagnosis_counts.items():
    percentage = (count / len(results)) * 100
    print(f"{diagnosis}: {count}/{len(results)} ({percentage:.1f}%)")


print("\n--------------------------------")
print("Experiment complete!")
print("Saved as: results_10.json")
print("--------------------------------")