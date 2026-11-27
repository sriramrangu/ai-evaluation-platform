import json
import re
from itertools import combinations

from google import genai
from langfuse import get_client as get_langfuse_client

from backend.config import GEMINI_API_KEY, GEMINI_MODEL


_client = None


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "to", "of", "in", "on", "for", "and", "or", "but",
    "with", "by", "as", "at", "from", "that", "this", "it",
    "its", "their", "they", "them", "we", "you", "your", "i"
}


def get_client():
    global _client

    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)

    return _client


def _tokens(text):
    words = re.findall(r"\b[a-z0-9]+\b", text.lower())
    return {word for word in words if word not in STOPWORDS}

# 1. CUSTOM PYTHON METRICS

def _evaluate_custom_metrics(
    question,
    answer,
    retrieved_contexts,
    response_time_seconds=None
):

    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    if not answer or not answer.strip():
        raise ValueError("Answer cannot be empty")

    if retrieved_contexts is None:
        raise ValueError("Retrieved contexts cannot be None")

    if not isinstance(retrieved_contexts, list):
        raise ValueError("Retrieved contexts must be a list")

    if any(not isinstance(context, str) for context in retrieved_contexts):
        raise ValueError("Every retrieved context must be a string")

    answer_tokens = _tokens(answer)

    context_tokens = _tokens(" ".join(retrieved_contexts))

    answer_context_overlap = (
        len(answer_tokens & context_tokens) / len(answer_tokens)
        if answer_tokens else 0.0
    )

    chunk_tokens = [_tokens(chunk) for chunk in retrieved_contexts]

    utilized_chunks = sum(
        1 for tokens in chunk_tokens
        if len(tokens & answer_tokens) >= 2
    )

    chunk_utilization = (
        utilized_chunks / len(chunk_tokens)
        if chunk_tokens else 0.0
    )

    similarities = []

    for first, second in combinations(chunk_tokens, 2):

        union = first | second

        similarity = (
            len(first & second) / len(union)
            if union else 0.0
        )

        similarities.append(similarity)

    retrieval_redundancy = (
        sum(similarities) / len(similarities)
        if similarities else 0.0
    )

    return {
        "custom_metrics": {
            "answer_context_overlap": {
                "score": round(answer_context_overlap, 4),
                "description": "Proportion of answer terms found in retrieved context"
            },
            "retrieved_chunk_utilization": {
                "score": round(chunk_utilization, 4),
                "utilized_chunks": utilized_chunks,
                "total_chunks": len(chunk_tokens),
                "description": "Proportion of retrieved chunks sharing terms with the answer"
            },
            "retrieval_redundancy": {
                "score": round(retrieval_redundancy, 4),
                "description": "Average pairwise Jaccard similarity between retrieved chunks"
            }
        },
        "response_time_seconds": (
            round(response_time_seconds, 4)
            if response_time_seconds is not None else None
        )
    }


# Preserve the original function name for compatibility.
def _evaluate_answer(
    question,
    answer,
    retrieved_contexts,
    response_time_seconds=None
):

    return _evaluate_custom_metrics(
        question,
        answer,
        retrieved_contexts,
        response_time_seconds
    )


# 2. GEMINI LLM-AS-A-JUDGE EVALUATION

def _evaluate_with_gemini(question, answer, retrieved_contexts):

    if not retrieved_contexts:
        raise ValueError("Retrieved contexts cannot be empty")

    context = "\n\n".join(retrieved_contexts)

    prompt = f"""
You are an evaluator for a Retrieval Augmented Generation system.

Evaluate the answer using the question and retrieved context.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Evaluate these three metrics:

1. faithfulness:
Does the answer contain only claims supported by the context?

2. answer_relevance:
Does the answer directly address the user's question?

3. context_relevance:
How relevant is the retrieved context to answering the question?

Scoring:
0.0 = completely poor
0.5 = partially satisfactory
1.0 = excellent

Return ONLY valid JSON in this exact structure:

{{
    "faithfulness": {{
        "score": 0.0,
        "reason": "Explanation"
    }},
    "answer_relevance": {{
        "score": 0.0,
        "reason": "Explanation"
    }},
    "context_relevance": {{
        "score": 0.0,
        "reason": "Explanation"
    }}
}}

Scores must be numbers between 0 and 1.
"""

    response = get_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    raw_output = response.text or ""

    raw_output = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        raw_output.strip()
    )

    try:
        results = json.loads(raw_output)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Gemini returned invalid evaluation JSON: {raw_output}"
        ) from error

    metric_names = [
        "faithfulness",
        "answer_relevance",
        "context_relevance"
    ]

    for metric in metric_names:

        if metric not in results:
            raise ValueError(f"Missing evaluation metric: {metric}")

        if not isinstance(results[metric], dict):
            raise ValueError(f"Invalid evaluation structure: {metric}")

        score = results[metric].get("score")

        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0 <= score <= 1
        ):
            raise ValueError(f"Invalid score for {metric}")

        if not isinstance(results[metric].get("reason"), str):
            raise ValueError(f"Invalid reason for {metric}")

    results["overall_score"] = round(
        sum(results[name]["score"] for name in metric_names) / 3,
        4
    )

    return results


# Preserve the original private function name.
def _evaluate_semantic_metrics(
    question,
    answer,
    retrieved_contexts
):

    return _evaluate_with_gemini(
        question,
        answer,
        retrieved_contexts
    )


# 3. COMPLETE EVALUATION

def evaluate_answer(
    question,
    answer,
    retrieved_contexts,
    response_time_seconds=None,
    use_llm_judge=True,
    use_langfuse=True
):

    results = _evaluate_custom_metrics(
        question,
        answer,
        retrieved_contexts,
        response_time_seconds
    )

    if not use_llm_judge:
        return results

    if not retrieved_contexts:
        raise ValueError(
            "Retrieved contexts cannot be empty for semantic evaluation"
        )

    if use_langfuse:

        langfuse = get_langfuse_client()

        with langfuse.start_as_current_observation(
            as_type="span",
            name="rag-answer-evaluation"
        ) as span:

            judge_results = _evaluate_with_gemini(
                question,
                answer,
                retrieved_contexts
            )

            results["semantic_metrics"] = judge_results

            span.update(
                input={
                    "question": question,
                    "answer": answer,
                    "retrieved_contexts": retrieved_contexts
                },
                output=results,
                metadata={
                    "overall_score": judge_results["overall_score"],
                    "faithfulness": judge_results["faithfulness"]["score"],
                    "answer_relevance": judge_results["answer_relevance"]["score"],
                    "context_relevance": judge_results["context_relevance"]["score"]
                }
            )

    else:

        results["semantic_metrics"] = _evaluate_with_gemini(
            question,
            answer,
            retrieved_contexts
        )

    return results
