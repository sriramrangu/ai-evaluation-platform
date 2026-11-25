
import json
import re

from google import genai
from langfuse import get_client as get_langfuse_client

from backend.config import GEMINI_API_KEY, GEMINI_MODEL


_client = None


def get_client():
    global _client

    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)

    return _client


def _evaluate_answer(question, answer, retrieved_contexts):

    if not question or not answer:
        raise ValueError("Question and answer are required")

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

        score = results[metric].get("score")

        if not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise ValueError(f"Invalid score for {metric}")

    results["overall_score"] = round(
        sum(results[name]["score"] for name in metric_names) / 3,
        4
    )

    return results


def evaluate_answer(question, answer, retrieved_contexts):

    langfuse = get_langfuse_client()

    with langfuse.start_as_current_observation(
        as_type="span",
        name="rag-answer-evaluation"
    ) as span:

        results = _evaluate_answer(
            question,
            answer,
            retrieved_contexts
        )

        span.update(
            input={
                "question": question,
                "answer": answer,
                "retrieved_contexts": retrieved_contexts
            },
            output=results,
            metadata={
                "overall_score": results["overall_score"],
                "faithfulness": results["faithfulness"]["score"],
                "answer_relevance": results["answer_relevance"]["score"],
                "context_relevance": results["context_relevance"]["score"]
            }
        )

        return results
