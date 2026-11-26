
def duplicate_retrieval_rate(retrieved_contexts):

    if retrieved_contexts is None:
        raise ValueError("Retrieved contexts cannot be None")

    if not retrieved_contexts:
        return {
            "score": None,
            "status": "not_applicable"
        }

    normalized = [
        " ".join(context.lower().split())
        for context in retrieved_contexts
    ]

    duplicates = len(normalized) - len(set(normalized))

    return {
        "score": round(duplicates / len(normalized), 4),
        "total_chunks": len(normalized),
        "duplicate_chunks": duplicates,
        "status": "evaluated"
    }


def context_utilization(answer, retrieved_contexts):

    if not answer:
        raise ValueError("Answer cannot be empty")

    if retrieved_contexts is None:
        raise ValueError("Retrieved contexts cannot be None")

    if not retrieved_contexts:
        return {
            "score": None,
            "status": "not_applicable"
        }

    answer_words = set(answer.lower().split())

    context_words = set(
        " ".join(retrieved_contexts).lower().split()
    )

    if not context_words:
        return {
            "score": 0.0,
            "status": "evaluated"
        }

    matched_words = answer_words.intersection(context_words)

    return {
        "score": round(
            len(matched_words) / len(context_words),
            4
        ),
        "matched_words": len(matched_words),
        "context_words": len(context_words),
        "status": "evaluated"
    }
