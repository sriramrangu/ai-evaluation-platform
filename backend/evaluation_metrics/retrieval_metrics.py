
def retrieval_metrics(retrieved_chunk_ids, relevant_chunk_ids):

    if retrieved_chunk_ids is None:
        raise ValueError("Retrieved chunk IDs cannot be None")

    if relevant_chunk_ids is None or not relevant_chunk_ids:
        raise ValueError("Relevant chunk IDs cannot be empty")

    retrieved = set(retrieved_chunk_ids)
    relevant = set(relevant_chunk_ids)

    true_positives = retrieved.intersection(relevant)
    false_positives = retrieved - relevant
    false_negatives = relevant - retrieved

    precision = (
        len(true_positives) / len(retrieved)
        if retrieved else 0.0
    )

    recall = len(true_positives) / len(relevant)

    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "true_positives": len(true_positives),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "retrieved_relevant_ids": sorted(true_positives),
        "missed_relevant_ids": sorted(false_negatives),
        "irrelevant_retrieved_ids": sorted(false_positives)
    }


def retrieval_diversity(retrieved_contexts):

    if retrieved_contexts is None:
        raise ValueError("Retrieved contexts cannot be None")

    if not retrieved_contexts:
        return {
            "score": None,
            "status": "not_applicable",
            "total_chunks": 0,
            "unique_chunks": 0,
            "duplicate_chunks": 0
        }

    normalized_chunks = {
        " ".join(context.lower().split())
        for context in retrieved_contexts
    }

    total = len(retrieved_contexts)
    unique = len(normalized_chunks)

    return {
        "score": round(unique / total, 4),
        "status": "evaluated",
        "total_chunks": total,
        "unique_chunks": unique,
        "duplicate_chunks": total - unique
    }
