import re

def citation_validity(answer, retrieved_contexts):

    if not answer:
        raise ValueError("Answer cannot be empty")

    if retrieved_contexts is None:
        raise ValueError("Retrieved contexts cannot be None")

    citations = set(
        int(number)
        for number in re.findall(
            r"\[Source\s+(\d+)\]",
            answer,
            re.IGNORECASE
        )
    )

    if not citations:
        return { 
            "score": None,
            "status": "not_applicable",
            "total_citations": 0,
            "valid_citations": 0,
            "invalid_citations": []
        }

    valid = {
        citation 
        for citation in citations
        if 1 <= citation <= len(retrieved_contexts)
    }

    invalid = sorted(citations - valid)

    score = len(valid) / len(citations)

    return {
        "score": round(score, 4),
        "status": "evaluated",
        "total_citations": len(citations),
        "valid_citations": len(valid),
        "invalid_citations": invalid
    }