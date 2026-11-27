from backend.evaluation_metrics.citation_metrics import citation_validity
from backend.evaluation_metrics.retrieval_metrics import retrieval_metrics
from backend.evaluation_metrics.retrieval_metrics import retrieval_diversity
from backend.evaluation_metrics.performance_metrics import (
    duplicate_retrieval_rate,
    context_utilization
)
from backend.evaluation import _evaluate_answer

def test_all_citations_valid():

    answer = "ML learns from data [Source 1]. It uses algorithms [Source 2]."

    contexts = [
        "Introduction to machine learning",
        "Machine learning algorithms"
    ]

    result = citation_validity(answer, contexts)

    assert result["score"] == 1.0
    assert result["invalid_citations"] == []


def test_invalid_citation():

    answer = "ML learns from data [Source 1]. Healthcare uses [Source 5]."

    contexts = [
        "Introduction to machine learning",
        "Machine learning algorithms"
    ]

    result = citation_validity(answer, contexts)

    assert result["score"] == 0.5
    assert result["invalid_citations"] == [5]


def test_no_citations():

    answer = "Machine learning enables computers to learn from data."

    contexts = ["Introduction to machine learning"]

    result = citation_validity(answer, contexts)

    assert result["status"] == "not_applicable"
    assert result["score"] is None


def test_duplicate_citations():

    answer = "ML learns from data [Source 1]. It improves predictions [Source 1]."

    contexts = ["Introduction to machine learning"]

    result = citation_validity(answer, contexts)

    assert result["total_citations"] == 1
    assert result["score"] == 1.0
    assert result["invalid_citations"] == []



def test_retrieval_metrics():

    retrieved = ["A", "B", "C", "D"]
    relevant = ["A", "B", "C", "E"]

    result = retrieval_metrics(retrieved, relevant)

    assert result["precision"] == 0.75
    assert result["recall"] == 0.75
    assert result["f1_score"] == 0.75

    assert result["true_positives"] == 3
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1


def test_perfect_retrieval():

    retrieved = ["A", "B", "C"]
    relevant = ["A", "B", "C"]

    result = retrieval_metrics(retrieved, relevant)

    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1_score"] == 1.0


def test_no_relevant_chunks_retrieved():

    retrieved = ["A", "B"]
    relevant = ["C", "D"]

    result = retrieval_metrics(retrieved, relevant)

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1_score"] == 0.0


def test_empty_retrieval():

    retrieved = []
    relevant = ["A", "B"]

    result = retrieval_metrics(retrieved, relevant)

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1_score"] == 0.0


def test_custom_evaluation_metrics():
    answer = "Machine learning systems learn patterns from data."

    contexts = [
        "Machine learning enables systems to learn from data.",
        "Deep learning uses neural networks for complex tasks."
    ]

    result = _evaluate_answer(
        "What is machine learning?",
        answer,
        contexts,
        1.25
    )

    metrics = result["custom_metrics"]

    assert metrics["answer_context_overlap"]["score"] > 0
    assert metrics["retrieved_chunk_utilization"]["score"] == 0.5
    assert metrics["retrieval_redundancy"]["score"] >= 0
    assert result["response_time_seconds"] == 1.25


def test_custom_evaluation_with_no_contexts():
    result = _evaluate_answer(
        "What is machine learning?",
        "Machine learning learns patterns from data.",
        []
    )

    metrics = result["custom_metrics"]

    assert metrics["answer_context_overlap"]["score"] == 0
    assert metrics["retrieved_chunk_utilization"]["score"] == 0
    assert metrics["retrieval_redundancy"]["score"] == 0


def test_retrieval_diversity():

    contexts = [
        "Machine learning uses data.",
        "Deep learning uses neural networks.",
        "Machine learning uses data.",
        "Artificial intelligence is evolving."
    ]

    result = retrieval_diversity(contexts)

    assert result["score"] == 0.75
    assert result["total_chunks"] == 4
    assert result["unique_chunks"] == 3
    assert result["duplicate_chunks"] == 1


def test_perfect_retrieval_diversity():

    contexts = [
        "Machine learning",
        "Deep learning",
        "Artificial intelligence"
    ]

    result = retrieval_diversity(contexts)

    assert result["score"] == 1.0


def test_zero_retrieval_diversity():

    contexts = [
        "Machine learning",
        "Machine learning",
        "Machine learning"
    ]

    result = retrieval_diversity(contexts)

    assert result["score"] == 0.3333


def test_empty_retrieval_diversity():

    result = retrieval_diversity([])

    assert result["score"] == 0.0
