from backend.rag import answer_question
from backend.evaluation import evaluate_answer
from langfuse import get_client


def test_rag():

    question = "What is machine learning?"

    result = answer_question(question)

    print("\nRAG QUESTION:")
    print(result["question"])

    print("\nGENERATED ANSWER:")
    print(result["answer"])

    print("\nRETRIEVED SOURCES:")

    for source in result["sources"]:
        print(source)

    assert result["status"] == "success"
    assert result["answer"]
    assert len(result["retrieved_contexts"]) > 0

    print("\nRUNNING RAG EVALUATION")

    evaluation = evaluate_answer(
        question=result["question"],
        answer=result["answer"],
        retrieved_contexts=result["retrieved_contexts"],
        response_time_seconds=result["response_time_seconds"]
    )

    assert "custom_metrics" in evaluation
    assert "semantic_metrics" in evaluation
    assert "overall_score" in evaluation["semantic_metrics"]

    print("\nCUSTOM METRICS")

    for metric, details in evaluation["custom_metrics"].items():
        print(f"{metric}: {details['score']}")

    print("\nSEMANTIC METRICS")

    for metric, details in evaluation["semantic_metrics"].items():
        if metric == "overall_score":
            continue

        print(f"{metric}: {details['score']}")
        print(f"Reason: {details['reason']}")

    print("\nRESPONSE TIME:")
    print(evaluation["response_time_seconds"])

    print("\nOVERALL SCORE:")
    print(evaluation["semantic_metrics"]["overall_score"])

    print("\nRAG AND EVALUATION TESTS PASSED")

    get_client().flush()

    print("\nLANGFUSE EVENTS FLUSHED")


if __name__ == "__main__":
    test_rag()
