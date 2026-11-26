
from backend.rag import answer_question
# from backend.evaluation import evaluate_answer
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
        retrieved_contexts=result["retrieved_contexts"]
    )

    print("\nCUSTOM EVALUATION RESULTS:")
    print(evaluation)

    print("\nEVALUATION RESULTS:")

    for metric, details in evaluation.items():
        if metric == "overall_score":
            continue

        print(f"\n{metric}: {details['score']}")
        print("Reason:", details["reason"])

    print("\nOVERALL SCORE:", evaluation["overall_score"])

    print("\nRAG AND EVALUATION TESTS PASSED")

    get_client().flush()
    print("\nLANGFUSE EVENTS FLUSHED")


if __name__ == "__main__":
    test_rag()
