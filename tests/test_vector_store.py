
from pathlib import Path

from backend.ingestion.document_loader import load_document
from backend.ingestion.text_cleaner import clean_text
from backend.ingestion.chunker import chunk_document
from backend.ingestion.vector_store import VectorStore


def test_vector_store():

    text = """
    Artificial intelligence enables machines to perform tasks
    that normally require human intelligence.

    Machine learning is a branch of artificial intelligence
    that allows systems to learn patterns from data.

    Deep learning uses multiple neural network layers
    to learn complex representations.

    Retrieval augmented generation combines document retrieval
    with large language models to generate grounded answers.
    """

    file_bytes = text.encode("utf-8")

    document = load_document("sample.txt", file_bytes)
    document.content = clean_text(document.content)

    chunks = chunk_document(
        document,
        chunk_size=200,
        chunk_overlap=40
    )

    test_db_path = Path("datasets/chroma_test_db")
    test_db_path.mkdir(parents=True, exist_ok=True)

    vector_store = VectorStore(
        persist_directory=test_db_path
    )

    result = vector_store.add_chunks(chunks)

    assert result["status"] == "success"
    assert result["stored_chunks"] == len(chunks)

    print("\nEMBEDDING GENERATION SUCCESSFUL")
    print("Stored vectors:", vector_store.count())

    search_results = vector_store.search(
        "What is retrieval augmented generation?",
        top_k=3
    )

    assert len(search_results) > 0

    print("\nSEMANTIC SEARCH SUCCESSFUL")

    for index, result in enumerate(search_results, start=1):
        print(f"\nResult {index}")
        print("Content:", result["content"])
        print("Similarity:", round(result["similarity"], 4))

    document_id = document.document_id

    vector_store.delete_document(document_id)

    assert vector_store.count() == 0

    print("\nVECTOR DELETION SUCCESSFUL")


if __name__ == "__main__":
    test_vector_store()
    print("\nALL VECTOR STORE TESTS PASSED")
