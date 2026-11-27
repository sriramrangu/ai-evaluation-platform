
import gc
import tempfile

from backend.ingestion.document_loader import load_document
from backend.ingestion.text_cleaner import clean_text
from backend.ingestion.chunker import chunk_document
from backend.ingestion.pipeline import process_document
from backend.ingestion.document_registry import DocumentRegistry
from backend.ingestion.document_store import DocumentStore
from backend.ingestion.vector_store import VectorStore

def clear_test_vectors(vector_store):
    """Delete all existing vectors from the test ChromaDB collection."""
    existing = vector_store.collection.get()
    ids = existing.get("ids", [])
    if ids:
        vector_store.collection.delete(ids=ids)

def test_txt_ingestion():

    filename = "sample.txt"
    file_bytes = b"Production AI evaluation platform"

    document = load_document(filename, file_bytes)

    cleaned = clean_text(document.content)

    assert document.file_type == "txt"
    assert document.source == filename
    assert cleaned == "Production AI evaluation platform"

    print("\nDOCUMENT INGESTION SUCCESSFUL")


def test_document_chunking():

    text = (
        "Artificial intelligence is transforming industries. "
        "Machine learning enables systems to learn from data. "
        "Deep learning uses neural networks for complex tasks. "
        "Natural language processing helps computers understand text. "
    ) * 30

    file_bytes = text.encode("utf-8")

    document = load_document("sample.txt", file_bytes)
    document.content = clean_text(document.content)

    chunks = chunk_document(
        document,
        chunk_size=300,
        chunk_overlap=50
    )

    assert len(chunks) > 1

    assert all(chunk["content"].strip() for chunk in chunks)

    assert all(
        chunk["document_id"] == document.document_id
        for chunk in chunks
    )

    assert all(
        chunk["chunk_index"] == index
        for index, chunk in enumerate(chunks)
    )

    print("\nDOCUMENT CHUNKING SUCCESSFUL")
    print("Total chunks:", len(chunks))
    print("\nFirst chunk:")
    print(chunks[0]["content"])
    print("\nFirst chunk metadata:")
    print(chunks[0]["metadata"])


def test_complete_ingestion_pipeline():

    text = (
        "Artificial intelligence is transforming industries. "
        "Machine learning enables systems to learn from data. "
        "Deep learning uses neural networks for complex tasks. "
    ) * 30

    with tempfile.TemporaryDirectory() as temp_dir:

        registry = DocumentRegistry(
            f"{temp_dir}/pipeline_test.db"
        )

        vector_store = VectorStore(
            persist_directory="datasets/chroma_test_db"
        )

        clear_test_vectors(vector_store)

        document_id = None

        try:
            result = process_document(
                "sample.txt",
                text.encode("utf-8"),
                registry=registry,
                vector_store=vector_store
            )

            document = result["document"]
            document_id = document.document_id
            chunks = result["chunks"]
            report = result["report"]

            assert result["status"] == "success"
            assert document.source == "sample.txt"
            assert len(chunks) > 0
            assert result["registry"]["status"] == "registered"
            assert result["storage"]["status"] == "saved"
            assert result["vectors"]["status"] == "success"

            assert report["status"] == "success"
            assert report["total_chunks"] == len(chunks)
            assert report["extracted_characters"] == len(document.content)
            assert report["processing_time_seconds"] >= 0

            assert vector_store.count() == len(chunks)

            print("\nCOMPLETE INGESTION PIPELINE SUCCESSFUL")

            for key, value in report.items():
                print(f"{key}: {value}")

        finally:
            if document_id:
                vector_store.delete_document(document_id)


def test_duplicate_document_detection():

    text = "Artificial intelligence is transforming industries."
    file_bytes = text.encode("utf-8")

    document = load_document("sample.txt", file_bytes)

    with tempfile.TemporaryDirectory() as temp_dir:

        registry = DocumentRegistry(
            f"{temp_dir}/test_registry.db"
        )

        first_result = registry.register_document(
            document,
            file_bytes
        )

        assert first_result["status"] == "registered"
        assert first_result["is_duplicate"] is False

        duplicate_result = registry.find_duplicate(file_bytes)

        assert duplicate_result is not None
        assert duplicate_result["document_id"] == document.document_id

        second_result = registry.register_document(
            document,
            file_bytes
        )

        assert second_result["status"] == "duplicate"
        assert second_result["is_duplicate"] is True

        print("\nDUPLICATE DETECTION SUCCESSFUL")
        print("First upload:", first_result["status"])
        print("Second upload:", second_result["status"])
        print("Document ID:", second_result["document_id"])


def test_pipeline_duplicate_detection():

    text = (
        "Production AI evaluation platform "
        "with document ingestion and automated evaluation."
    )

    file_bytes = text.encode("utf-8")

    with tempfile.TemporaryDirectory() as temp_dir:

        registry = DocumentRegistry(
            f"{temp_dir}/pipeline_registry.db"
        )

        vector_store = VectorStore(
            persist_directory="datasets/chroma_test_db"
        )

        clear_test_vectors(vector_store)

        document_id = None

        try:
            first_result = process_document(
                "sample.txt",
                file_bytes,
                registry=registry,
                vector_store=vector_store
            )

            document_id = first_result["document"].document_id

            second_result = process_document(
                "sample.txt",
                file_bytes,
                registry=registry,
                vector_store=vector_store
            )

            assert first_result["status"] == "success"
            assert first_result["registry"]["status"] == "registered"
            assert len(first_result["chunks"]) > 0

            assert second_result["status"] == "duplicate"
            assert second_result["document"] is None
            assert second_result["chunks"] == []
            assert second_result["registry"]["is_duplicate"] is True

            assert (
                document_id
                == second_result["duplicate"]["document_id"]
            )

            assert (
                vector_store.count()
                == len(first_result["chunks"])
            )

            print("\nPIPELINE DUPLICATE DETECTION SUCCESSFUL")
            print("First upload:", first_result["status"])
            print("Second upload:", second_result["status"])
            print("Existing document ID:", document_id)

        finally:
            if document_id:
                vector_store.delete_document(document_id)


def test_document_storage():

    text = (
        "Production AI evaluation platform with "
        "document ingestion, chunking and storage. "
    ) * 10

    document = load_document(
        "storage_test.txt",
        text.encode("utf-8")
    )

    document.content = clean_text(document.content)

    chunks = chunk_document(
        document,
        chunk_size=100,
        chunk_overlap=20
    )

    with tempfile.TemporaryDirectory() as temp_dir:

        db_path = f"{temp_dir}/document_store.db"

        store = DocumentStore(db_path)

        save_result = store.save_document(
            document,
            chunks
        )

        assert save_result["status"] == "stored"
        assert save_result["total_chunks"] == len(chunks)

        store = DocumentStore(db_path)

        saved_document = store.get_document(
            document.document_id
        )

        saved_chunks = store.get_chunks(
            document.document_id
        )

        assert saved_document is not None
        assert saved_document["content"] == document.content
        assert saved_document["source"] == document.source

        assert len(saved_chunks) == len(chunks)

        assert saved_chunks[0]["content"] == chunks[0]["content"]

        documents = store.list_documents()

        assert len(documents) == 1

        deleted = store.delete_document(
            document.document_id
        )

        assert deleted is True
        assert store.get_document(document.document_id) is None
        assert store.get_chunks(document.document_id) == []

        print("\nDOCUMENT STORAGE SUCCESSFUL")
        print("Stored document:", document.source)
        print("Stored chunks:", len(chunks))
        print("Persistence: verified")
        print("Document deletion: successful")


if __name__ == "__main__":

    test_txt_ingestion()
    test_document_chunking()
    test_complete_ingestion_pipeline()
    test_duplicate_document_detection()
    test_pipeline_duplicate_detection()
    test_document_storage()

    print("\nALL INGESTION TESTS PASSED")
