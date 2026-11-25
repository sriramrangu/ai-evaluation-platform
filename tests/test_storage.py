
import tempfile

from backend.ingestion.document_loader import load_document
from backend.ingestion.chunker import chunk_document
from backend.ingestion.text_cleaner import clean_text
from backend.ingestion.storage import DocumentStorage
from backend.ingestion.document_registry import DocumentRegistry


def test_document_storage():

    text = (
        "Artificial intelligence enables intelligent systems. "
        "Machine learning processes data to discover patterns. "
    ) * 20

    file_bytes = text.encode("utf-8")

    with tempfile.TemporaryDirectory() as temp_dir:

        db_path = f"{temp_dir}/storage.db"

        registry = DocumentRegistry(db_path)
        storage = DocumentStorage(db_path)

        document = load_document("sample.txt", file_bytes)
        document.content = clean_text(document.content)

        chunks = chunk_document(document)

        registry_result = registry.register_document(
            document,
            file_bytes
        )

        assert registry_result["status"] == "registered"

        result = storage.save_document(
            document,
            chunks
        )

        assert result["status"] == "saved"

        saved_document = storage.get_document(
            document.document_id
        )

        saved_chunks = storage.get_chunks(
            document.document_id
        )

        assert saved_document is not None
        assert len(saved_chunks) == len(chunks)

        print("\nDOCUMENT STORAGE SUCCESSFUL")
        print("Document ID:", saved_document["document_id"])
        print("Stored chunks:", len(saved_chunks))


if __name__ == "__main__":
    test_document_storage()
    print("\nALL STORAGE TESTS PASSED")
