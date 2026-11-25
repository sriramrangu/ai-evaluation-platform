
import time

from backend.ingestion.file_validator import validate_file
from backend.ingestion.document_loader import load_document
from backend.ingestion.text_cleaner import clean_text
from backend.ingestion.chunker import chunk_document
from backend.ingestion.ingestion_report import generate_ingestion_report
from backend.ingestion.document_registry import DocumentRegistry
from backend.ingestion.storage import DocumentStorage

from backend.ingestion.vector_store import VectorStore

_vector_store = None


def get_vector_store():
    global _vector_store

    if _vector_store is None:
        _vector_store = VectorStore()

    return _vector_store

def process_document(
    filename,
    file_bytes,
    registry=None,
    storage=None,
    vector_store=None
):
    start_time = time.perf_counter()

    validate_file(filename, file_bytes)

    if registry is None:
        registry = DocumentRegistry()

    if storage is None:
        storage = DocumentStorage(registry.db_path)

    duplicate = registry.find_duplicate(file_bytes)

    if duplicate is not None:
                return {
            "status": "duplicate",
            "document": None,
            "chunks": [],
            "duplicate": duplicate,
            "registry": {
                "status": "duplicate",
                "is_duplicate": True,
                "document_id": duplicate["document_id"]
            },
            "storage": None,
            "report": {
                "source": duplicate["source"],
                "status": "duplicate",
                "warnings": ["Document was already uploaded"]
            }
        }

    document = load_document(filename, file_bytes)

    document.content = clean_text(document.content)

    chunks = chunk_document(document)

    registry_result = registry.register_document(
        document,
        file_bytes
    )

    if registry_result["is_duplicate"]:
        return {
            "status": "duplicate",
            "document": None,
            "chunks": [],
            "duplicate": registry_result,
            "storage": None,
            "report": {
                "source": document.source,
                "status": "duplicate",
                "warnings": ["Document was already uploaded"]
            }
        }

    storage_result = storage.save_document(
        document,
        chunks
    )
    if vector_store is None:
        vector_store = get_vector_store()

    vector_result = vector_store.add_chunks(chunks)

    processing_time = time.perf_counter() - start_time

    report = generate_ingestion_report(
        document,
        chunks,
        processing_time
    )

    return {
    "status": "success",
    "document": document,
    "chunks": chunks,
    "duplicate": None,
    "registry": registry_result,
    "storage": storage_result,
    "vectors": vector_result,
    "report": report
}
