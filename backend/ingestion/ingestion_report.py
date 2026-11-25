
from datetime import datetime, timezone


def generate_ingestion_report(document, chunks, processing_time=0):

    content = document.content

    chunk_sizes = [
        len(chunk["content"])
        for chunk in chunks
    ]

    total_chunks = len(chunks)

    average_chunk_size = (
        sum(chunk_sizes) / total_chunks
        if total_chunks else 0
    )

    maximum_chunk_size = (
        max(chunk_sizes)
        if chunk_sizes else 0
    )

    minimum_chunk_size = (
        min(chunk_sizes)
        if chunk_sizes else 0
    )

    warnings = []

    if not content.strip():
        warnings.append("Document contains no extracted text")

    if not chunks:
        warnings.append("No chunks were generated")

    if len(content) < 100:
        warnings.append("Document contains very little text")

    report = {
        "document_id": document.document_id,
        "source": document.source,
        "file_type": document.file_type,

        "status": (
            "success"
            if content.strip() and chunks
            else "warning"
        ),

        "file_size_bytes": document.metadata.get(
            "file_size_bytes", 0
        ),

        "extracted_characters": len(content),

        "total_chunks": total_chunks,

        "average_chunk_size": round(
            average_chunk_size, 2
        ),

        "maximum_chunk_size": maximum_chunk_size,

        "minimum_chunk_size": minimum_chunk_size,

        "processing_time_seconds": round(
            processing_time, 4
        ),

        "warnings": warnings,

        "generated_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    return report
