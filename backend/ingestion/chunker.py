
from backend.ingestion.schemas import ProcessedDocument


def split_text(
    text,
    chunk_size=1000,
    chunk_overlap=200
):

    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative "
            "and smaller than chunk_size"
        )

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )

        if end < text_length:

            minimum_boundary = start + int(
                chunk_size * 0.7
            )

            candidates = [
                text.rfind("\n\n", minimum_boundary, end),
                text.rfind("\n", minimum_boundary, end),
                text.rfind(". ", minimum_boundary, end),
                text.rfind(" ", minimum_boundary, end)
            ]

            boundary = max(candidates)

            if boundary != -1:
                end = boundary + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append({
                "content": chunk,
                "start_index": start,
                "end_index": end
            })

        if end >= text_length:
            break

        next_start = end - chunk_overlap

        if next_start <= start:
            raise RuntimeError(
                "Chunking failed to make progress"
            )

        start = next_start

    return chunks


def chunk_document(
    document: ProcessedDocument,
    chunk_size=1000,
    chunk_overlap=200
):

    text_chunks = split_text(
        document.content,
        chunk_size,
        chunk_overlap
    )

    results = []

    for index, chunk in enumerate(text_chunks):

        results.append({
            "chunk_id": f"{document.document_id}_{index}",
            "document_id": document.document_id,
            "chunk_index": index,
            "content": chunk["content"],
            "metadata": {
                "source": document.source,
                "file_type": document.file_type,
                "start_index": chunk["start_index"],
                "end_index": chunk["end_index"],
                **document.metadata
            }
        })

    return results
