import time

from google import genai

from backend.config import GEMINI_API_KEY, GEMINI_MODEL
from backend.ingestion.vector_store import VectorStore

from langfuse import get_client as get_langfuse_client

_client = None
_vector_store = None


def get_client():
    global _client

    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)

    return _client


def get_vector_store():
    global _vector_store

    if _vector_store is None:
        _vector_store = VectorStore()

    return _vector_store


def _answer_question(question, top_k=4):

    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    start_time = time.perf_counter()

    vector_store = get_vector_store()

    retrieved_chunks = vector_store.search(
        question,
        top_k=top_k
    )

    if not retrieved_chunks:
        return {
            "status": "no_context",
            "question": question,
            "answer": "No relevant documents were found. Please upload a document first.",
            "sources": [],
            "retrieved_contexts": [],
            "response_time_seconds": round(
                time.perf_counter() - start_time, 3
            )
        }

    context_parts = []
    sources = []

    for index, chunk in enumerate(retrieved_chunks, start=1):

        metadata = chunk["metadata"]

        source = {
            "source": metadata.get("source", "Unknown"),
            "chunk_id": chunk["chunk_id"],
            "chunk_index": metadata.get("chunk_index"),
            "similarity": round(chunk["similarity"], 4)
        }

        sources.append(source)

        context_parts.append(
            f"[Source {index}: {source['source']}]\n"
            f"{chunk['content']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using only the supplied document context.

Rules:
1. Do not invent facts or use outside knowledge.
2. If the context does not contain the answer, clearly say that
   the uploaded documents do not provide enough information.
3. Give a clear and concise answer.
4. Mention the relevant source numbers when appropriate.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = None

    for attempt in range(3):
        try:
            response = get_client().models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            break

        except Exception as error:
            if "503" not in str(error) and "UNAVAILABLE" not in str(error):
                raise

            if attempt == 2:
                raise RuntimeError(
                    "Gemini is temporarily unavailable. Please retry later."
                ) from error

            time.sleep(2 ** (attempt + 1))

    answer = response.text or "Unable to generate an answer."

    return {
        "status": "success",
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_contexts": [
            chunk["content"] for chunk in retrieved_chunks
        ],
        "response_time_seconds": round(
            time.perf_counter() - start_time, 3
        )
    }


def answer_question(question, top_k=4):

    langfuse = get_langfuse_client()

    with langfuse.start_as_current_observation(
        as_type="span",
        name="rag-question-answering"
    ) as span:

        result = _answer_question(question, top_k)

        span.update(
            input={
                "question": question,
                "top_k": top_k
            },
            output={
                "answer": result["answer"],
                "sources": result["sources"],
                "status": result["status"]
            },
            metadata={
                "retrieved_chunks": len(result["retrieved_contexts"]),
                "response_time_seconds": result["response_time_seconds"]
            }
        )

        return result
