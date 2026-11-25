
import chromadb

from pathlib import Path
from sentence_transformers import SentenceTransformer


DEFAULT_VECTOR_DB = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "chroma_db"
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class VectorStore:

    def __init__(
        self,
        persist_directory=DEFAULT_VECTOR_DB,
        collection_name="ai_evaluation_documents"
    ):
        self.client = chromadb.PersistentClient(
            path=str(persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def generate_embeddings(self, texts):

        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        return embeddings.tolist()

    @staticmethod
    def prepare_metadata(chunk):

        metadata = chunk.get("metadata", {}).copy()

        metadata["document_id"] = str(chunk["document_id"])
        metadata["chunk_index"] = int(chunk["chunk_index"])

        for key, value in metadata.items():
            if value is None:
                metadata[key] = ""
            elif isinstance(value, (dict, list)):
                metadata[key] = str(value)
            elif not isinstance(value, (str, int, float, bool)):
                metadata[key] = str(value)

        return metadata

    def add_chunks(self, chunks):

        if not chunks:
            return {
                "status": "empty",
                "stored_chunks": 0
            }

        texts = [chunk["content"] for chunk in chunks]

        embeddings = self.generate_embeddings(texts)

        self.collection.upsert(
            ids=[chunk["chunk_id"] for chunk in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                self.prepare_metadata(chunk)
                for chunk in chunks
            ]
        )

        return {
            "status": "success",
            "stored_chunks": len(chunks)
        }

    def search(self, query, top_k=5):

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if self.collection.count() == 0:
            return []

        query_embedding = self.generate_embeddings([query])

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        matches = []

        for index in range(len(results["ids"][0])):

            distance = results["distances"][0][index]

            matches.append({
                "chunk_id": results["ids"][0][index],
                "content": results["documents"][0][index],
                "metadata": results["metadatas"][0][index],
                "distance": distance,
                "similarity": 1 - distance
            })

        return matches

    def delete_document(self, document_id):

        self.collection.delete(
            where={"document_id": str(document_id)}
        )

        return {
            "status": "success",
            "document_id": document_id
        }

    def count(self):

        return self.collection.count()
