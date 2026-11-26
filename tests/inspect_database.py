
from backend.ingestion.vector_store import VectorStore

vector_store = VectorStore()

results = vector_store.collection.get(
    include=["documents", "metadatas"]
)

print("\nDATABASE ANALYSIS")
print("Total stored chunks:", len(results["ids"]))

print("\nDOCUMENT SOURCES")

sources = set()

for metadata in results["metadatas"]:
    sources.add(metadata.get("source", "Unknown"))

for source in sources:
    print(source)

print("\nSTORED CHUNK CONTENT")

for index, chunk_id in enumerate(results["ids"]):

    print("\n" + "=" * 60)
    print("CHUNK ID:", chunk_id)
    print("METADATA:", results["metadatas"][index])
    print("CONTENT:", results["documents"][index][:500])
