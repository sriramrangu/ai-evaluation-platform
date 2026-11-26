from backend.rag import get_vector_store


question = "What is machine learning?"

vector_store = get_vector_store()

results = vector_store.search(
    question,
    top_k=4
)

print("\nQUESTION:", question)
print("\nRETRIEVED CHUNKS:", len(results))

for index, chunk in enumerate(results, start=1):

    print(f"\n{'=' * 50}")
    print("RANK:", index)
    print("CHUNK ID:", chunk["chunk_id"])
    print("SIMILARITY:", chunk["similarity"])
    print("METADATA:", chunk["metadata"])
    print("CONTENT:", chunk["content"])
