
from pathlib import Path

from backend.ingestion.chunker import split_text


file_path = Path("datasets/sample.txt")

if not file_path.exists():
    raise FileNotFoundError(
        f"File not found: {file_path.resolve()}"
    )

text = file_path.read_text(encoding="utf-8-sig")

chunks = split_text(
    text,
    chunk_size=1000,
    chunk_overlap=200
)

print("\nDOCUMENT ANALYSIS")
print("Total characters:", len(text))
print("Total chunks:", len(chunks))

print("\nCHUNK ANALYSIS")

for index, chunk in enumerate(chunks):

    print(f"\nCHUNK {index}")
    print("Start:", chunk["start_index"])
    print("End:", chunk["end_index"])
    print("Characters:", len(chunk["content"]))
    print("Preview:", chunk["content"][:200])

unique_chunks = set(
    chunk["content"] for chunk in chunks
)

print("\nDUPLICATE ANALYSIS")
print("Total chunks:", len(chunks))
print("Unique chunks:", len(unique_chunks))
print("Duplicate chunks:", len(chunks) - len(unique_chunks))

print("\nDOCUMENT PREVIEW")
print(text[:1500])
