from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import os

# Files to load
doc_files = [
    "docs/customer_api.txt",
    "docs/storekeeper_api.txt",
    "docs/delivery_api.txt",
    "docs/superadmin_api.txt"
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

embedding_model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="sahachari_docs"
)

chunk_count = 0

for file in doc_files:

    print(f"Processing {file}")

    with open(file, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = splitter.split_text(text)

    for i, chunk in enumerate(chunks):

        embedding = embedding_model.encode(chunk).tolist()

        collection.add(
            ids=[f"{os.path.basename(file)}_{i}"],
            documents=[chunk],
            metadatas=[{"source": file}],
            embeddings=[embedding]
        )

        chunk_count += 1

print(f"\nTotal chunks stored: {chunk_count}")
print("Knowledge Base Created Successfully")