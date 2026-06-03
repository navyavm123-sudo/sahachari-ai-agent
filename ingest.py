from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# Read document

with open("docs/customer_api.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Chunking

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)

print(f"Total chunks: {len(chunks)}")

# Embedding model

embedding_model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

# ChromaDB

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="sahachari_docs"
)

# Store chunks

for i, chunk in enumerate(chunks):

    embedding = embedding_model.encode(
        chunk
    ).tolist()

    collection.add(
        ids=[str(i)],
        documents=[chunk],
        embeddings=[embedding]
    )

print("Knowledge Base Created Successfully")