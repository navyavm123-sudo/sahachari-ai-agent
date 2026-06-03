from sentence_transformers import SentenceTransformer
import chromadb

# Same embedding model used during ingestion
embedding_model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

# Load ChromaDB
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "sahachari_docs"
)

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    print("\nTop Retrieved Chunks:\n")

    for i, doc in enumerate(results["documents"][0], start=1):

        print(f"\n----- Result {i} -----\n")
        print(doc)