from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker

# Load ChromaDB
client = PersistentClient(path="./chroma_db")
collection = client.get_collection("sahachari_docs")

# Embedding model
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Reranker
reranker = FlagReranker("BAAI/bge-reranker-base")

while True:
    query = input("\nYou: ")

    if query.lower() == "exit":
        break

    # Create query embedding
    query_embedding = embedding_model.encode(query).tolist()

    # Retrieve top 5 chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    docs = results["documents"][0]

    # Rerank
    scores = reranker.compute_score(
        [[query, doc] for doc in docs]
    )

    # Sort by score
    ranked_docs = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    print("\n===== TOP 3 RERANKED RESULTS =====\n")

    for i, (doc, score) in enumerate(ranked_docs[:3], start=1):
        print(f"\nResult {i}")
        print(f"Score: {score:.4f}")
        print("-" * 50)
        print(doc[:1000])