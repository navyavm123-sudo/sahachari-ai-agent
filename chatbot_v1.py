from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
import chromadb

# Embedding model
embedding_model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

# Reranker model
reranker = FlagReranker(
    "BAAI/bge-reranker-base"
)

# ChromaDB
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

    # Step 1: Convert query to embedding
    query_embedding = embedding_model.encode(
        question
    ).tolist()

    # Step 2: Retrieve top 10 chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )

    docs = results["documents"][0]

    # Step 3: Rerank
    pairs = [[question, doc] for doc in docs]

    scores = reranker.compute_score(pairs)

    ranked_docs = sorted(
        zip(scores, docs),
        reverse=True
    )

    print("\n===== TOP 3 RERANKED RESULTS =====\n")

    for i, (score, doc) in enumerate(ranked_docs[:3], start=1):

        print(f"\nResult {i}")
        print(f"Score: {score:.4f}")
        print("-" * 50)
        print(doc[:1000])
        print("\n")