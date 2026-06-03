from FlagEmbedding import FlagReranker

reranker = FlagReranker(
    "BAAI/bge-reranker-base"
)

query = "How do I register?"

docs = [
    "POST /auth/register",
    "GET /customer/products",
    "POST /customer/orders"
]

scores = reranker.compute_score(
    [[query, doc] for doc in docs]
)

for doc, score in zip(docs, scores):
    print(score, doc)