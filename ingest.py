import os
import re
import hashlib
import logging

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ingest")

# ---------------------------------------------------------------------------
# Config — must match chatbot_v3.py
# ---------------------------------------------------------------------------

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR    = os.path.join(BASE_DIR, "docs")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION  = "sahachari_docs"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

# Chunking settings
CHUNK_SIZE    = 300   # target words per chunk (reduced for Q&A style docs)
CHUNK_OVERLAP = 40    # words of overlap between chunks

# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def _uid(text: str, idx: int) -> str:
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"chunk_{idx:05d}_{h}"


def split_by_qa_sections(text: str) -> list[str]:
    """
    Smart splitter that handles both:
    1. Markdown ## headings (Q&A style docs like sahachari_*.md)
    2. === section headers (knowledge base style docs)
    Each heading + its content becomes one chunk (good for RAG retrieval).
    """
    # Match ## headings (Q&A style) or === section separators
    header_re = re.compile(
        r"(?:^#{1,3}\s+.+$|^={10,}.*?={10,}$)",
        re.MULTILINE,
    )

    positions = [m.start() for m in header_re.finditer(text)]

    # No headers found — return whole text as one chunk
    if not positions:
        return [text.strip()]

    sections = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)

    return sections


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Secondary splitter: breaks a long section into word-based sliding windows.
    Short sections (most Q&A entries) are returned as-is.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def load_documents(docs_dir: str) -> list[dict]:
    """
    Load all .txt and .md files from docs_dir.
    Returns list of {"source": filename, "text": content}.
    """
    docs = []
    if not os.path.isdir(docs_dir):
        log.error(f"docs/ directory not found at {docs_dir}")
        return docs

    for fname in sorted(os.listdir(docs_dir)):
        if not (fname.endswith(".txt") or fname.endswith(".md")):
            continue
        fpath = os.path.join(docs_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                docs.append({"source": fname, "text": content})
                log.info(f"Loaded: {fname} ({len(content):,} chars)")
        except Exception as e:
            log.warning(f"Could not read {fname}: {e}")

    return docs


def build_chunks(docs: list[dict]) -> list[dict]:
    """
    Split each document into chunks.
    For Q&A markdown files, each ## section becomes its own chunk.
    Returns list of {"id": uid, "text": chunk, "source": fname}.
    """
    all_chunks = []
    idx        = 0

    for doc in docs:
        sections = split_by_qa_sections(doc["text"])
        log.info(f"{doc['source']} -> {len(sections)} section(s)")

        for section in sections:
            # Clean up excessive whitespace
            section = re.sub(r"\n{3,}", "\n\n", section).strip()
            if len(section.split()) < 8:
                continue  # skip tiny fragments

            sub_chunks = chunk_text(section)
            for chunk in sub_chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue
                all_chunks.append({
                    "id":     _uid(chunk, idx),
                    "text":   chunk,
                    "source": doc["source"],
                })
                idx += 1

    log.info(f"Total chunks: {len(all_chunks)}")
    return all_chunks


# ---------------------------------------------------------------------------
# Embedding + ChromaDB
# ---------------------------------------------------------------------------

def ingest(docs_dir: str = DOCS_DIR, chroma_path: str = CHROMA_PATH):
    # 1. Load docs
    docs = load_documents(docs_dir)
    if not docs:
        log.error("No documents found. Place .txt or .md files in the docs/ folder.")
        return

    # 2. Chunk
    chunks = build_chunks(docs)
    if not chunks:
        log.error("No usable chunks generated.")
        return

    # 3. Load embedding model
    log.info(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    # 4. Embed in batches
    texts      = [c["text"] for c in chunks]
    batch_size = 64
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs  = model.encode(batch, show_progress_bar=False).tolist()
        embeddings.extend(vecs)
        log.info(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")

    # 5. Connect to ChromaDB
    log.info(f"Connecting to ChromaDB at {chroma_path}")
    client = PersistentClient(path=chroma_path)

    # Delete and recreate collection for a clean rebuild
    try:
        client.delete_collection(COLLECTION)
        log.info(f"Deleted existing collection '{COLLECTION}'")
    except Exception:
        pass

    col = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    log.info(f"Created collection '{COLLECTION}'")

    # 6. Upsert in batches
    upsert_batch = 100
    for i in range(0, len(chunks), upsert_batch):
        batch_chunks = chunks[i : i + upsert_batch]
        col.upsert(
            ids        = [c["id"]     for c in batch_chunks],
            documents  = [c["text"]   for c in batch_chunks],
            embeddings = embeddings[i : i + upsert_batch],
            metadatas  = [{"source": c["source"]} for c in batch_chunks],
        )
        log.info(f"Upserted {min(i + upsert_batch, len(chunks))}/{len(chunks)} chunks")

    # 7. Verify
    count = col.count()
    log.info(f"Ingestion complete. ChromaDB now has {count} chunks in '{COLLECTION}'.")

    # 8. Sanity tests
    test_queries = [
        "what is sahachari",
        "refund policy",
        "vegetables for biryani",
        "dishwash service",
        "track my order",
        "rotten vegetables complaint",
    ]
    log.info("Running sanity queries...")
    for q in test_queries:
        test_vec = model.encode(q).tolist()
        results  = col.query(query_embeddings=[test_vec], n_results=1)
        if results and results.get("documents") and results["documents"][0]:
            snippet = results["documents"][0][0][:100].replace("\n", " ")
            log.info(f"  '{q}' -> {snippet}...")
        else:
            log.warning(f"  '{q}' -> NO RESULT (check your docs)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ingest()
