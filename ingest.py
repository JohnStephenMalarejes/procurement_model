import os
import chromadb
from sentence_transformers import SentenceTransformer
from app.config import CHROMA_DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME

DATA_FILE = "data/procurement_policies.txt"


def seed_vector_db():
    if not os.path.exists(DATA_FILE):
        print(f"File not found: {DATA_FILE}")
        return

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Split policies by double newline
    chunks = [chunk.strip() for chunk in raw_text.split("\n\n") if chunk.strip()]
    ids = [f"policy_chunk_{i+1}" for i in range(len(chunks))]

    print(f"Encoding {len(chunks)} chunks using {EMBEDDING_MODEL_NAME}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embedder.encode(chunks).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    collection.upsert(documents=chunks, embeddings=embeddings, ids=ids)
    print(f"Successfully indexed {collection.count()} policy chunks into ChromaDB.")


if __name__ == "__main__":
    seed_vector_db()