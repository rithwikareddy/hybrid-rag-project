from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
VECTOR_DB_DIR = BASE_DIR / "vector_db"


# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


# Connect to ChromaDB
client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

# Delete old collection so the knowledge base is rebuilt
try:
    client.delete_collection("company_documents")
    print("Old knowledge base removed.")
except Exception:
    pass

# Create fresh collection
collection = client.create_collection("company_documents")


# Read documents
documents = []

for file_path in DOCUMENTS_DIR.glob("*.txt"):
    text = file_path.read_text(encoding="utf-8")

    # Split document into smaller chunks
    chunk_size = 500

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()

        if chunk:
            documents.append({
                "text": chunk,
                "source": file_path.name
            })


# Generate embeddings
texts = [doc["text"] for doc in documents]

print(f"Creating embeddings for {len(texts)} chunks...")

embeddings = model.encode(texts).tolist()


# Store in ChromaDB
collection.add(
    ids=[f"doc_{i}" for i in range(len(texts))],
    documents=texts,
    embeddings=embeddings,
    metadatas=[
        {"source": doc["source"]}
        for doc in documents
    ]
)


print("\n====================================")
print("   RAG KNOWLEDGE BASE CREATED")
print("====================================")
print(f"Documents : {len(documents)} chunks")
print("Collection: company_documents")
print("====================================")