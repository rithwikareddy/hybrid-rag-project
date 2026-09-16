from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_DB_FOLDER = PROJECT_ROOT / "vector_db"


# Load embedding model
print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# Connect to ChromaDB
client = chromadb.PersistentClient(
    path=str(VECTOR_DB_FOLDER)
)

collection = client.get_collection(
    name="company_documents"
)


# Ask the user a question
query = input("\nEnter your question: ")


# Convert question into embedding
query_embedding = embedding_model.encode(
    query
).tolist()


# Search ChromaDB
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)


# Display results
print("\n==============================")
print("RETRIEVED INFORMATION")
print("==============================")

for i, document in enumerate(results["documents"][0]):

    print(f"\nResult {i + 1}:")
    print(document)

    print(
        f"Source: {results['metadatas'][0][i]['source']}"
    )