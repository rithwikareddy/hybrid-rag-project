from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
import ollama


# ==========================================
# 1. Project paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_DB_FOLDER = PROJECT_ROOT / "vector_db"


# ==========================================
# 2. Load embedding model
# ==========================================

print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# ==========================================
# 3. Connect to ChromaDB
# ==========================================

client = chromadb.PersistentClient(
    path=str(VECTOR_DB_FOLDER)
)

collection = client.get_collection(
    name="company_documents"
)


# ==========================================
# 4. Get question
# ==========================================

query = input("\nEnter your question: ")


# ==========================================
# 5. Convert question into embedding
# ==========================================

query_embedding = embedding_model.encode(
    query
).tolist()


# ==========================================
# 6. Retrieve relevant documents
# ==========================================

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)


retrieved_documents = results["documents"][0]


# ==========================================
# 7. Combine retrieved documents
# ==========================================

context = "\n\n".join(retrieved_documents)


# ==========================================
# 8. Create prompt for Qwen
# ==========================================

prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not present in the context, say:
"I don't have enough information in the provided documents."

Do not make up information.

CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""


# ==========================================
# 9. Send prompt to local Qwen
# ==========================================

print("\nGenerating answer...\n")

response = ollama.chat(
    model="qwen2.5:3b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# ==========================================
# 10. Display final answer
# ==========================================

answer = response["message"]["content"]


print("==============================")
print("FINAL ANSWER")
print("==============================")

print(answer)