import ollama


def classify_question(question):

    prompt = f"""
You are a question classification system.

Classify the user's question into exactly ONE of these categories:

SQL
RAG
BOTH

Use these rules:

SQL:
Use SQL when the question requires numerical,
structured, or database information such as:
sales, revenue, quantity, dates, regions, products,
totals, averages, counts, maximums, minimums.

RAG:
Use RAG when the question asks about information
contained in documents, policies, descriptions,
company information, or general text.

BOTH:
Use BOTH when the question requires information
from both the database and documents.

Return ONLY one word:
SQL
RAG
or
BOTH

User question:
{question}
"""

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    result = response["message"]["content"].strip().upper()

    if "BOTH" in result:
        return "BOTH"

    if "SQL" in result:
        return "SQL"

    if "RAG" in result:
        return "RAG"

    return "RAG"


# Test the router

question = input("\nEnter your question: ")

route = classify_question(question)

print()
print("==============================")
print("QUERY ROUTER")
print("==============================")

print(f"Question: {question}")
print(f"Selected route: {route}")