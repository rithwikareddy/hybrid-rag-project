from pathlib import Path
import re
import sqlite3

import chromadb
from sentence_transformers import SentenceTransformer
import ollama


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_PATH = BASE_DIR / "data" / "warehouse.db"
VECTOR_DB_PATH = BASE_DIR / "vector_db"


# ============================================================
# MODELS / DATABASE
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5:3b"

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
collection = chroma_client.get_collection("company_documents")


# ============================================================
# QUESTION ROUTER
# ============================================================

def route_question(question):
    """
    Decide whether the question needs:
    SQL, RAG, BOTH, or is OUT_OF_SCOPE.
    """

    q = question.lower().strip()

    # --------------------------------------------------------
    # SQL / DATABASE KEYWORDS
    # --------------------------------------------------------

    sql_keywords = [
        "sales",
        "sale",
        "revenue",
        "quantity",
        "quantities",
        "units",
        "unit",
        "sold",
        "sell",
        "selling",
        "total",
        "average",
        "highest",
        "lowest",
        "least",
        "most",
        "maximum",
        "minimum",
        "region",
        "product",
        "products",
        "how many",
        "how much",
        "compare",
        "comparison",
        "amount",
        "money",
        "earnings",
    ]

    # --------------------------------------------------------
    # RAG / DOCUMENT KEYWORDS
    # --------------------------------------------------------

    rag_keywords = [
        "company",
        "technova",
        "return",
        "returns",
        "refund",
        "replacement",
        "replace",
        "policy",
        "policies",
        "support",
        "customer support",
        "customer service",
        "office",
        "location",
        "headquarters",
        "products",
        "smartphone",
        "smartphones",
        "laptop",
        "laptops",
        "tablet",
        "tablets",
        "headphone",
        "headphones",
        "business",
        "market",
        "markets",
        "hyderabad",
        "mumbai",
        "bangalore",
    ]

    has_sql = any(keyword in q for keyword in sql_keywords)
    has_rag = any(keyword in q for keyword in rag_keywords)

    # --------------------------------------------------------
    # BOTH
    # --------------------------------------------------------

    if has_sql and has_rag:
        return "BOTH"

    # --------------------------------------------------------
    # SQL
    # --------------------------------------------------------

    if has_sql:
        return "SQL"

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    if has_rag:
        return "RAG"

    # --------------------------------------------------------
    # OUT OF SCOPE
    # --------------------------------------------------------

    return "OUT_OF_SCOPE"


# ============================================================
# SQL GENERATION
# ============================================================

def generate_sql(question):
    """
    Generate SQL for database-related questions.
    Uses deterministic SQL for common questions.
    Falls back to Qwen for other valid database questions.
    """

    q = question.lower().strip()

    # --------------------------------------------------------
    # TOTAL SALES / REVENUE
    # --------------------------------------------------------

    if (
        ("total sales" in q)
        or ("total revenue" in q)
        or ("sales total" in q)
        or ("revenue total" in q)
    ):
        regions = ["hyderabad", "mumbai", "bangalore"]

        for region in regions:
            if region in q:
                return f"""
SELECT SUM(revenue) AS total_sales
FROM sales
WHERE region = '{region.title()}';
""".strip()

        return """
SELECT SUM(revenue) AS total_sales
FROM sales;
""".strip()

    # --------------------------------------------------------
    # TOTAL QUANTITY / UNITS
    # --------------------------------------------------------

    if (
        ("how many" in q)
        or ("total quantity" in q)
        or ("total units" in q)
        or ("quantity sold" in q)
        or ("units sold" in q)
    ):
        products = [
            "smartphone",
            "smartphones",
            "laptop",
            "laptops",
            "tablet",
            "tablets",
            "headphone",
            "headphones",
        ]

        for product in products:

            if product in q:

                if product in ["smartphone", "smartphones"]:
                    product_name = "Smartphone"

                elif product in ["laptop", "laptops"]:
                    product_name = "Laptop"

                elif product in ["tablet", "tablets"]:
                    product_name = "Tablet"

                else:
                    product_name = "Headphones"

                regions = ["hyderabad", "mumbai", "bangalore"]

                for region in regions:
                    if region in q:
                        return f"""
SELECT SUM(quantity) AS total_quantity
FROM sales
WHERE product = '{product_name}'
AND region = '{region.title()}';
""".strip()

                return f"""
SELECT SUM(quantity) AS total_quantity
FROM sales
WHERE product = '{product_name}';
""".strip()

        return """
SELECT SUM(quantity) AS total_quantity
FROM sales;
""".strip()

    # --------------------------------------------------------
    # LEAST / LOWEST / MINIMUM PRODUCT SOLD
    # --------------------------------------------------------

    if (
        "least" in q
        or "lowest" in q
        or "minimum" in q
    ) and (
        "product" in q
        or "products" in q
        or "sold" in q
        or "quantity" in q
        or "units" in q
    ):

        return """
SELECT product, SUM(quantity) AS total_quantity
FROM sales
GROUP BY product
HAVING SUM(quantity) = (
    SELECT MIN(total_quantity)
    FROM (
        SELECT SUM(quantity) AS total_quantity
        FROM sales
        GROUP BY product
    )
)
ORDER BY product;
""".strip()

    # --------------------------------------------------------
    # MOST / HIGHEST / MAXIMUM PRODUCT SOLD
    # --------------------------------------------------------

    if (
        "most" in q
        or "highest" in q
        or "maximum" in q
    ) and (
        "product" in q
        or "products" in q
        or "sold" in q
        or "quantity" in q
        or "units" in q
    ):

        return """
SELECT product, SUM(quantity) AS total_quantity
FROM sales
GROUP BY product
HAVING SUM(quantity) = (
    SELECT MAX(total_quantity)
    FROM (
        SELECT SUM(quantity) AS total_quantity
        FROM sales
        GROUP BY product
    )
)
ORDER BY product;
""".strip()

    # --------------------------------------------------------
    # SPECIFIC PRODUCT
    # --------------------------------------------------------

    product_name = None

    if "smartphone" in q:
        product_name = "Smartphone"

    elif "laptop" in q:
        product_name = "Laptop"

    elif "tablet" in q:
        product_name = "Tablet"

    elif "headphone" in q:
        product_name = "Headphones"

    if product_name:

        regions = ["hyderabad", "mumbai", "bangalore"]

        for region in regions:

            if region in q:

                return f"""
SELECT SUM(quantity) AS total_quantity
FROM sales
WHERE product = '{product_name}'
AND region = '{region.title()}';
""".strip()

        return f"""
SELECT SUM(quantity) AS total_quantity
FROM sales
WHERE product = '{product_name}';
""".strip()

    # --------------------------------------------------------
    # QWEN TEXT-TO-SQL FALLBACK
    # --------------------------------------------------------

    prompt = f"""
You are generating SQL for a SQLite database.

The database has only this table:

sales(
    sale_id INTEGER PRIMARY KEY,
    sale_date TEXT,
    region TEXT,
    product TEXT,
    quantity INTEGER,
    revenue REAL
)

User question:
{question}

Generate ONLY one SQLite SELECT query.

Do not use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or any other modifying statement.

Return only SQL.
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    sql = response["message"]["content"].strip()

    # Remove markdown SQL fences if Qwen adds them
    sql = sql.replace("```sql", "").replace("```", "").strip()

    return sql


# ============================================================
# SQL VALIDATION
# ============================================================

def validate_sql(sql):
    """
    Basic safety validation for generated SQL.
    """

    sql = sql.strip()

    # Only SELECT / WITH queries
    if not re.match(r"^(SELECT|WITH)\b", sql, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed.")

    # Prevent multiple statements
    if ";" in sql[:-1]:
        raise ValueError("Multiple SQL statements are not allowed.")

    # Prevent modifying operations
    forbidden = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "REPLACE",
        "TRUNCATE",
        "ATTACH",
        "DETACH",
    ]

    for word in forbidden:
        if re.search(rf"\b{word}\b", sql, re.IGNORECASE):
            raise ValueError("Unsafe SQL detected.")

    # Only allow our sales table
    tables = re.findall(
        r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)",
        sql,
        re.IGNORECASE
    )

    allowed_tables = {"sales"}

    for table in tables:
        if table.lower() not in allowed_tables:
            raise ValueError("Only the sales table is allowed.")

    return sql


# ============================================================
# SQL EXECUTION
# ============================================================

def execute_sql(sql):
    """
    Execute validated SQL and return columns + rows.
    """

    conn = sqlite3.connect(DATABASE_PATH)

    try:

        cursor = conn.cursor()

        cursor.execute(sql)

        rows = cursor.fetchall()

        columns = [
            description[0]
            for description in cursor.description
        ]

        return {
            "columns": columns,
            "rows": rows
        }

    finally:
        conn.close()


# ============================================================
# VALUE FORMATTING
# ============================================================

def format_value(column, value):

    if value is None:
        return "0"

    column_lower = column.lower()

    # Quantity / units / count
    if (
        "quantity" in column_lower
        or "units" in column_lower
        or "count" in column_lower
    ):

        try:
            return f"{int(value):,}"
        except:
            return str(value)

    # Revenue / sales
    if (
        "revenue" in column_lower
        or "sales" in column_lower
        or "amount" in column_lower
    ):

        try:
            return f"₹{float(value):,.0f}"
        except:
            return f"₹{value}"

    # Other numbers
    if isinstance(value, float):

        return f"{value:,.2f}"

    return str(value)


# ============================================================
# FORMAT SQL RESULT
# ============================================================

def format_sql_result(database_result):

    if not database_result:
        return ""

    columns = database_result["columns"]
    rows = database_result["rows"]

    if not rows:
        return "No matching data was found."

    lines = []

    for row in rows:

        parts = []

        for column, value in zip(columns, row):

            parts.append(
                f"{column} = {format_value(column, value)}"
            )

        lines.append(" | ".join(parts))

    return "\n".join(lines)


# ============================================================
# RAG RETRIEVAL
# ============================================================

def retrieve_documents(question, top_k=2):
    """
    Retrieve relevant company documents from ChromaDB.
    """

    query_embedding = embedding_model.encode(
        [question]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    documents = results.get("documents", [[]])[0]

    return documents


# ============================================================
# FINAL ANSWER
# ============================================================

def generate_final_answer(
    question,
    route,
    database_result=None,
    documents=None
):

    # --------------------------------------------------------
    # OUT OF SCOPE GUARDRAIL
    # --------------------------------------------------------

    if route == "OUT_OF_SCOPE":

        return (
            "Sorry, I can only answer questions related to "
            "the company's sales data and company information."
        )

    # --------------------------------------------------------
    # PREPARE DATABASE INFORMATION
    # --------------------------------------------------------

    database_text = ""

    if database_result:

        database_text = format_sql_result(
            database_result
        )

    # --------------------------------------------------------
    # PREPARE DOCUMENT INFORMATION
    # --------------------------------------------------------

    document_text = ""

    if documents:

        document_text = "\n\n".join(documents)

    # --------------------------------------------------------
    # QWEN FINAL ANSWER
    # --------------------------------------------------------

    prompt = f"""
You are the final answer assistant for a company question-answering system.

User question:
{question}

Question route:
{route}

DATABASE RESULT:
{database_text}

COMPANY DOCUMENTS:
{document_text}

IMPORTANT RULES:

1. Answer the user's question directly and clearly.
2. Use the database result as the source of truth for numerical answers.
3. Never invent numerical values.
4. Never change or recalculate database values.
5. For quantities, units, and counts, NEVER use the ₹ symbol.
6. For money or revenue, use the ₹ symbol.
7. If there is a tie, mention all tied products.
8. Use company documents for company information and policies.
9. Do not include irrelevant database information.
10. Do not answer questions using information that is unrelated to the user's question.
11. Keep the answer simple.
12. Do not mention internal technical details unless the user asks.

Give only the final answer to the user.
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"].strip()


# ============================================================
# MAIN HYBRID PIPELINE
# ============================================================

def ask_hybrid(question):

    # --------------------------------------------------------
    # STEP 1: ROUTE QUESTION
    # --------------------------------------------------------

    route = route_question(question)

    # --------------------------------------------------------
    # OUT OF SCOPE
    # --------------------------------------------------------

    if route == "OUT_OF_SCOPE":

        answer = generate_final_answer(
            question=question,
            route=route,
            database_result=None,
            documents=None
        )

        return {
            "question": question,
            "route": route,
            "sql": None,
            "database_result": None,
            "documents": None,
            "answer": answer
        }

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    sql = None
    database_result = None

    if route in ["SQL", "BOTH"]:

        sql = generate_sql(question)

        sql = validate_sql(sql)

        database_result = execute_sql(sql)

    # --------------------------------------------------------
    # DOCUMENTS
    # --------------------------------------------------------

    documents = None

    if route in ["RAG", "BOTH"]:

        documents = retrieve_documents(question)

    # --------------------------------------------------------
    # FINAL ANSWER
    # --------------------------------------------------------

    answer = generate_final_answer(
        question=question,
        route=route,
        database_result=database_result,
        documents=documents
    )

    return {
        "question": question,
        "route": route,
        "sql": sql,
        "database_result": database_result,
        "documents": documents,
        "answer": answer
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    question = input(
        "\nAsk a question: "
    )

    result = ask_hybrid(question)

    print("\n==============================")
    print("ROUTE")
    print("==============================")
    print(result["route"])

    print("\n==============================")
    print("SQL")
    print("==============================")
    print(result["sql"])

    print("\n==============================")
    print("DATABASE RESULT")
    print("==============================")
    print(result["database_result"])

    print("\n==============================")
    print("DOCUMENTS")
    print("==============================")
    print(result["documents"])

    print("\n==============================")
    print("FINAL ANSWER")
    print("==============================")
    print(result["answer"])