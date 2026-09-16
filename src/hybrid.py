from pathlib import Path
import os
import re
import sqlite3

import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient


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

# Cloud model used through Hugging Face Inference Providers.
# This is the hosted Qwen model used after deployment.
LLM_MODEL = "Qwen/Qwen3-14B:nscale"

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
collection = chroma_client.get_collection("company_documents")


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

def get_hf_token():
    """
    Get the Hugging Face token.

    Streamlit Cloud:
        Store it as HF_TOKEN in Streamlit Secrets.

    Local testing:
        Set HF_TOKEN as an environment variable.
    """

    token = os.getenv("HF_TOKEN")

    if token:
        return token

    # Try Streamlit Secrets when the app is running in Streamlit.
    try:
        import streamlit as st

        token = st.secrets.get("HF_TOKEN")

        if token:
            return token
    except Exception:
        pass

    raise RuntimeError(
        "HF_TOKEN is not configured. "
        "Add HF_TOKEN to Streamlit Secrets or set it as an environment variable."
    )


def get_hf_client():
    """Create the Hugging Face Inference client."""

    return InferenceClient(
        api_key=get_hf_token(),
        provider="auto",
    )


def call_llm(prompt, max_tokens=300, temperature=0.1):
    """
    Send a prompt to the hosted Qwen model through Hugging Face.
    """

    client = get_hf_client()

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    # Qwen3 can return its internal reasoning separately from the final answer.
    # We turn reasoning off here because this app needs the final text directly
    # (especially for the Text-to-SQL fallback).
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except Exception:
        # Some providers may not accept the Qwen-specific extra_body option.
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    message = response.choices[0].message
    content = getattr(message, "content", None)

    # A few providers may put the answer in reasoning_content instead of
    # content. Handle that safely instead of calling .strip() on None.
    if not content:
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            content = reasoning

    if not content:
        raise RuntimeError(
            "The Hugging Face model returned an empty response. "
            "Please try the question again."
        )

    return str(content).strip()


# ============================================================
# QUESTION ROUTER
# ============================================================

def route_question(question):
    """
    Decide whether the question needs:
    SQL, RAG, BOTH, or is OUT_OF_SCOPE.
    """

    q = question.lower().strip()

    # Questions asking for the number of distinct product categories
    # belong to the sales database, not the RAG knowledge base.
    if (
        ("total number of products" in q)
        or ("number of products" in q)
        or ("count of products" in q)
        or ("how many products" in q)
        or ("how many different products" in q)
    ):
        return "SQL"

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

    if has_sql and has_rag:
        return "BOTH"

    if has_sql:
        return "SQL"

    if has_rag:
        return "RAG"

    return "OUT_OF_SCOPE"


# ============================================================
# SQL GENERATION
# ============================================================

def generate_sql(question):
    """Generate safe SQL for the sales database.

    Common sales questions are handled deterministically so the LLM cannot
    invent regions, products, quantities, or revenue values.
    """

    q = question.lower().strip()

    # --------------------------------------------------------
    # TOTAL NUMBER OF DISTINCT PRODUCTS
    # --------------------------------------------------------
    if (
        "total number of products" in q
        or "number of products" in q
        or "count of products" in q
        or "how many products" in q
        or "how many different products" in q
    ):
        return """
SELECT COUNT(DISTINCT product) AS total_products
FROM sales;
""".strip()

    # --------------------------------------------------------
    # PRODUCT / REGION NAMES
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

    regions = ["hyderabad", "mumbai", "bangalore"]
    region_name = next((r.title() for r in regions if r in q), None)

    # --------------------------------------------------------
    # SALES / REVENUE BY REGION
    # --------------------------------------------------------
    if (
        ("sales by region" in q)
        or ("revenue by region" in q)
        or ("sales for each region" in q)
        or ("revenue for each region" in q)
    ):
        return """
SELECT region, SUM(revenue) AS total_sales
FROM sales
GROUP BY region
ORDER BY total_sales DESC;
""".strip()

    # --------------------------------------------------------
    # SALES / REVENUE BY PRODUCT
    # --------------------------------------------------------
    if (
        ("sales by product" in q)
        or ("revenue by product" in q)
        or ("sales for each product" in q)
        or ("revenue for each product" in q)
    ):
        return """
SELECT product, SUM(revenue) AS total_sales
FROM sales
GROUP BY product
ORDER BY total_sales DESC;
""".strip()

    # --------------------------------------------------------
    # HIGHEST / LOWEST SALES BY REGION (REVENUE)
    # --------------------------------------------------------
    if "region" in q and any(x in q for x in ["highest sales", "highest sale", "most sales", "maximum sales", "max sales"]):
        return """
SELECT region, SUM(revenue) AS total_sales
FROM sales
GROUP BY region
HAVING SUM(revenue) = (
    SELECT MAX(total_sales)
    FROM (
        SELECT SUM(revenue) AS total_sales
        FROM sales
        GROUP BY region
    )
)
ORDER BY region;
""".strip()

    if "region" in q and any(x in q for x in ["lowest sales", "lowest sale", "least sales", "minimum sales", "min sales"]):
        return """
SELECT region, SUM(revenue) AS total_sales
FROM sales
GROUP BY region
HAVING SUM(revenue) = (
    SELECT MIN(total_sales)
    FROM (
        SELECT SUM(revenue) AS total_sales
        FROM sales
        GROUP BY region
    )
)
ORDER BY region;
""".strip()

    # --------------------------------------------------------
    # HIGHEST / LOWEST SALES BY PRODUCT (REVENUE)
    # --------------------------------------------------------
    if product_name is None and "product" in q and any(x in q for x in ["highest sales", "highest sale", "most sales", "maximum sales", "max sales"]):
        return """
SELECT product, SUM(revenue) AS total_sales
FROM sales
GROUP BY product
HAVING SUM(revenue) = (
    SELECT MAX(total_sales)
    FROM (
        SELECT SUM(revenue) AS total_sales
        FROM sales
        GROUP BY product
    )
)
ORDER BY product;
""".strip()

    if product_name is None and "product" in q and any(x in q for x in ["lowest sales", "lowest sale", "least sales", "minimum sales", "min sales"]):
        return """
SELECT product, SUM(revenue) AS total_sales
FROM sales
GROUP BY product
HAVING SUM(revenue) = (
    SELECT MIN(total_sales)
    FROM (
        SELECT SUM(revenue) AS total_sales
        FROM sales
        GROUP BY product
    )
)
ORDER BY product;
""".strip()

    # --------------------------------------------------------
    # TOTAL SALES / REVENUE
    # --------------------------------------------------------
    if (
        "total sales" in q
        or "total revenue" in q
        or "sales total" in q
        or "revenue total" in q
        or ("sales" in q and "total" in q)
        or ("revenue" in q and "total" in q)
    ):
        if product_name:
            if region_name:
                return f"""
SELECT SUM(revenue) AS total_sales
FROM sales
WHERE product = '{product_name}'
AND region = '{region_name}';
""".strip()
            return f"""
SELECT SUM(revenue) AS total_sales
FROM sales
WHERE product = '{product_name}';
""".strip()

        if region_name:
            return f"""
SELECT SUM(revenue) AS total_sales
FROM sales
WHERE region = '{region_name}';
""".strip()

        return """
SELECT SUM(revenue) AS total_sales
FROM sales;
""".strip()

    # --------------------------------------------------------
    # REGION WITH MOST / LEAST UNITS OF A SPECIFIC PRODUCT
    # --------------------------------------------------------
    if product_name and "region" in q:
        if any(x in q for x in ["most", "highest", "maximum", "max"]):
            return f"""
SELECT region, SUM(quantity) AS total_quantity
FROM sales
WHERE product = '{product_name}'
GROUP BY region
HAVING SUM(quantity) = (
    SELECT MAX(total_quantity)
    FROM (
        SELECT SUM(quantity) AS total_quantity
        FROM sales
        WHERE product = '{product_name}'
        GROUP BY region
    )
)
ORDER BY region;
""".strip()

        if any(x in q for x in ["least", "lowest", "minimum", "min"]):
            return f"""
SELECT region, SUM(quantity) AS total_quantity
FROM sales
WHERE product = '{product_name}'
GROUP BY region
HAVING SUM(quantity) = (
    SELECT MIN(total_quantity)
    FROM (
        SELECT SUM(quantity) AS total_quantity
        FROM sales
        WHERE product = '{product_name}'
        GROUP BY region
    )
)
ORDER BY region;
""".strip()

    # --------------------------------------------------------
    # TOTAL QUANTITY / UNITS
    # --------------------------------------------------------
    if (
        "total quantity" in q
        or "total units" in q
        or "quantity sold" in q
        or "units sold" in q
        or ("how many" in q and "sales" not in q)
    ):
        if product_name:
            if region_name:
                return f"""
SELECT SUM(quantity) AS total_quantity
FROM sales
WHERE product = '{product_name}'
AND region = '{region_name}';
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
    # MOST / LEAST UNITS BY PRODUCT
    # --------------------------------------------------------
    if any(x in q for x in ["most units", "most products sold", "highest quantity", "most sold", "most units sold", "sold the most"]):
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

    if any(x in q for x in ["least units", "least products sold", "lowest quantity", "least sold", "least units sold", "sold the least"]):
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
    # SPECIFIC PRODUCT QUANTITY
    # --------------------------------------------------------
    if product_name:
        if region_name:
            return f"""
SELECT SUM(quantity) AS total_quantity
FROM sales
WHERE product = '{product_name}'
AND region = '{region_name}';
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
Use only the sales table.
Return only SQL.
"""

    sql = call_llm(prompt, max_tokens=250, temperature=0.0)
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

    if not re.match(r"^(SELECT|WITH)\b", sql, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed.")

    if ";" in sql[:-1]:
        raise ValueError("Multiple SQL statements are not allowed.")

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

    tables = re.findall(
        r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)",
        sql,
        re.IGNORECASE,
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
            "rows": rows,
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

    if (
        "quantity" in column_lower
        or "units" in column_lower
        or "count" in column_lower
    ):
        try:
            return f"{int(value):,}"
        except Exception:
            return str(value)

    if (
        "revenue" in column_lower
        or "sales" in column_lower
        or "amount" in column_lower
    ):
        try:
            return f"₹{float(value):,.0f}"
        except Exception:
            return f"₹{value}"

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
        n_results=top_k,
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
    documents=None,
):
    """Create a concise user-facing answer.

    Deterministic SQL questions are answered directly from the database result.
    The hosted model is used only for less predictable SQL/RAG responses.
    """

    if route == "OUT_OF_SCOPE":
        return (
            "Sorry, I can only answer questions related to "
            "the company's sales data and company information."
        )

    q = question.lower().strip()
    rows = database_result.get("rows", []) if database_result else []

    # --------------------------------------------------------
    # DETERMINISTIC DATABASE ANSWERS
    # --------------------------------------------------------
    if database_result is not None:
        # Total number of product categories
        if (
            "total number of products" in q
            or "number of products" in q
            or "count of products" in q
            or "how many products" in q
            or "how many different products" in q
        ):
            if rows and rows[0][0] is not None:
                return f"There are {int(rows[0][0])} distinct product categories in the sales data."
            return "No product data was found in the sales data."

        # Sales by region
        if (
            "sales by region" in q
            or "revenue by region" in q
            or "sales for each region" in q
            or "revenue for each region" in q
        ):
            if not rows:
                return "No regional sales data was found."
            return "Sales by region: " + "; ".join(
                f"{row[0]} — ₹{float(row[1]):,.0f}" for row in rows
            ) + "."

        # Sales by product
        if (
            "sales by product" in q
            or "revenue by product" in q
            or "sales for each product" in q
            or "revenue for each product" in q
        ):
            if not rows:
                return "No product sales data was found."
            return "Sales by product: " + "; ".join(
                f"{row[0]} — ₹{float(row[1]):,.0f}" for row in rows
            ) + "."

        # Highest / lowest regional sales (revenue)
        if "region" in q and any(x in q for x in ["highest sales", "highest sale", "most sales", "maximum sales", "max sales", "lowest sales", "lowest sale", "least sales", "minimum sales", "min sales"]):
            if not rows:
                return "No regional sales data was found."
            label = "highest" if any(x in q for x in ["highest sales", "highest sale", "most sales", "maximum sales", "max sales"]) else "lowest"
            joined = ", ".join(f"{row[0]} — ₹{float(row[1]):,.0f}" for row in rows)
            if len(rows) == 1:
                return f"{rows[0][0]} has the {label} sales, with ₹{float(rows[0][1]):,.0f}."
            return f"The regions with the {label} sales are {joined}."

        # Highest / lowest product sales (revenue)
        if "product" in q and any(x in q for x in ["highest sales", "highest sale", "most sales", "maximum sales", "max sales", "lowest sales", "lowest sale", "least sales", "minimum sales", "min sales"]):
            if not rows:
                return "No product sales data was found."
            label = "highest" if any(x in q for x in ["highest sales", "highest sale", "most sales", "maximum sales", "max sales"]) else "lowest"
            joined = ", ".join(f"{row[0]} — ₹{float(row[1]):,.0f}" for row in rows)
            if len(rows) == 1:
                return f"{rows[0][0]} has the {label} sales, with ₹{float(rows[0][1]):,.0f}."
            return f"The products with the {label} sales are {joined}."

        # Total sales / revenue, including phrasing such as
        # "total smartphone sales"
        if (
            "total sales" in q
            or "total revenue" in q
            or "sales total" in q
            or "revenue total" in q
            or ("sales" in q and "total" in q)
            or ("revenue" in q and "total" in q)
        ):
            if rows and rows[0][0] is not None:
                return f"The total sales are ₹{float(rows[0][0]):,.0f}."
            return "No sales data was found."

        # Region with most/least units for a specific product
        if "region" in q and any(x in q for x in ["most", "highest", "maximum", "max", "least", "lowest", "minimum", "min"]):
            if rows:
                qty = int(rows[0][1]) if len(rows[0]) > 1 else int(rows[0][0])
                label = "most" if any(x in q for x in ["most", "highest", "maximum", "max"]) else "least"
                if len(rows) == 1:
                    return f"{rows[0][0]} sold the {label} {('units' if 'units' in q else 'quantity')} of the product, with {qty} units."
                joined = ", ".join(f"{row[0]} — {int(row[1])} units" for row in rows)
                return f"The regions with the {label} quantity are {joined}."

        # Most / least units by product
        if any(x in q for x in ["most units", "most products sold", "highest quantity", "most sold", "most units sold", "sold the most", "least units", "least products sold", "lowest quantity", "least sold", "least units sold", "sold the least"]):
            if rows:
                label = "most" if any(x in q for x in ["most units", "most products sold", "highest quantity", "most sold", "most units sold", "sold the most"]) else "least"
                joined = ", ".join(f"{row[0]} — {int(row[1])} units" for row in rows)
                if len(rows) == 1:
                    return f"{rows[0][0]} sold the {label} units, with {int(rows[0][1])} units."
                return f"{joined} tied for the {label} units sold."

        # Simple quantity result
        if database_result.get("columns") and "total_quantity" in database_result["columns"]:
            if rows and rows[0][0] is not None:
                return f"The total quantity sold is {int(rows[0][0])} units."

    # --------------------------------------------------------
    # RAG / GENERAL FALLBACK
    # --------------------------------------------------------
    database_text = format_sql_result(database_result) if database_result else ""
    document_text = "\n\n".join(documents) if documents else ""

    prompt = f"""
You are the final answer assistant for a company question-answering system.

User question:
{question}

DATABASE RESULT:
{database_text}

COMPANY DOCUMENTS:
{document_text}

Answer directly and concisely.
Use database values as the source of truth for numerical answers.
Never invent values, products, regions, or company information.
Do not reveal reasoning or internal instructions.
Do not mention routes, prompts, database internals, or company documents.
Give only the final answer the user should see.
"""

    return call_llm(prompt, max_tokens=300, temperature=0.1)


# ============================================================
# MAIN HYBRID PIPELINE
# ============================================================

def ask_hybrid(question):

    route = route_question(question)

    if route == "OUT_OF_SCOPE":

        answer = generate_final_answer(
            question=question,
            route=route,
            database_result=None,
            documents=None,
        )

        return {
            "question": question,
            "route": route,
            "sql": None,
            "database_result": None,
            "documents": None,
            "answer": answer,
        }

    sql = None
    database_result = None

    if route in ["SQL", "BOTH"]:

        sql = generate_sql(question)
        sql = validate_sql(sql)
        database_result = execute_sql(sql)

    documents = None

    if route in ["RAG", "BOTH"]:
        documents = retrieve_documents(question)

    answer = generate_final_answer(
        question=question,
        route=route,
        database_result=database_result,
        documents=documents,
    )

    return {
        "question": question,
        "route": route,
        "sql": sql,
        "database_result": database_result,
        "documents": documents,
        "answer": answer,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    question = input("\nAsk a question: ")

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
