import sqlite3
from pathlib import Path
import ollama


# ============================================================
# 1. DATABASE LOCATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_PATH = PROJECT_ROOT / "data" / "warehouse.db"


# ============================================================
# 2. DATABASE SCHEMA
# ============================================================

SCHEMA = """
Table: sales

Columns:

sale_id
- Unique ID of each sale

sale_date
- Date on which the sale happened

region
- Sales region such as Hyderabad, Mumbai, Bangalore

product
- Product sold

quantity
- Number of units sold

revenue
- Revenue generated from the sale
"""


# ============================================================
# 3. GENERATE SQL FROM NATURAL LANGUAGE
# ============================================================

def generate_sql(question):

    prompt = f"""
You are an expert SQLite SQL generator.

You must convert the user's natural-language question
into ONE correct SQLite SQL query.

DATABASE SCHEMA:

{SCHEMA}


IMPORTANT SQL RULES:

1. If the user asks for "total sales",
   use SUM(revenue).

2. If the user asks for "total revenue",
   use SUM(revenue).

3. If the user asks for "total quantity",
   use SUM(quantity).

4. If the user asks for "average sales" or
   "average revenue",
   use AVG(revenue).

5. If the user asks "how many sales",
   use COUNT(*).

6. If the user asks for the highest revenue,
   use MAX(revenue).

7. If the user asks for the lowest revenue,
   use MIN(revenue).

8. If the user mentions a region,
   filter using the region column.

9. If the user mentions a product,
   filter using the product column.

10. If the user mentions a year,
    filter using sale_date.

11. When the user asks for a TOTAL, do NOT return
    individual rows. Use an aggregate such as SUM().

12. When the user asks for an AVERAGE, use AVG().

13. When the user asks for a COUNT, use COUNT().

14. Use only the sales table.

15. Use only columns that exist in the schema.

16. Do not invent tables or columns.

17. Return ONLY the SQL query.

18. Do NOT use markdown.

19. Do NOT explain the SQL.


EXAMPLE 1:

User:
What were the total sales in Hyderabad?

SQL:
SELECT SUM(revenue) AS total_sales
FROM sales
WHERE region = 'Hyderabad';


EXAMPLE 2:

User:
What were the total sales in Mumbai?

SQL:
SELECT SUM(revenue) AS total_sales
FROM sales
WHERE region = 'Mumbai';


EXAMPLE 3:

User:
How many sales happened in Hyderabad?

SQL:
SELECT COUNT(*) AS number_of_sales
FROM sales
WHERE region = 'Hyderabad';


EXAMPLE 4:

User:
What was the average revenue in Hyderabad?

SQL:
SELECT AVG(revenue) AS average_revenue
FROM sales
WHERE region = 'Hyderabad';


EXAMPLE 5:

User:
What is the total quantity of AI Product sold?

SQL:
SELECT SUM(quantity) AS total_quantity
FROM sales
WHERE product = 'AI Product';


USER QUESTION:

{question}

RETURN ONLY THE SQL QUERY.
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


    sql = response["message"]["content"].strip()


    # Remove markdown if Qwen accidentally adds it
    sql = sql.replace("```sql", "")
    sql = sql.replace("```", "")

    sql = sql.strip()


    return sql


# ============================================================
# 4. EXECUTE SQL
# ============================================================

def execute_sql(sql):

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()


    cursor.execute(sql)


    results = cursor.fetchall()


    # Get column names
    if cursor.description:

        column_names = [
            description[0]
            for description in cursor.description
        ]

    else:

        column_names = []


    connection.close()


    return column_names, results


# ============================================================
# 5. GENERATE NATURAL LANGUAGE ANSWER
# ============================================================

def generate_answer(question, columns, results):

    prompt = f"""
You are a professional data analyst.

The user asked:

{question}


The SQLite database returned this EXACT result:

Columns:
{columns}

Results:
{results}


IMPORTANT RULES:

1. The database result is the SOURCE OF TRUTH.

2. Do NOT invent any numbers.

3. Do NOT change any numbers from the database result.

4. Do NOT incorrectly recalculate the result.

5. If the database returns 295000, the answer must
   say 295000, not 155000.

6. When talking about company sales or revenue,
   use the Indian Rupee symbol ₹.

7. Format large numbers using Indian-style commas.
   Example:
   295000 → ₹2,95,000

8. Answer in simple natural English.

9. Keep the answer concise.

10. Do not mention SQL unless the user asks about SQL.

11. Do not make assumptions that are not present
    in the database result.


Now answer the user's question using ONLY
the database result.
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


    return response["message"]["content"]


# ============================================================
# 6. MAIN PROGRAM
# ============================================================

def main():

    print()
    print("======================================")
    print("      LOCAL TEXT-TO-SQL SYSTEM")
    print("======================================")


    question = input(
        "\nEnter your question: "
    )


    # --------------------------------------------------------
    # Generate SQL
    # --------------------------------------------------------

    print("\nGenerating SQL...\n")


    sql = generate_sql(question)


    print("==============================")
    print("GENERATED SQL")
    print("==============================")


    print(sql)


    # --------------------------------------------------------
    # Execute SQL
    # --------------------------------------------------------

    print("\nExecuting SQL...\n")


    try:

        columns, results = execute_sql(sql)


        print("==============================")
        print("DATABASE RESULT")
        print("==============================")


        print(columns)
        print(results)


        # ----------------------------------------------------
        # Generate final answer
        # ----------------------------------------------------

        print("\nGenerating final answer...\n")


        answer = generate_answer(
            question,
            columns,
            results
        )


        print("==============================")
        print("FINAL ANSWER")
        print("==============================")


        print(answer)


    except Exception as error:

        print()
        print("==============================")
        print("SQL ERROR")
        print("==============================")


        print(error)


# ============================================================
# 7. START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()