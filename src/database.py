import sqlite3
from pathlib import Path

# Database location
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "warehouse.db"

# Make sure data folder exists
DATA_DIR.mkdir(exist_ok=True)

# Remove the old demo database
if DATABASE_PATH.exists():
    DATABASE_PATH.unlink()
    print("Old demo database removed.")

# Connect to SQLite
connection = sqlite3.connect(DATABASE_PATH)
cursor = connection.cursor()

# Create sales table
cursor.execute("""
CREATE TABLE sales (
    sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sale_date TEXT,
    region TEXT,
    product TEXT,
    quantity INTEGER,
    revenue REAL
)
""")

# Realistic electronics sales data
sales_data = [
    ("2025-01-10", "Hyderabad", "Smartphone", 10, 300000),
    ("2025-01-15", "Mumbai", "Laptop", 5, 300000),
    ("2025-02-02", "Hyderabad", "Laptop", 4, 240000),
    ("2025-02-10", "Bangalore", "Tablet", 8, 200000),
    ("2025-03-05", "Hyderabad", "Headphones", 20, 100000),
    ("2025-03-20", "Mumbai", "Smartphone", 12, 360000),
    ("2025-04-12", "Hyderabad", "Smartphone", 7, 210000),
    ("2025-05-01", "Bangalore", "Laptop", 6, 360000),
    ("2025-06-18", "Hyderabad", "Tablet", 10, 250000),
    ("2025-07-10", "Mumbai", "Headphones", 18, 90000),
    ("2025-08-05", "Bangalore", "Smartphone", 9, 270000),
    ("2025-08-20", "Hyderabad", "Laptop", 3, 180000),
]

# Insert data
cursor.executemany("""
INSERT INTO sales (sale_date, region, product, quantity, revenue)
VALUES (?, ?, ?, ?, ?)
""", sales_data)

# Save changes
connection.commit()

# Show confirmation
cursor.execute("SELECT COUNT(*) FROM sales")
record_count = cursor.fetchone()[0]

cursor.execute("SELECT SUM(revenue) FROM sales")
total_revenue = cursor.fetchone()[0]

cursor.execute("SELECT SUM(quantity) FROM sales")
total_quantity = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(DISTINCT product) FROM sales")
product_count = cursor.fetchone()[0]

connection.close()

print("\n====================================")
print("   DATA WAREHOUSE CREATED")
print("====================================")
print(f"Sales Records : {record_count}")
print(f"Total Revenue : ₹{total_revenue:,.0f}")
print(f"Units Sold    : {total_quantity}")
print(f"Products      : {product_count}")
print("====================================")