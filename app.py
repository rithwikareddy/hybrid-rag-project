import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path

from src.hybrid import (
    route_question,
    retrieve_documents,
    generate_sql,
    execute_sql,
    format_sql_result,
    generate_final_answer,
    validate_sql
)


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="HybridIQ",
    page_icon="✦",
    layout="wide"
)


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATABASE_PATH = PROJECT_ROOT / "data" / "warehouse.db"


# ============================================================
# SESSION STATE
# ============================================================

if "question_box" not in st.session_state:
    st.session_state.question_box = ""

if "selected_product" not in st.session_state:
    st.session_state.selected_product = None

if "recent_questions" not in st.session_state:
    st.session_state.recent_questions = []


# ============================================================
# GET DASHBOARD STATISTICS
# ============================================================

def get_dashboard_stats():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        "SELECT SUM(revenue) FROM sales"
    )

    total_revenue = cursor.fetchone()[0] or 0

    cursor.execute(
        "SELECT COUNT(*) FROM sales"
    )

    total_records = cursor.fetchone()[0] or 0

    cursor.execute(
        "SELECT SUM(quantity) FROM sales"
    )

    total_quantity = cursor.fetchone()[0] or 0

    cursor.execute(
        "SELECT COUNT(DISTINCT product) FROM sales"
    )

    total_products = cursor.fetchone()[0] or 0

    connection.close()

    return (
        total_revenue,
        total_records,
        total_quantity,
        total_products
    )


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* ============================================================
   HIDE STREAMLIT DEFAULT UI
   ============================================================ */

header[data-testid="stHeader"] {
    display: none;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

[data-testid="stDecoration"] {
    display: none;
}


/* ============================================================
   MAIN APP
   ============================================================ */

.stApp {

    background:
        radial-gradient(
            circle at 8% 5%,
            rgba(100, 80, 255, 0.18),
            transparent 28%
        ),

        radial-gradient(
            circle at 92% 10%,
            rgba(20, 200, 180, 0.12),
            transparent 25%
        ),

        #080a12;

    color: white;
}


.block-container {

    max-width: 1150px;

    padding-top: 25px;

    padding-bottom: 60px;
}


/* ============================================================
   HEADER
   ============================================================ */

.brand {

    display: flex;

    align-items: center;

    margin-bottom: 25px;
}

.brand-icon {

    width: 45px;

    height: 45px;

    border-radius: 14px;

    background:
        linear-gradient(
            135deg,
            #765cff,
            #20d4c2
        );

    display: flex;

    align-items: center;

    justify-content: center;

    font-size: 23px;

    font-weight: bold;

    box-shadow:
        0 8px 30px
        rgba(100,80,255,0.35);
}

.brand-name {

    margin-left: 12px;

    font-size: 22px;

    font-weight: 800;
}

.brand-status {

    margin-left: auto;

    color: #707991;

    font-size: 11px;

    letter-spacing: 1px;
}


/* ============================================================
   HERO
   ============================================================ */

.hero {

    text-align: center;

    padding-top: 25px;

    padding-bottom: 25px;
}

.badge {

    display: inline-block;

    padding: 8px 16px;

    border-radius: 30px;

    border:
        1px solid
        rgba(130,110,255,0.35);

    background:
        rgba(120,90,255,0.10);

    color: #a99dff;

    font-size: 11px;

    font-weight: 700;

    letter-spacing: 1px;

    margin-bottom: 18px;
}

.hero-title {

    font-size: 50px;

    font-weight: 850;

    letter-spacing: -2px;

    line-height: 1.05;

    color: white;
}

.hero-subtitle {

    margin-top: 15px;

    color: #8992aa;

    font-size: 16px;

    line-height: 1.6;
}


/* ============================================================
   DASHBOARD
   ============================================================ */

.dashboard-title {

    margin-top: 15px;

    margin-bottom: 12px;

    color: #737c95;

    font-size: 11px;

    font-weight: 700;

    letter-spacing: 1.5px;
}


.metric-card {

    background:
        linear-gradient(
            145deg,
            #121522,
            #0d1019
        );

    border:
        1px solid #252a3d;

    border-radius: 17px;

    padding: 20px;

    min-height: 110px;

    box-shadow:
        0 10px 35px
        rgba(0,0,0,0.15);
}

.metric-label {

    color: #6f7891;

    font-size: 10px;

    font-weight: 700;

    letter-spacing: 1.2px;
}

.metric-value {

    color: #f2f4ff;

    font-size: 24px;

    font-weight: 800;

    margin-top: 10px;
}

.metric-description {

    color: #555e76;

    font-size: 10px;

    margin-top: 5px;
}


/* ============================================================
   QUICK QUESTIONS
   ============================================================ */

.quick-title {

    margin-top: 30px;

    margin-bottom: 12px;

    color: #737c95;

    font-size: 11px;

    font-weight: 700;

    letter-spacing: 1.5px;
}


/* ============================================================
   BUTTONS
   ============================================================ */

div[data-testid="stButton"] button {

    background: #10131e;

    border:
        1px solid #272c40;

    color: #aeb6ca;

    border-radius: 14px;

    min-height: 48px;

    font-size: 12px;

    font-weight: 600;

    transition: all 0.2s ease;
}

div[data-testid="stButton"] button:hover {

    border-color: #7561ff;

    color: white;

    background: #15182a;

    transform: translateY(-2px);
}


/* ============================================================
   QUESTION
   ============================================================ */

.question-label {

    color: #737c95;

    font-size: 11px;

    font-weight: 700;

    letter-spacing: 1.5px;

    margin-top: 25px;

    margin-bottom: 8px;
}

div[data-testid="stTextInput"] input {

    background: #10131e;

    color: white;

    border:
        1px solid #292e42;

    border-radius: 15px;

    padding: 17px;

    font-size: 15px;
}

div[data-testid="stTextInput"] input:focus {

    border-color: #7863ff;

    box-shadow:
        0 0 0 1px #7863ff;
}


/* ============================================================
   ANSWER
   ============================================================ */

.answer-box {

    margin-top: 35px;

    padding: 28px;

    border-radius: 20px;

    background:
        linear-gradient(
            145deg,
            #151927,
            #0f121d
        );

    border:
        1px solid #292e42;

    box-shadow:
        0 20px 60px
        rgba(0,0,0,0.25);
}

.answer-label {

    color: #8e98b2;

    font-size: 11px;

    font-weight: 750;

    letter-spacing: 1.5px;

    margin-bottom: 15px;
}

.answer-content {

    color: #f4f6ff;

    font-size: 18px;

    line-height: 1.7;
}


/* ============================================================
   CHART
   ============================================================ */

.chart-box {

    margin-top: 25px;

    padding: 25px;

    border-radius: 20px;

    background: #10131e;

    border:
        1px solid #292e42;
}

.chart-title {

    color: #8e98b2;

    font-size: 11px;

    font-weight: 750;

    letter-spacing: 1.5px;

    margin-bottom: 15px;
}


/* ============================================================
   SYSTEM STATUS
   ============================================================ */

.status-title {

    margin-top: 35px;

    margin-bottom: 12px;

    color: #737c95;

    font-size: 11px;

    font-weight: 700;

    letter-spacing: 1.5px;
}

.status-card {

    background: #10131e;

    border:
        1px solid #272c40;

    border-radius: 16px;

    padding: 20px;
}

.status-name {

    color: #707a94;

    font-size: 10px;

    font-weight: 700;

    letter-spacing: 1.2px;
}

.status-value {

    margin-top: 9px;

    color: #e8ebf5;

    font-size: 14px;

    font-weight: 650;
}

.dot {

    display: inline-block;

    width: 7px;

    height: 7px;

    background: #38d7a2;

    border-radius: 50%;

    margin-right: 7px;

    box-shadow:
        0 0 9px
        rgba(56,215,162,0.7);
}


/* ============================================================
   EXPANDER
   ============================================================ */

div[data-testid="stExpander"] {

    margin-top: 20px;

    background: #0e111b;

    border:
        1px solid #252a3d;

    border-radius: 14px;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {

    text-align: center;

    color: #4f576d;

    font-size: 11px;

    margin-top: 50px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="brand">'
    '<div class="brand-icon">✦</div>'
    '<div class="brand-name">HybridIQ</div>'
    '<div class="brand-status">'
    '● LOCAL AI • ONLINE'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="hero">'
    '<div class="badge">'
    '✦ HYBRID DATA INTELLIGENCE'
    '</div>'
    '<div class="hero-title">'
    'Ask your data anything.'
    '</div>'
    '<div class="hero-subtitle">'
    'One intelligent assistant for your data warehouse '
    'and company knowledge.'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD STATISTICS
# ============================================================

st.markdown(
    '<div class="dashboard-title">'
    'LIVE DATA OVERVIEW'
    '</div>',
    unsafe_allow_html=True
)


total_revenue, total_records, total_quantity, total_products = (
    get_dashboard_stats()
)


d1, d2, d3, d4 = st.columns(4)


with d1:

    st.markdown(
        '<div class="metric-card">'
        '<div class="metric-label">'
        'TOTAL REVENUE'
        '</div>'
        '<div class="metric-value">'
        f'₹{total_revenue:,.0f}'
        '</div>'
        '<div class="metric-description">'
        'Across all sales'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


with d2:

    st.markdown(
        '<div class="metric-card">'
        '<div class="metric-label">'
        'SALES RECORDS'
        '</div>'
        '<div class="metric-value">'
        f'{total_records}'
        '</div>'
        '<div class="metric-description">'
        'Transactions in warehouse'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


with d3:

    st.markdown(
        '<div class="metric-card">'
        '<div class="metric-label">'
        'UNITS SOLD'
        '</div>'
        '<div class="metric-value">'
        f'{total_quantity:,}'
        '</div>'
        '<div class="metric-description">'
        'Total quantity'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


with d4:

    st.markdown(
        '<div class="metric-card">'
        '<div class="metric-label">'
        'PRODUCTS'
        '</div>'
        '<div class="metric-value">'
        f'{total_products}'
        '</div>'
        '<div class="metric-description">'
        'Unique products'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# QUICK QUESTIONS
# ============================================================

st.markdown(
    '<div class="quick-title">'
    'TRY A QUICK QUESTION'
    '</div>',
    unsafe_allow_html=True
)


q1, q2, q3 = st.columns(3)


with q1:

    if st.button(
        "💰  Total sales in Hyderabad",
        use_container_width=True
    ):

        st.session_state.question_box = (
            "What were the total sales in Hyderabad?"
        )

        st.rerun()


with q2:

    if st.button(
        "📄  What is the return policy?",
        use_container_width=True
    ):

        st.session_state.question_box = (
            "What is the return policy?"
        )

        st.rerun()


with q3:

    if st.button(
        "📊  Sales by product",
        use_container_width=True
    ):

        st.session_state.question_box = (
            "What are the sales by product?"
        )

        st.rerun()


# ============================================================
# AVAILABLE PRODUCTS
# ============================================================

st.markdown(
    '<div class="quick-title">'
    'AVAILABLE PRODUCTS'
    '</div>',
    unsafe_allow_html=True
)

p1, p2, p3, p4 = st.columns(4)

products = [
    (p1, "📱  Smartphones", "Smartphones"),
    (p2, "💻  Laptops", "Laptops"),
    (p3, "📲  Tablets", "Tablets"),
    (p4, "🎧  Headphones", "Headphones"),
]

for column, label, product_name in products:
    with column:
        if st.button(label, use_container_width=True, key=f"product_{product_name}"):
            st.session_state.selected_product = product_name
            st.rerun()


if st.session_state.selected_product:

    selected = st.session_state.selected_product

    st.markdown(
        '<div class="quick-title">'
        'SUGGESTED QUESTIONS'
        '</div>',
        unsafe_allow_html=True
    )

    suggestion_map = {
        "Smartphones": [
            "How many smartphones were sold?",
            "What were the total smartphone sales?",
            "Which region sold the most smartphones?"
        ],
        "Laptops": [
            "How many laptops were sold?",
            "What were the total laptop sales?",
            "Which region sold the most laptops?"
        ],
        "Tablets": [
            "How many tablets were sold?",
            "What were the total tablet sales?",
            "Which region sold the most tablets?"
        ],
        "Headphones": [
            "How many headphones were sold?",
            "What were the total headphone sales?",
            "Which region sold the most headphones?"
        ]
    }

    s1, s2, s3 = st.columns(3)

    for column, suggestion in zip(
        [s1, s2, s3],
        suggestion_map[selected]
    ):
        with column:
            if st.button(
                suggestion,
                use_container_width=True,
                key=f"suggestion_{selected}_{suggestion}"
            ):
                st.session_state.question_box = suggestion
                st.rerun()


# ============================================================
# RECENTLY ASKED QUESTIONS
# ============================================================

if st.session_state.recent_questions:

    st.markdown(
        '<div class="quick-title">'
        'RECENTLY ASKED'
        '</div>',
        unsafe_allow_html=True
    )

    recent_columns = st.columns(3)

    for column, recent_question in zip(
        recent_columns,
        st.session_state.recent_questions[:3]
    ):
        with column:
            if st.button(
                f"💬  {recent_question}",
                use_container_width=True,
                key=f"recent_{hash(recent_question)}"
            ):
                st.session_state.question_box = recent_question
                st.rerun()


# ============================================================
# QUESTION INPUT
# ============================================================

st.markdown(
    '<div class="question-label">'
    'YOUR QUESTION'
    '</div>',
    unsafe_allow_html=True
)


question = st.text_input(
    "Question",
    key="question_box",
    placeholder=(
        "Ask about sales, company information, policies..."
    ),
    label_visibility="collapsed"
)


# ============================================================
# ASK BUTTON
# ============================================================

ask = st.button(
    "✦  Ask HybridIQ",
    type="primary",
    use_container_width=True
)


if ask:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        # ====================================================
        # SAVE TO RECENT QUESTIONS
        # ====================================================

        clean_question = question.strip()

        # Keep only unique questions, with the newest first.
        st.session_state.recent_questions = [
            q for q in st.session_state.recent_questions
            if q.lower() != clean_question.lower()
        ]

        st.session_state.recent_questions.insert(
            0,
            clean_question
        )

        st.session_state.recent_questions = (
            st.session_state.recent_questions[:3]
        )

        # ====================================================
        # ROUTER
        # ====================================================

        with st.spinner(
            "Understanding your question..."
        ):

            route = route_question(
                question
            )


        sql_result = None
        rag_context = None

        chart_columns = None
        chart_results = None


        # ====================================================
        # SQL PATH
        # ====================================================

        if route in ["SQL", "BOTH"]:

            with st.spinner(
                "Analyzing the data warehouse..."
            ):

                try:

                    sql = generate_sql(
                        question
                    )

                    sql = validate_sql(sql)

                    database_result = execute_sql(
                        sql
                    )

                    chart_columns = database_result["columns"]
                    chart_results = database_result["rows"]

                    sql_result = format_sql_result(
                        database_result
                    )

                except Exception as error:

                    sql_result = (
                        f"Database error: {error}"
                    )


        # ====================================================
        # RAG PATH
        # ====================================================

        if route in ["RAG", "BOTH"]:

            with st.spinner(
                "Searching company knowledge..."
            ):

                rag_context = retrieve_documents(
                    question
                )


        # ====================================================
        # FINAL ANSWER
        # ====================================================

        with st.spinner(
            "Generating answer..."
        ):

            database_result_for_answer = None

            if route in ["SQL", "BOTH"] and "database_result" in locals():
                database_result_for_answer = database_result

            answer = generate_final_answer(
                question=question,
                route=route,
                database_result=database_result_for_answer,
                documents=rag_context
            )


        # ====================================================
        # ROUTE
        # ====================================================

        route_names = {

            "SQL":
                "● DATABASE QUERY",

            "RAG":
                "● DOCUMENT SEARCH",

            "BOTH":
                "● HYBRID QUERY",

            "OUT_OF_SCOPE":
                "● OUT OF SCOPE"
        }


        st.markdown(
            f'<div style="margin-top:35px;'
            f'color:#7f89a3;'
            f'font-size:11px;'
            f'font-weight:700;'
            f'letter-spacing:1.5px;">'
            f'{route_names.get(route, route)}'
            f'</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # ANSWER
        # ====================================================

        safe_answer = (
            answer
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )


        st.markdown(
            '<div class="answer-box">'
            '<div class="answer-label">'
            '✦ ANSWER'
            '</div>'
            '<div class="answer-content">'
            f'{safe_answer}'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # CHART
        # ====================================================

        if chart_columns and chart_results:

            question_lower = question.lower()


            is_product_chart = (

                "by product" in question_lower

                or "product wise" in question_lower

                or "product-wise" in question_lower

                or "each product" in question_lower

                or "per product" in question_lower
            )


            is_region_chart = (

                "by region" in question_lower

                or "region wise" in question_lower

                or "region-wise" in question_lower

                or "each region" in question_lower

                or "per region" in question_lower
            )


            # ------------------------------------------------
            # PRODUCT CHART
            # ------------------------------------------------

            if is_product_chart:

                df = pd.DataFrame(
                    chart_results,
                    columns=chart_columns
                )


                if len(df) > 0:

                    category_column = df.columns[0]

                    value_column = df.columns[-1]


                    df = df[
                        [category_column, value_column]
                    ]


                    df = df.set_index(
                        category_column
                    )


                    st.markdown(
                        '<div class="chart-box">'
                        '<div class="chart-title">'
                        '📊 SALES BY PRODUCT'
                        '</div>',
                        unsafe_allow_html=True
                    )


                    st.bar_chart(
                        df,
                        use_container_width=True
                    )


                    st.markdown(
                        '</div>',
                        unsafe_allow_html=True
                    )


            # ------------------------------------------------
            # REGION CHART
            # ------------------------------------------------

            elif is_region_chart:

                df = pd.DataFrame(
                    chart_results,
                    columns=chart_columns
                )


                if len(df) > 0:

                    category_column = df.columns[0]

                    value_column = df.columns[-1]


                    df = df[
                        [category_column, value_column]
                    ]


                    df = df.set_index(
                        category_column
                    )


                    st.markdown(
                        '<div class="chart-box">'
                        '<div class="chart-title">'
                        '📊 SALES BY REGION'
                        '</div>',
                        unsafe_allow_html=True
                    )


                    st.bar_chart(
                        df,
                        use_container_width=True
                    )


                    st.markdown(
                        '</div>',
                        unsafe_allow_html=True
                    )


        # ====================================================
        # SYSTEM STATUS
        # ====================================================

        st.markdown(
            '<div class="status-title">'
            'SYSTEM STATUS'
            '</div>',
            unsafe_allow_html=True
        )


        c1, c2, c3 = st.columns(3)


        with c1:

            st.markdown(
                '<div class="status-card">'
                '<div class="status-name">'
                'DATA WAREHOUSE'
                '</div>'
                '<div class="status-value">'
                '<span class="dot"></span>'
                'Connected'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )


        with c2:

            st.markdown(
                '<div class="status-card">'
                '<div class="status-name">'
                'KNOWLEDGE BASE'
                '</div>'
                '<div class="status-value">'
                '<span class="dot"></span>'
                'ChromaDB Ready'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )


        with c3:

            st.markdown(
                '<div class="status-card">'
                '<div class="status-name">'
                'AI ENGINE'
                '</div>'
                '<div class="status-value">'
                '<span class="dot"></span>'
                'Qwen2.5:3b'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )


        # ====================================================
        # TECHNICAL DETAILS
        # ====================================================

        with st.expander(
            "View system details"
        ):

            st.write(
                f"**Selected route:** {route}"
            )


            if sql_result:

                st.write(
                    "**Database result:**"
                )

                st.code(
                    sql_result
                )


            if rag_context:

                st.write(
                    "**Retrieved document information:**"
                )

                st.code(
                    rag_context
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'HybridIQ · Hybrid RAG Data Warehouse QA · '
    'Powered by local AI'
    '</div>',
    unsafe_allow_html=True
)