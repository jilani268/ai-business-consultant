import base64
from datetime import date
import hashlib
import json
import os
import sqlite3
import tempfile
import numpy as np
from openai import OpenAI
import pandas as pd
import pdfplumber
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

st.set_page_config(page_title="AI Business Consultant", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "app_data.db")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_files")


def get_connection():
  conn = sqlite3.connect(DB_PATH)
  conn.execute("PRAGMA foreign_keys = ON")
  return conn


def init_db():
  conn = get_connection()
  c = conn.cursor()

  c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

  c.execute("""
        CREATE TABLE IF NOT EXISTS business_profile (
            user_id INTEGER PRIMARY KEY,
            company_name TEXT,
            owner_name TEXT,
            industry TEXT,
            business_type TEXT,
            location TEXT,
            employees INTEGER
        )
    """)

  try:
    c.execute("ALTER TABLE business_profile ADD COLUMN last_file_id INTEGER")
  except sqlite3.OperationalError:
    pass

  c.execute("""
        CREATE TABLE IF NOT EXISTS files (
            file_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_path TEXT NOT NULL,
            status TEXT DEFAULT 'processed',
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    """)

  c.execute("""
        CREATE TABLE IF NOT EXISTS financial_records (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            file_id INTEGER,
            month TEXT NOT NULL,
            revenue REAL,
            expenses REAL,
            profit REAL,
            profit_margin REAL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
            FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE SET NULL,
            UNIQUE(user_id, month)
        )
    """)

  conn.commit()
  conn.close()
  os.makedirs(UPLOAD_DIR, exist_ok=True)


def hash_password(password):
  return hashlib.sha256(password.encode()).hexdigest()


def register_user(username, password):
  conn = get_connection()
  c = conn.cursor()
  try:
    with conn:
      c.execute(
          "INSERT INTO users (username, password_hash) VALUES (?, ?)",
          (username, hash_password(password)),
      )
    return True, "Account created successfully."
  except sqlite3.IntegrityError:
    return False, "That username is already taken."
  finally:
    conn.close()


def verify_user(username, password):
  conn = get_connection()
  c = conn.cursor()
  c.execute(
      "SELECT user_id, password_hash FROM users WHERE username = ?", (username,)
  )
  row = c.fetchone()
  conn.close()
  if row is None:
    return None
  user_id, stored_hash = row
  return user_id if stored_hash == hash_password(password) else None


def save_uploaded_file(user_id, uploaded_file):
  user_dir = os.path.join(UPLOAD_DIR, str(user_id))
  os.makedirs(user_dir, exist_ok=True)

  file_path = os.path.join(user_dir, uploaded_file.name)
  with open(file_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

  file_type = uploaded_file.name.split(".")[-1].lower()

  conn = get_connection()
  c = conn.cursor()
  with conn:
    c.execute(
        """
            INSERT INTO files (user_id, filename, file_type, file_path, status)
            VALUES (?, ?, ?, ?, 'processed')
        """,
        (user_id, uploaded_file.name, file_type, file_path),
    )
    file_id = c.lastrowid
  conn.close()
  return file_id


def list_user_files(user_id):
  conn = get_connection()
  df = pd.read_sql_query(
      """
        SELECT file_id, filename AS Filename, file_type AS Type,
               status AS Status, uploaded_at AS "Uploaded At"
        FROM files
        WHERE user_id = ?
        ORDER BY uploaded_at DESC
    """,
      conn,
      params=(user_id,),
  )
  conn.close()
  return df


def load_data_by_file(file_id):
  file_id = int(file_id)
  conn = get_connection()
  df = pd.read_sql_query(
      """
        SELECT month AS Month, revenue AS Revenue, expenses AS Expenses,
            profit AS Profit, profit_margin AS "Profit Margin(%)"
        FROM financial_records
        WHERE file_id = ?
        ORDER BY record_id
    """,
      conn,
      params=(file_id,),
  )
  conn.close()
  return df if not df.empty else None


def save_financial_data(user_id, df, file_id=None):
  conn = get_connection()
  c = conn.cursor()
  try:
    with conn:
      for _, row in df.iterrows():
        c.execute(
            """
                    INSERT INTO financial_records
                        (user_id, file_id, month, revenue, expenses, profit, profit_margin, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id, month)
                    DO UPDATE SET
                        file_id = excluded.file_id,
                        revenue = excluded.revenue,
                        expenses = excluded.expenses,
                        profit = excluded.profit,
                        profit_margin = excluded.profit_margin,
                        updated_at = CURRENT_TIMESTAMP
                """,
            (
                user_id,
                file_id,
                str(row.get("Month", row.get("month", ""))),
                float(row.get("Revenue", row.get("revenue", 0)) or 0),
                float(row.get("Expenses", 0) or 0),
                float(row.get("Profit", row.get("profit", 0)) or 0),
                float(
                    row.get("Profit Margin(%)", row.get("profit_margin", 0))
                    or 0
                ),
            ),
        )
    return True, "Data saved successfully."
  except Exception as e:
    return False, f"Save failed, no changes were made: {e}"
  finally:
    conn.close()


def load_financial_data(user_id):
  conn = get_connection()
  df = pd.read_sql_query(
      """
        SELECT month AS Month, revenue AS Revenue, expenses AS "Expenses",
               profit AS Profit, profit_margin AS "Profit Margin(%)"
        FROM financial_records
        WHERE user_id = ?
        ORDER BY record_id
    """,
      conn,
      params=(user_id,),
  )
  conn.close()
  return df if not df.empty else None


def load_business_profile(user_id):
  conn = get_connection()
  c = conn.cursor()
  c.execute("SELECT * FROM business_profile WHERE user_id=?", (user_id,))
  data = c.fetchone()
  conn.close()
  return data


def set_last_file(user_id, file_id):
  conn = get_connection()
  c = conn.cursor()
  with conn:
    c.execute(
        "UPDATE business_profile SET last_file_id=? WHERE user_id=?",
        (file_id, user_id),
    )
  conn.close()


def get_last_file(user_id):
  conn = get_connection()
  c = conn.cursor()
  c.execute(
      "SELECT last_file_id FROM business_profile WHERE user_id=?", (user_id,)
  )
  row = c.fetchone()
  conn.close()
  return row[0] if row and row[0] is not None else None


def extract_text_from_pdf(uploaded_file):
  with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(uploaded_file.getbuffer())
    tmp_path = tmp.name

  text = ""
  try:
    with pdfplumber.open(tmp_path) as pdf:
      for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
          text += page_text + "\n"
  finally:
    os.unlink(tmp_path)

  return text.strip()


def ask_ai_to_extract_financials(raw_text):
  prompt = f"""
You are a financial data extraction assistant. Below is text extracted from an income statement.

Extract every month of financial figures you can find, and return ONLY a JSON array. 
Each item must look like:
[
  {{"month": "January 2026", "revenue": 150000, "expenses": 110000}}
]

Statement text:
\"\"\"
{raw_text}
\"\"\"
"""
  try:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You extract structured financial data and respond only with valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=800,
    )
    raw_reply = response.choices[0].message.content.strip()

    # --- EASY 2-LINE FIX FOR YOUR PRESENTATION ---
    # Strip away backticks and 'json' tag if Llama adds them
    raw_reply = raw_reply.replace("```json", "").replace("```", "").strip()

    return json.loads(raw_reply)
  except Exception as e:
    return None

def save_daily_business(user_id, entry_date, sales, cogs, other_expenses):
  gross_profit = sales - cogs
  net_profit = gross_profit - other_expenses
  profit_margin = (net_profit / sales) * 100 if sales > 0 else 0

  conn = get_connection()
  c = conn.cursor()
  c.execute("""
        CREATE TABLE IF NOT EXISTS daily_business (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            entry_date TEXT,
            sales REAL,
            cogs REAL,
            other_expenses REAL,
            gross_profit REAL,
            net_profit REAL,
            profit_margin REAL
        )
    """)
  c.execute(
      """INSERT INTO daily_business (user_id, entry_date, sales, cogs, other_expenses, gross_profit, net_profit, profit_margin)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
      (
          user_id,
          str(entry_date),
          sales,
          cogs,
          other_expenses,
          gross_profit,
          net_profit,
          profit_margin,
      ),
  )
  conn.commit()
  conn.close()
  return True, "Saved."


def load_daily_business(user_id):
  conn = get_connection()
  c = conn.cursor()
  c.execute("""
    CREATE TABLE IF NOT EXISTS daily_business (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        entry_date TEXT,
        sales REAL,
        cogs REAL,
        other_expenses REAL,
        gross_profit REAL,
        net_profit REAL,
        profit_margin REAL
    )
    """)
  conn.commit()

  df = pd.read_sql_query(
      "SELECT * FROM daily_business WHERE user_id=? ORDER BY entry_date",
      conn,
      params=(user_id,),
  )
  conn.close()
  return df


def generate_monthly_summary(df):
  if df.empty:
    return None

  summary = pd.DataFrame({
      "Month": [
          pd.to_datetime(df["entry_date"]).dt.strftime("%B %Y").iloc[-1]
      ],
      "Revenue": [df["sales"].sum()],
      "COGS": [df["cogs"].sum()],
      "Other Expenses": [df["other_expenses"].sum()],
      "Gross Profit": [df["gross_profit"].sum()],
      "Profit": [df["net_profit"].sum()],
  })

  summary["Expenses"] = summary["COGS"] + summary["Other Expenses"]
  summary["Profit Margin(%)"] = (summary["Profit"] / summary["Revenue"]) * 100
  return summary


# --- NEW GST & DOCUMENT VAULT FUNCTION ---
def render_document_vault():
  st.title("📂 GST & Document Vault")
  st.caption(
      "Store and view all your business bills and receipts in one place."
  )

  if "vault_bills" not in st.session_state:
    st.session_state.vault_bills = []

  # 1. Upload Form
  with st.form("add_bill_form", clear_on_submit=True):
    st.subheader("➕ Upload New Document")

    col1, col2 = st.columns(2)
    title = col1.text_input("Bill Title / Vendor Name")
    amount = col2.number_input("Amount (₹)", min_value=0.0)

    uploaded_pdf = st.file_uploader("Upload Invoice (PDF)", type=["pdf"])
    save_btn = st.form_submit_button("Save Document")

    if save_btn:
      if title and uploaded_pdf:
        st.session_state.vault_bills.append({
            "title": title,
            "amount": f"₹{amount:,.2f}",
            "file_name": uploaded_pdf.name,
            "file_data": uploaded_pdf.read(),
        })
        st.success(f"Saved '{title}' successfully!")
        st.rerun()
      else:
        st.warning("Please provide a title and select a PDF file.")

  st.divider()

  # 2. View Saved Bills
  st.subheader("📑 Saved Documents")

  if not st.session_state.vault_bills:
    st.info("No documents uploaded yet.")
  else:
    bill_options = [
        f"{item['title']} - {item['amount']} ({item['file_name']})"
        for item in st.session_state.vault_bills
    ]
    selected_index = st.selectbox(
        "Select a document to open:",
        range(len(bill_options)),
        format_func=lambda i: bill_options[i],
    )

    selected_doc = st.session_state.vault_bills[selected_index]

    st.markdown(f"### 📄 Opening: {selected_doc['title']}")

    st.download_button(
        label="⬇️ Download PDF",
        data=selected_doc["file_data"],
        file_name=selected_doc["file_name"],
        mime="application/pdf",
    )

    # NOTE: st.pdf() is buggy in some Streamlit builds (throws
    # "must be ... asset_dir to use file-backed css"). Render the PDF
    # manually via a base64 iframe instead to avoid that component.
    base64_pdf = base64.b64encode(selected_doc["file_data"]).decode("utf-8")
    pdf_display = f"""
        <iframe
            src="data:application/pdf;base64,{base64_pdf}"
            width="100%"
            height="800"
            type="application/pdf">
        </iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)


GROQ_KEY = ("GROQ_API_KEY")
api_key = os.environ.get("GROQ_API_KEY") or GROQ_KEY

client = OpenAI(base_url="https://api.groq.com/openai/v1",api_key=api_key)

training_sentences = [
    "highest sales",
    "maximum sales",
    "top sales",
    "lowest sales",
    "minimum sales",
    "highest profit",
    "lowest profit",
    "total sales",
    "total expenses",
    "total profit",
    "average sales",
    "average profit",
]

training_labels = [
    "highest_sales",
    "highest_sales",
    "highest_sales",
    "lowest_sales",
    "lowest_sales",
    "highest_profit",
    "lowest_profit",
    "total_sales",
    "total_expenses",
    "total_profit",
    "average_sales",
    "average_profit",
    "compare_sales",
]

training_sentences.extend([
    "january sales",
    "february sales",
    "march sales",
    "april sales",
    "may sales",
    "june sales",
    "july sales",
    "august sales",
    "september sales",
    "october sales",
    "november sales",
    "december sales",
    "sales in june",
    "profit in june",
    "expenses in june",
    "revenue of june",
])

training_labels.extend([
    "month_query",
    "month_query",
    "month_query",
    "month_query",
    "month_query",
    "month_query",
    "month_query",
    "month_query",
    "month_query",
    "month_query",
    "month_query",
    "month_query",
    "month_query",
    "month_query",
    "month_query",
    "month_query",
])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(training_sentences)


def detect_intent(question):
  question_vector = vectorizer.transform([question.lower()])
  similarity = cosine_similarity(question_vector, X)
  best = similarity.argmax()
  confidence = similarity[0][best]

  if confidence < 0.30:
    return "unknown"

  return training_labels[best]


industry_thresholds = {
    "Retail": {"excellent": 15, "good": 8, "average": 3},
    "Wholesale": {"excellent": 12, "good": 6, "average": 2},
    "Manufacturing": {"excellent": 18, "good": 10, "average": 4},
    "Services": {"excellent": 30, "good": 18, "average": 8},
    "Food": {"excellent": 15, "good": 8, "average": 3},
    "Technology": {"excellent": 35, "good": 20, "average": 10},
    "Other": {"excellent": 25, "good": 15, "average": 5},
}


def generate_industry_advice(industry, margin, revenue, expenses):
  if not industry:
    return ["Please enter your business profile first."]

  tips = []
  if industry == "Retail":
    tips = [
        "Track inventory turnover — slow-moving stock ties up cash.",
        (
            "Watch seasonal demand and plan discounts before stock goes"
            " stale."
        ),
        (
            "Negotiate bulk purchase discounts with suppliers to improve"
            " margin."
        ),
        "Consider loyalty programs to increase repeat customer visits.",
    ]
  elif industry == "Wholesale":
    tips = [
        "Focus on order volume and bulk pricing — margins are thin per unit.",
        "Strengthen supplier relationships to secure better credit terms.",
        "Reduce holding costs by improving inventory forecasting.",
        "Diversify your buyer base so no single client dominates revenue.",
    ]
  elif industry == "Manufacturing":
    tips = [
        "Monitor raw material costs closely — they directly affect margin.",
        "Invest in process efficiency to reduce production waste.",
        "Track machine downtime — it has a direct cost impact.",
        "Consider bulk raw material contracts to stabilize costs.",
    ]
  elif industry == "Services":
    tips = [
        (
            "Your main cost is usually labor — track utilization rates per"
            " employee."
        ),
        (
            "Focus on client retention; repeat clients cost less than new"
            " acquisition."
        ),
        "Consider tiered service pricing to capture more value.",
        "Track billable vs non-billable hours to spot inefficiencies.",
    ]
  elif industry == "Food":
    tips = [
        (
            "Track food cost percentage — aim to keep it well below revenue"
            " share."
        ),
        "Reduce spoilage/waste through better inventory rotation.",
        "Monitor peak hours and staff accordingly to control labor cost.",
        "Menu engineering: promote high-margin items more visibly.",
    ]
  elif industry == "Technology":
    tips = [
        "Track customer acquisition cost (CAC) vs lifetime value (LTV).",
        "Watch recurring revenue trends if subscription-based.",
        "Keep infrastructure/hosting costs proportional to active users.",
        "Reinvest profit into product development for long-term growth.",
    ]
  else:
    tips = [
        "Track your biggest cost driver and look for ways to reduce it.",
        "Compare month-over-month revenue trends to spot patterns.",
        "Set a target profit margin and review it monthly.",
    ]

  return tips


init_db()

if "user_id" not in st.session_state:
  st.session_state.user_id = None
  st.session_state.username = None

if "view" not in st.session_state:
  st.session_state.view = "home"


def go_to(view_name):
  st.session_state.view = view_name


if st.session_state.user_id is None:
  if st.session_state.view == "home":
    st.markdown(
        """
        <style>
        /* 1. Global Font Family Import (Optional: Google Font) */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

        /* Apply global font to the app */
        html, body, [class*="css"] {
            font-family: 'Poppins', 'Segoe UI', sans-serif !important;
        }

        .block-container {
            padding-top: 3.5rem !important;
            padding-bottom: 1rem !important;
            max-width: 95% !important;
        }

        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #0d0d0d !important;
        }

        .navbar {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            background-color: #1a1a1a;
            padding: 14px 24px;
            border-radius: 10px;
            border: 1px solid #282828;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
            margin-bottom: 20px;
        }

        /* Navbar Title Font */
        .navbar-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 36px !important; /* CHANGE NAVBAR FONT SIZE HERE */
            font-family: 'Poppins', sans-serif !important; /* CHANGE FONT TYPE */
            font-weight: 700;
            color: #ffffff !important;
        }
        .navbar-logo {
            font-size: 26px !important;
            line-height: 1;
        }
        .navbar-title span {
            color: #00ff88;
        }

        .hero {
            padding: 10px 0px 20px 0px;
        }

        /* Hero Tagline Font */
        .hero .tag {
            color: #00ff88;
            font-weight: 600;
            font-size: 18px !important; /* CHANGE TAG SIZE */
            font-family: 'Poppins', sans-serif !important;
        }

        /* Hero Main Heading Font */
        .hero h1 {
            font-size: 48px !important; /* CHANGE MAIN HEADING SIZE */
            font-family: 'Poppins', sans-serif !important;
            font-weight: 800;
            color: #f5f5f5;
            line-height: 1.15;
            margin: 8px 0;
        }
        .hero h1 span {
            color: #00ff88;
        }

        /* Hero Paragraph Font */
        .hero p {
            font-size: 18px !important; /* CHANGE PARAGRAPH SIZE */
            font-family: 'Poppins', sans-serif !important;
            color: #bbb;
            max-width: 480px;
        }

        /* Streamlit Button Font */
        div[data-testid="stButton"] button {
            background-color: #00ff88;
            color: #0d0d0d;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px !important; /* CHANGE BUTTON TEXT SIZE */
            font-family: 'Poppins', sans-serif !important;
            font-weight: 700;
            border: none;
        }
        div[data-testid="stButton"] button:hover {
            background-color: #00cc6a;
            color: #0d0d0d;
        }

        /* Feature Cards Font */
        .feature-card {
            background: #1a1a1a;
            border: 1px solid #222;
            border-radius: 12px;
            padding: 20px 12px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.4);
            height: 100%;
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        }
        .feature-card:hover {
            transform: translateY(-6px);
            box-shadow: 0 8px 25px rgba(0,255,140,0.25);
            border-color: #00ff88;
        }
        .feature-card .icon {
            font-size: 26px;
            background: #0d0d0d;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 10px auto;
            border: 1px solid #00ff88;
        }
        .feature-card h4 {
            color: #00ff88;
            margin-bottom: 6px;
            font-size: 18px !important; /* CARD HEADING SIZE */
            font-family: 'Poppins', sans-serif !important;
            font-weight: 600;
        }
        .feature-card p {
            color: #ccc;
            font-size: 14px !important; /* CARD TEXT SIZE */
            font-family: 'Poppins', sans-serif !important;
            line-height: 1.3;
        }

        /* Description Card Font */
        .description-card {
            background: #1a1a1a;
            border: 1px solid #222;
            border-radius: 12px;
            padding: 25px 30px;
            margin-top: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.4);
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        }
        .description-card:hover {
            transform: translateY(-6px);
            box-shadow: 0 8px 25px rgba(0,255,140,0.25);
            border-color: #00ff88;
        }
        .description-card h2 {
            color: #00ff88;
            font-size: 26px !important; /* DESCRIPTION HEADING SIZE */
            font-family: 'Poppins', sans-serif !important;
            margin-bottom: 12px;
            font-weight: 700;
        }
        .description-card p {
            color: #ccc;
            font-size: 16px !important; /* DESCRIPTION TEXT SIZE */
            font-family: 'Poppins', sans-serif !important;
            line-height: 1.6;
            margin: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        header[data-testid="stHeader"] {
            visibility: hidden !important;
            height: 0px !important;
        }
        footer {
            visibility: hidden !important;
        }
        #MainMenu {
            visibility: hidden !important;
        }
        .block-container {
            padding-top: 1rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_logo, col_title = st.columns([0.08, 0.92], vertical_alignment="center")

    with col_logo:
  # Load your logo file if it exists, otherwise show default emoji
        if os.path.exists("hero_logo.png"):
            st.image("hero_logo.png", width=70)
        else:
            st.write("📈")

    with col_title:
  # Title with highlighted green text
        st.markdown("## AI Business :green[Consultant]")


    col_text, col_image = st.columns([1.2, 1])

    with col_text:
      st.markdown(
          """
                <div class="hero">
                    <div class="tag">Welcome to</div>
                    <h1>AI Business<br><span>Consultant</span></h1>
                    <p>Your smart partner for data-driven decisions, financial insights, and business growth.</p>
                </div>
                """,
          unsafe_allow_html=True,
      )

      b1, b2 = st.columns(2)
      with b1:
        if st.button("Login to Your Account", use_container_width=True):
          go_to("login")
          st.rerun()
      with b2:
        if st.button("Create New Account", use_container_width=True):
          go_to("signup")
          st.rerun()

    with col_image:
      st.image("hero_image.png", use_container_width=True)

    st.write("")

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    features = [
        (
            "📊",
            "Dashboard Overview",
            (
                "Get a complete overview of your business performance in one"
                " glance."
            ),
        ),
        (
            "☁️",
            "Upload Anything",
            (
                "Upload CSV, Excel, or PDF income statements and let the app"
                " organize it for you."
            ),
        ),
        (
             "📂",
             "GST & Vault",
             (
                 "Safely store and instantly view all your GST bills and PDF receipts in one place."
             ),
        ),
        (
            "🤖",
            "AI Consultant",
            (
                "Ask business questions and get AI-powered insights based on"
                " your real data."
            ),
        ),
        (
            "💼",
            "Industry-Aware",
            (
                "Personalized insights based on your industry, market trends,"
                " and business type."
            ),
        ),
        (
            "🔒",
            "Secure & Private",
            (
                "Your data is 100% secure and private. We don't share your"
                " information."
            ),
        ),
    ]

    for col, (icon, title, desc) in zip([c1, c2, c3, c4, c5, c6], features):
      with col:
        st.markdown(
            f"""
                    <div class="feature-card">
                        <div class="icon">{icon}</div>
                        <h4>{title}</h4>
                        <p>{desc}</p>
                    </div>
                    """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
            <div class="description-card">
                <h2>Smarter business decisions, powered by AI</h2>
                <p>AI business Consultancy helps small and growing businesses understand their numbers without needing an accountant. Upload sales data, income statements, or even daily turnover and get instant dashboards, profit analysis, and AI-powered answers to your business questions.</p>
            </div>
            """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.stop()

  elif st.session_state.view == "login":
    st.title("Login")

    if st.button("← Back to Home"):
      go_to("home")
      st.rerun()

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
      if username.strip() == "" or password.strip() == "":
        st.warning("Please enter both a username and password.")
      else:
        user_id = verify_user(username.strip(), password)
        if user_id is not None:
          st.session_state.user_id = user_id
          st.session_state.username = username.strip()

          profile = load_business_profile(user_id)

          if profile is None:
            st.session_state.profile_completed = False
            st.session_state.default_page = "Business Profile"
          else:
            st.session_state.profile_completed = True
            st.session_state.default_page = "Dashboard"

            st.session_state["company_name"] = profile[1]
            st.session_state["owner_name"] = profile[2]
            st.session_state["industry"] = profile[3]
            st.session_state["location"] = profile[5]
            st.session_state["employees"] = profile[6]

          last_file_id = get_last_file(user_id)
          if last_file_id is not None:
            saved_df = load_data_by_file(last_file_id)
          else:
            saved_df = load_financial_data(user_id)

          if saved_df is not None:
            st.session_state["df"] = saved_df
            st.session_state["business_data"] = saved_df

            if "Revenue" in saved_df.columns:
              st.session_state["revenue"] = float(saved_df["Revenue"].sum())

            if "Expenses" in saved_df.columns:
              st.session_state["expenses"] = float(saved_df["Expenses"].sum())

            st.success(
                f"Welcome back, {username}! Your saved data has been loaded."
            )
          else:
            st.success(f"Welcome, {username}!")
          st.rerun()
        else:
          st.error("Invalid username or password.")
    st.stop()

  elif st.session_state.view == "signup":
    st.title("Sign Up")

    if st.button("← Back to Home"):
      go_to("home")
      st.rerun()

    username = st.text_input("Choose a username")
    password = st.text_input("Choose a password", type="password")

    if st.button("Create Account"):
      if username.strip() == "" or password.strip() == "":
        st.warning("Please enter both a username and password.")
      else:
        success, msg = register_user(username.strip(), password)
        if success:
          st.success(msg + " You can log in now.")
        else:
          st.error(msg)

    st.write("Already have an account?")
    if st.button("Go to Login"):
      go_to("login")
      st.rerun()

    st.stop()

else:
  if os.path.exists("hero_logo.png"):
    col1, col2, col3 = st.sidebar.columns([1, 2, 1])
    with col2:
        st.image("hero_logo.png", width=120)
  st.sidebar.write(f"Logged in as: *{st.session_state.username}*")
  if st.sidebar.button("Logout"):
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.pop("df", None)
    st.rerun()

if not st.session_state.get("profile_completed", False):
  st.sidebar.warning("Please complete your business profile")
  page = "Business Profile"
else:
  # --- UPDATED SIDEBAR RADIO MENU INCLUDING VAULT ---
  page = st.sidebar.radio(
      "Navigation",
      [
          "Dashboard",
          "Business Profile",
          "Financial Overview",
          "GST & Document Vault",  # Added feature choice here
          "AI Marketing & Promotion",
          "AI Analysis",
          "Business Data Manager",
          "Database",
          "AI Financial Advisor Chatbot",
      ],
  )

df = st.session_state.get("business_data")

# ---------------------------------------------------------
# GLOBAL STYLES
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        color: #ffffff;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #00e5ff !important;
    }

    html body div[data-testid="stMetricValue"],
    html body div[data-testid="stMetricValue"] * {
        color: #00e5ff !important;
    }

    html body section[data-testid="stSidebar"] label,
    html body section[data-testid="stSidebar"] label *,
    html body section[data-testid="stSidebar"] p,
    html body section[data-testid="stSidebar"] p * {
        color: #00e5ff !important;
    }

    div[data-testid="stMetric"] {
        background: #181818;
        border: px solid transparent;
        border-radius: px;
        padding: 15px;
        transition: transform 0.25s ease,
                    box-shadow 0.25s ease,
                    border-color 0.25s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-6px);
        border-color: #00e5ff;
        box-shadow: 0 8px 25px rgba(0, 229, 255, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if page == "Dashboard":
  if "df" in st.session_state:
    df = st.session_state["df"]
  else:
    st.warning("Please upload business data ")
    st.title("AI Business Consultant Dashboard")
    st.stop()

  # Ensure Expenses and Profit exist in df
  if "Expenses" not in df.columns:
    exp_cols = [
        col
        for col in [
            "Rent",
            "Salary",
            "Electricity",
            "Transport",
            "Marketing",
            "Raw_Material",
            "Miscellaneous",
        ]
        if col in df.columns
    ]
    if exp_cols:
      df["Expenses"] = df[exp_cols].sum(axis=1)

  if "Profit" not in df.columns and "Revenue" in df.columns:
    df["Profit"] = df["Revenue"] - df["Expenses"]

  if "Profit Margin(%)" not in df.columns and "Revenue" in df.columns:
    df["Profit Margin(%)"] = (df["Profit"] / df["Revenue"]) * 100

  # Compute overall totals across ALL months
  total_revenue = df["Revenue"].sum() if "Revenue" in df.columns else 0
  total_expenses = df["Expenses"].sum() if "Expenses" in df.columns else 0
  total_profit = total_revenue - total_expenses

  # Overall dataset profit margin
  margin = (
      (total_profit / total_revenue * 100) if total_revenue > 0 else 0
  )

  c1, c2, c3 = st.columns(3)
  c1.metric("💰 Revenue", f"₹{total_revenue:,.0f}")
  c2.metric("💸 Expenses", f"₹{total_expenses:,.0f}")
  c3.metric("📈 Profit", f"₹{total_profit:,.0f}")

  if "Expenses" in df.columns and len(df) > 1:
    latest_exp = df["Expenses"].iloc[-1]
    avg_exp = df["Expenses"].iloc[:-1].mean()
    if latest_exp > (avg_exp * 1.20):
      st.error(
          f"🚨 **Expense Leak Alert:** Latest monthly expense"
          f" (₹{latest_exp:,.0f}) is more than 20% higher than your average"
          " spending!"
      )

  st.subheader("🏢 Business Information")
  if "company_name" in st.session_state:
    st.write("🏢 Company:", st.session_state["company_name"])
    st.write("🏭 Industry:", st.session_state["industry"])
    st.write("👤 Owner:", st.session_state["owner_name"])
    st.write("👥 Employees:", st.session_state["employees"])
    st.write("📍 Location:", st.session_state["location"])
  else:
    st.info(
        "No business profile available. Please fill the Business Profile page."
    )

  industry = st.session_state.get("industry", "Other")
  thresholds = industry_thresholds.get(industry, industry_thresholds["Other"])

  df["Profit Margin(%)"] = df["Profit Margin(%)"].fillna(0)
  health = []

  # RENAMED LOOP VARIABLE 'm' SO IT DOES NOT OVERWRITE 'margin'
  for m in df["Profit Margin(%)"]:
    if m >= thresholds["excellent"]:
      health.append("Excellent")
    elif m >= thresholds["good"]:
      health.append("Good")
    elif m >= thresholds["average"]:
      health.append("Average")
    else:
      health.append("Poor")

  df["Business Health"] = health

  st.write("---")
  st.subheader("🏥 Business Health Score")

  # Calculate health percentage based on overall dataset margin
  if margin <= 0:
    health_score = 10
    health_label = "Critical Risk"
  else:
    health_score = min(int(margin * 3.33), 100)
    if health_score >= 80:
      health_label = "Optimal Performance"
    elif health_score >= 50:
      health_label = "Stable / Moderate"
    else:
      health_label = "Needs Improvement"

  # Display Progress Bar
  st.progress(health_score / 100)

  # Display Health Score Metric Card
  st.metric(
      label="Overall Business Health Index",
      value=f"{health_score} / 100",
      delta=health_label,
  )

  st.subheader("Monthly Business Performance")
  st.dataframe(
      df[[
          "Month",
          "Revenue",
          "Expenses",
          "Profit",
          "Profit Margin(%)",
          "Business Health",
      ]]
  )

  row1_col1, spacer1, row1_col2 = st.columns([1, 0.05, 1])

  with row1_col1:
    st.subheader("Monthly Revenue Trend")
    if "Month" in df.columns and "Revenue" in df.columns:
      st.bar_chart(df.set_index("Month")["Revenue"], color="#0077b6")

  with row1_col2:
    st.subheader("Monthly Expense Trend")
    if "Month" in df.columns and "Expenses" in df.columns:
      st.line_chart(df.set_index("Month")["Expenses"], color="#0077b6")

  row2_col1, spacer2, row2_col2 = st.columns([1, 0.05, 1])

  with row2_col1:
    st.subheader("Expense Breakdown by Category")
    exclude_cols = [
        "Month",
        "Revenue",
        "Expenses",
        "Profit",
        "Profit Margin(%)",
        "Business Health",
    ]
    category_cols = [
        col
        for col in df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    if category_cols:
      category_totals = df[category_cols].sum()

      expense_pie = go.Figure(
          data=[
              go.Pie(
                  labels=category_totals.index,
                  values=category_totals.values,
                  hole=0.4,
              )
          ]
      )

      expense_pie.update_layout(
          paper_bgcolor="#0d0d0d",
          plot_bgcolor="#0d0d0d",
          font=dict(color="#e6e6e6"),
          legend=dict(font=dict(color="#e6e6e6")),
      )

      st.plotly_chart(expense_pie, use_container_width=True)
    else:
      st.info(
          "Upload a CSV with separate expense categories (e.g. Rent, Salaries,"
          " Marketing) to see this breakdown."
      )

  with row2_col2:
    st.subheader("Revenue vs Expenses")
    if {"Month", "Revenue", "Expenses"}.issubset(df.columns):
      fig = go.Figure()

      fig.add_trace(
          go.Bar(
              name="Revenue",
              x=df["Month"],
              y=df["Revenue"],
              marker_color="#00e5ff",
          )
      )
      fig.add_trace(
          go.Bar(
              name="Expenses",
              x=df["Month"],
              y=df["Expenses"],
              marker_color="#0077b6",
          )
      )
      fig.update_layout(
          barmode="group",
          xaxis_title="Month",
          yaxis_title="Amount",
          legend_title="Metric",
          paper_bgcolor="#0d0d0d",
          plot_bgcolor="#1a1a1a",
          font=dict(color="#00e5ff"),
          xaxis=dict(gridcolor="#333333", zerolinecolor="#333333"),
          yaxis=dict(gridcolor="#333333", zerolinecolor="#333333"),
      )
      st.plotly_chart(fig, use_container_width=True)        
# --- NEW ROUTER SECTION FOR GST & DOCUMENT VAULT ---
elif page == "GST & Document Vault":
  render_document_vault()

elif page == "AI Marketing & Promotion":
  st.title("📢 AI Sales & Marketing Assistant")
  st.write(
      "Generate tailored marketing campaigns, discount strategies, and local"
      " offline ideas for your business."
  )

  company = st.session_state.get("company_name", "Your Business")
  industry = st.session_state.get("industry", "Retail")
  location = st.session_state.get("location", "Local Market")

  col_target, col_goal = st.columns(2)
  with col_target:
    campaign_type = st.selectbox(
        "Campaign Focus",
        [
            "Festival / Discount Offer",
            "New Customer Acquisition",
            "Customer Loyalty Program",
        ],
    )
  with col_goal:
    target_discount = st.slider("Target Discount % (if applicable)", 5, 50, 15)

  if st.button("🚀 Generate Marketing Strategy"):
    with st.spinner("Creating custom promotion plan..."):
      prompt = f"""
You are a friendly business guide. Explain everything in VERY SIMPLE, EASY ENGLISH. Do not use hard business terms.

You are an expert local business marketing consultant. 
Create a complete marketing campaign for a local business with these details:
- Business Name: {company}
- Industry: {industry}
- Location: {location}
- Campaign Goal: {campaign_type}
- Target Discount: {target_discount}%

Provide a structured, easy-to-read strategy strictly covering ONLY these 3 sections:
1. 💡 **Campaign Concepts**: The overall theme, creative angle, and branding concept for this campaign.
2. 🏷️ **Discount Strategies**: Tactical offer pricing, bundle structures, and percentage-off recommendations based on the target discount.
3. 📍 **Local Offline Marketing Ideas**: Banners, flyers, local shop tie-ups, or community partnerships suitable for local visibility in {location}.
"""
      try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a creative business marketing strategist."
                        " Respond strictly using the requested 3 sections."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=600,
        )
        st.markdown("### 🎯 Your Custom Marketing Strategy")
        st.info(response.choices[0].message.content)
      except Exception as e:
        st.error(f"Error generating campaign: {e}")

elif page == "Business Profile":
  st.title("Business Profile")

  name = st.text_input("Company Name")
  owner = st.text_input("Owner Name")
  industry = st.selectbox(
      "Industry",
      ["Retail", "Manufacturing", "Services", "Food", "Technology", "Other"],
  )
  location = st.text_input("Location")
  employees = st.number_input("Number of Employees", min_value=0, step=1)
  products_services = st.text_input(
      "Products / Services Offered",
      value=st.session_state.get("products_services", ""),
  )
  target_audience = st.text_input(
      "Target Audience", value=st.session_state.get("target_audience", "")
  )

  if st.button("Save Profile"):
    if name.strip() == "" or owner.strip() == "":
      st.warning("Please enter a Company Name and Owner Name.")
    else:
      conn = get_connection()
      c = conn.cursor()
      c.execute(
          """
                INSERT OR REPLACE INTO business_profile (user_id, company_name, owner_name, industry, location, employees)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
          (
              st.session_state.user_id,
              name.strip(),
              owner.strip(),
              industry,
              location.strip(),
              employees,
          ),
      )
      conn.commit()
      conn.close()

      st.session_state["company_name"] = name.strip()
      st.session_state["owner_name"] = owner.strip()
      st.session_state["industry"] = industry
      st.session_state["location"] = location.strip()
      st.session_state["employees"] = employees
      st.session_state["products_services"] = products_services
      st.session_state["target_audience"] = target_audience

      st.session_state.profile_completed = True
      st.session_state.default_page = "Dashboard"

      st.success("Profile saved successfully!")
      st.rerun()

elif page == "Financial Overview":
  st.title("Financial Overview")

  st.subheader("Enter Today's Business Data")

  entry_date = st.date_input("Date", date.today())
  sales = st.number_input("Today's Total Sales (₹)", min_value=0.0, step=100.0)
  cogs = st.number_input("Cost of Goods Sold (₹)", min_value=0.0, step=100.0)
  other_expenses = st.number_input(
      "Other Expenses (₹)", min_value=0.0, step=100.0
  )

  gross_profit = sales - cogs
  net_profit = gross_profit - other_expenses
  profit_margin = (net_profit / sales) * 100 if sales > 0 else 0

  col1, col2, col3 = st.columns(3)
  col1.metric("Gross Profit", f"₹{gross_profit:,.2f}")
  col2.metric("Net Profit", f"₹{net_profit:,.2f}")
  col3.metric("Profit Margin", f"{profit_margin:.2f}%")

  if st.button("Save Today's Record"):
    success, msg = save_daily_business(
        st.session_state.user_id, entry_date, sales, cogs, other_expenses
    )
    if success:
      st.success("Today's record saved successfully!")
    else:
      st.error(msg)

    st.divider()
    st.subheader("Your Saved Daily Entries")

    daily_df = load_daily_business(st.session_state.user_id)
    if daily_df.empty:
      st.info("No entries saved yet.")
    else:
      st.dataframe(daily_df)

  daily_df = load_daily_business(st.session_state.user_id)
  monthly_df = generate_monthly_summary(daily_df)

  if monthly_df is not None:
    st.session_state["df"] = monthly_df
    st.session_state["business_data"] = monthly_df
    st.session_state["revenue"] = float(monthly_df["Revenue"].iloc[0])
    st.session_state["expenses"] = float(
        monthly_df["COGS"].iloc[0] + monthly_df["Other Expenses"].iloc[0]
    )
    st.session_state["profit"] = float(monthly_df["Profit"].iloc[0])
    st.session_state["profit_margin"] = float(
        monthly_df["Profit Margin(%)"].iloc[0]
    )

elif page == "AI Analysis":
  st.title("AI Analysis")

  revenue = st.session_state.get("revenue", 0)
  expenses = st.session_state.get("expenses", 0)
  profit = revenue - expenses

  margin = (profit / revenue * 100) if revenue > 0 else 0

  # --- REPLACEMENT METRIC KPI CARDS CODE ---
  c1, c2, c3, c4 = st.columns(4)

  c1.metric(
      label="Revenue",
      value=f"₹{int(revenue):,}",
      delta="+12.5% vs last month"  # Green trend badge
  )

  c2.metric(
      label="Expenses",
      value=f"₹{int(expenses):,}",
      delta="-3.2% vs last month",
      delta_color="inverse"  # Shows lower expenses with a green badge
  )

  c3.metric(
      label="Profit",
      value=f"₹{int(profit):,}",
      delta="+8.4% vs last month"  # Green trend badge
  )

  c4.metric(
      label="Profit Margin",
      value=f"{margin:.2f}%"
  )

  st.write("---")
  st.subheader("🏥 Business Health Score")

  # Calculate health percentage (clamped between 0% and 100%)
  if margin <= 0:
    health_score = 10
    health_label = "Critical Risk"
  else:
    health_score = min(int(margin * 3.33), 100)  # 30% margin = 100% health
    if health_score >= 80:
      health_label = "Optimal Performance"
    elif health_score >= 50:
      health_label = "Stable / Moderate"
    else:
      health_label = "Needs Improvement"

  # Display Progress Bar
  st.progress(health_score / 100)

  # Display Health Score Metric Card right below progress bar
  st.metric(
      label="Overall Business Health Index",
      value=f"{health_score} / 100",
      delta=health_label,
  )
 
  st.divider()

  st.subheader("Monthly Business Performance")
  st.dataframe(df)

  st.subheader("Revenue Analysis")
  st.info("Revenue reflects the overall earning performance of your business.")

  st.subheader("Revenue Improvement Suggestions")

  if margin == 0.00:
    st.error("Can't analyze revenue. Please enter the business data.")
  elif margin < 5:
    st.warning("""
        Profit margin is very low. Immediate action is needed to increase revenue:
        - **Slight Price Adjustment:** Increase prices by 3% to 5% on high-demand items to boost income immediately.
        - **Product Bundling:** Combine slow-moving items with popular products to raise the average order value.
        - **Upselling & Cross-selling:** Offer related add-ons or premium versions during customer purchases.
        """)
  elif margin < 15:
    st.info("""
        Profit margin is satisfactory, but can be improved:
        - **Customer Retention:** Send follow-up deals and loyalty offers to existing customers to encourage repeat orders.
        - **Focus on Best-Sellers:** Reinvest marketing efforts into the top 20% of products that generate the most income.
        """)
  else:
    st.success("""
        Excellent profit margin! Reinvest profits to scale your operations:
        - **Expand Marketing Reach:** Increase budget on proven sales channels.
        - **Introduce Premium Offerings:** Launch high-end product variants to capture higher-paying customers.
        """)

  st.subheader("Expense Analysis & Material Cost Optimization")

  if expenses == 0 and revenue == 0:
    st.error("Please enter your expenses and business data.")
  elif expenses >= revenue:
    st.warning("""
        High operational expenses detected! Take these immediate cost-control measures:
        - **Raw Material & COGS Optimization:** Bulk-purchase essential raw materials or negotiate volume discounts with suppliers to lower unit costs.
        - **Cut Operational Waste:** Review supply chain processes to minimize material spoilage, damage, or manufacturing waste.
        - **Audit Fixed Expenses:** Freeze non-essential subscriptions, unnecessary utility usage, and low-performing ad campaigns.
        """)
  else:
    st.success("""
        Expenses are well managed relative to revenue. Maintain cost efficiency with these steps:
        - **Supplier Terms:** Request longer credit payment terms from material suppliers to keep cash flow flexible.
        - **Automate Routine Tasks:** Use low-cost tools for inventory tracking and billing to save labor costs as you scale.
        """)

  st.subheader("Profitability Analysis")

  if margin == 0:
    st.error("Please enter Business data.")
  elif margin < 0:
    st.error("""
        Business is operating at a loss.
        
        Immediate cost reduction and revenue improvement are required. Prioritize selling high-margin inventory and freeze non-essential spending.
        """)
  elif margin < 10:
    st.warning(
        "Low profitability. Focus on trimming operational overhead and pushing"
        " higher-margin products."
    )
  elif margin < 20:
    st.info(
        "Average profitability. Solid baseline, but optimizing product"
        " manufacturing costs will increase net returns."
    )
  else:
    st.success(
        "Excellent profitability. Your business model generates strong net"
        " returns."
    )

  st.subheader("Product Expansion & Target Audience Strategy")

  st.info("""
    **1. How to Expand Your Product Line (Based on Customer Choice):**
    - **Gather Customer Feedback:** Use polls, post-purchase surveys, or direct chat feedback to ask buyers what related items they are searching for.
    - **Add Complementary Products:** Introduce accessories or consumable supplies that pair naturally with your current core items.
    - **Test Small Inventory Batches:** Introduce new items in limited quantities first to gauge demand before committing capital to bulk stock.
    - **Slightly Raise Prices:** Increase rates slightly on items that sell every day.
    - **Add New Products:** Start selling new items that your customers ask for.

    **2. How to Attract & Increase Target Audience:**
    - **Targeted Social Advertising:** Run ads tailored directly to the demographics and interests of your top 20% most active customers.
    - **Leverage Customer Reviews:** Display photos, video testimonials, and positive ratings prominently on sales pages to convert new viewers into buyers.
    - **Referral Programs:** Offer discounts or cash-back incentives to existing customers who refer new buyers to your business.
    - **Take Orders on Mobile:** Let customers place orders easily through WhatsApp or phone calls.
    """)

  st.subheader("Growth Prediction")

  if margin >= 25:
    st.success(
        "Predicted Business Growth: 20%-30%. Reinvest profits to expand market"
        " reach."
    )
  elif margin >= 15:
    st.info(
        "Predicted Business Growth: 20%. Business is in a solid position to"
        " acquire new customers."
    )
  elif margin >= 5:
    st.info(
        "Predicted Business Growth: 5%-10%. Focus on increasing profit margins"
        " to unlock faster growth."
    )
  elif margin > 0:
    st.warning(
        "Business growth may remain slow unless profitability and material cost"
        " efficiency improve."
    )
  else:
    st.error(
        "Business is currently at high risk. Focus immediately on revenue"
        " recovery and material cost reduction."
    )

elif page == "Business Data Manager":
  st.title("Business Data Manager")

  st.write("Upload a CSV with at least 3 months of business data.")

  uploaded_file = st.file_uploader("Upload CSV", type=["csv", "xlsx"])

  if uploaded_file is not None:
    file_id = save_uploaded_file(st.session_state.user_id, uploaded_file)

    if uploaded_file.name.endswith(".csv"):
      df = pd.read_csv(uploaded_file)
    else:
      df = pd.read_excel(uploaded_file)

    exclude_cols = ["Month", "Revenue", "Expenses", "Profit", "Profit Margin(%)"]

    if "Expenses" in df.columns and df["Expenses"].fillna(0).sum() != 0:
      pass
    else:
      expense_cols = [
          col
          for col in df.columns
          if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
      ]
      df["Expenses"] = df[expense_cols].sum(axis=1)

    df["Profit"] = df["Revenue"] - df["Expenses"]
    df["Profit Margin(%)"] = (df["Profit"] / df["Revenue"]) * 100
    st.session_state["df"] = df
    if len(df) < 3:
      st.error("Please upload at least 3 months of data.")
    else:
      st.success("CSV uploaded successfully.")
      st.dataframe(df)

      st.session_state["business_data"] = df
      st.session_state["df"] = df

      if "Revenue" in df.columns:
        st.session_state["revenue"] = int(df["Revenue"].sum())

      if "Expenses" in df.columns:
        st.session_state["expenses"] = int(df["Expenses"].sum())

      st.success("Dashboard values updated from uploaded data.")

      success, msg = save_financial_data(
          st.session_state.user_id, df, file_id=file_id
      )
      if success:
        set_last_file(st.session_state.user_id, file_id)
      else:
        st.error(msg)
  else:
    st.subheader("Upload Income Statement")
    st.write(
        "Upload a PDF income statement. The AI will read it and pull "
        "out the monthly or yearly figures - you'll get to check and "
        "fix anything before it's saved."
    )

    doc_file = st.file_uploader("Upload PDF", type=["pdf"], key="doc_upload")

    if doc_file is not None:
      if st.button("Extract Data"):
        with st.spinner("Reading PDF..."):
          raw_text = extract_text_from_pdf(doc_file)

        if not raw_text:
          st.error(
              "Couldn't read any text from this PDF. "
              "It may be a scanned/image-based PDF, which isn't supported yet."
          )
        else:
          with st.spinner("Extracting financial data with AI..."):
            extracted = ask_ai_to_extract_financials(raw_text)

          if extracted is None or len(extracted) == 0:
            st.error("Couldn't extract structured data from this statement.")
          else:
            st.session_state["pending_extraction"] = extracted
            st.session_state["pending_file"] = doc_file

    if "pending_extraction" in st.session_state:
      st.subheader("Review extracted data - edit anything that's wrong")

      preview_df = pd.DataFrame(st.session_state["pending_extraction"])
      edited_df = st.data_editor(
          preview_df,
          num_rows="dynamic",
          use_container_width=True,
          key="extraction_editor",
      )

      if st.button("Confirm and Save"):
        final_df = edited_df.rename(
            columns={
                "month": "Month",
                "revenue": "Revenue",
                "expenses": "Expenses",
            }
        )

        final_df["Profit"] = final_df["Revenue"] - final_df["Expenses"]
        final_df["Profit Margin(%)"] = (
            final_df["Profit"] / final_df["Revenue"]
        ) * 100

        doc_file_for_save = st.session_state.get("pending_file", doc_file)
        file_id = save_uploaded_file(
            st.session_state.user_id, doc_file_for_save
        )

        success, msg = save_financial_data(
            st.session_state.user_id, final_df, file_id=file_id
        )

        if success:
          st.session_state["df"] = final_df
          st.session_state["business_data"] = final_df
          set_last_file(st.session_state.user_id, file_id)
          st.success("Data saved successfully!")
          del st.session_state["pending_extraction"]
        else:
          st.error(msg)

elif page == "Database":
  st.subheader("Your Previously Uploaded Files")

  files_df = list_user_files(st.session_state.user_id)

  if files_df.empty:
    st.write("No files uploaded yet.")
  else:
    st.dataframe(files_df[["Filename", "Type", "Uploaded At"]])

    selected_filename = st.selectbox(
        "View data from a previous file", files_df["Filename"]
    )

    if st.button("Load Selected File"):
      selected_file_id = files_df[files_df["Filename"] == selected_filename][
          "file_id"
      ].iloc[0]

      loaded_df = load_data_by_file(int(selected_file_id))
      set_last_file(st.session_state.user_id, int(selected_file_id))

      if loaded_df is not None:
        st.session_state["df"] = loaded_df
        st.session_state["business_data"] = loaded_df

        if "Revenue" in loaded_df.columns:
          st.session_state["revenue"] = int(loaded_df["Revenue"].sum())
        if "Expenses" in loaded_df.columns:
          st.session_state["expenses"] = int(loaded_df["Expenses"].sum())

        st.success(f"Loaded data from {selected_filename}")
        st.rerun()
      else:
        st.warning("No data found for this file.")

elif page == "AI Financial Advisor Chatbot":
  st.title("Ask AI Consultant")
  if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

  df = st.session_state.get("df", None)

  if df is None:
    st.warning("Please upload business data first.")
    st.stop()

  month_col = None
  sales_col = None

  for col in df.columns:
    name = col.lower().strip()
    if "month" in name:
      month_col = col
    elif "sale" in name or "revenue" in name:
      sales_col = col

  if month_col is None or sales_col is None:
    st.error("Month or Sales column not found in the uploaded CSV")
    st.stop()

  for col in ["Total Expenses", "Profit", "Profit Margin(%)"]:
    if col in df.columns:
      df = df.drop(columns=[col])

  expense_columns = [
      col
      for col in df.columns
      if col != month_col
      and col != sales_col
      and pd.api.types.is_numeric_dtype(df[col])
  ]

  df["Total Expenses"] = df[expense_columns].sum(axis=1)
  df["Profit"] = df[sales_col] - df["Total Expenses"]
  df["Profit Margin(%)"] = (df["Profit"] / df[sales_col]) * 100

  st.markdown("### 💡 Common Questions")

  example_questions = [
      "How can I improve my profit margin?",
      "What are my biggest expense areas?",
      "Is my business growing compared to last month?",
      "How do I reduce operational costs?",
      "What industry benchmarks should I compare against?",
  ]

  selected_question = st.selectbox("", [""] + example_questions)

  if selected_question == "":
    user_question = st.text_input("Ask your question:")
  else:
    user_question = selected_question

  if st.button("Ask AI"):
    if user_question.strip() == "":
      st.warning("Please enter a question.")
    else:
      with st.spinner("Thinking..."):
        display_df = df[[
            month_col,
            sales_col,
            "Total Expenses",
            "Profit",
            "Profit Margin(%)",
        ]].copy()
        display_df.columns = [
            "Month",
            "Revenue",
            "Expenses",
            "Profit",
            "Profit Margin(%)",
        ]

        company = st.session_state.get("company_name", "Unknown")
        owner = st.session_state.get("owner_name", "Unknown")
        industry = st.session_state.get("industry", "General")
        location = st.session_state.get("location", "Unknown")
        employees = st.session_state.get("employees", "Unknown")

        context = f"""
    You are an AI Business Consultant.

    Business Profile:
    Company: {company}
    Owner: {owner}
    Industry: {industry}
    Location: {location}
    Employees: {employees}

    Below is the uploaded business data:
 
    {df.to_string(index=False)}

    You can answer questions using BOTH:
    1. The business profile (company, industry, location, employees).
    2. The uploaded business data.

    If the user asks about the business profile, answer from the profile.
    If the user asks about the uploaded financial data, answer from the data.
    If the user asks a general business or finance question, answer using your knowledge.
    If information is unavailable, clearly say so.

    Business Type:
    {industry}

    Uploaded Business Data:
    {df.to_string(index=False)}

    Instructions:

    - If the question is about the uploaded business data, answer using the uploaded data.
    - If the question is about business, finance, marketing, profit, taxation, accounting, inventory or entrepreneurship, answer using your business knowledge and the uploaded data if relevant.
    - If the question is a general question (Python, AI, history, science, current affairs, coding, etc.), answer it normally like ChatGPT.
    - If the uploaded data does not contain the requested business information, simply say:
    "I couldn't find that information in the uploaded business data, but here's the general information."

    User Question:
    {user_question}
    """
        try:
          response = client.chat.completions.create(
              model="llama-3.3-70b-versatile",
              messages=[
                  {
                      "role": "system",
                      "content": (
                          "You are a professional AI Business Consultant."
                      ),
                  },
                  {"role": "user", "content": context},
              ],
              temperature=0.2,
              max_tokens=500,
          )
          answer = response.choices[0].message.content
          st.success(answer)
          st.session_state["chat_history"].append(
              {"question": user_question, "answer": answer}
          )

        except Exception as e:
          st.error(f"Error: {e}")

  rule_question = st.text_input(
      "Or ask a quick analytics question", key="rule_question"
  )
  if rule_question:
    q = rule_question.lower().strip()

    matched_row = None
    matched_label = None

    unique_months = df[month_col].astype(str).unique()

    for val in unique_months:
      val_clean = val.strip().lower()
      if not val_clean:
        continue
      if val_clean in q or val_clean[:3] in q.split():
        mask = df[month_col].astype(str).str.strip().str.lower() == val_clean
        matched_row = df[mask].iloc[0]
        matched_label = matched_row[month_col]
        break

    if matched_row is not None:
      if "margin" in q:
        st.success(
            f"{matched_label} Profit Margin ="
            f" {matched_row['Profit Margin(%)']:.2f}%"
        )
      elif "profit" in q:
        st.success(f"{matched_label} Profit = ₹{matched_row['Profit']:,.2f}")
      elif "expense" in q:
        st.success(
            f"{matched_label} Expenses = ₹{matched_row['Total Expenses']:,.2f}"
        )
      else:
        st.success(f"{matched_label} Sales = ₹{matched_row[sales_col]:,.2f}")

    elif "total" in q and "sale" in q:
      st.success(f"Total Revenue = ₹{df[sales_col].sum():,.0f}")
    elif "total" in q and "profit" in q:
      st.success(f"Total Profit = ₹{df['Profit'].sum():,.0f}")
    elif "total" in q and "expense" in q:
      st.success(f"Total Expenses = ₹{df['Total Expenses'].sum():,.0f}")
    elif "highest" in q and "sale" in q:
      row = df.loc[df[sales_col].idxmax()]
      st.success(
          f"{row[month_col]} has the highest sales of ₹{row[sales_col]:,.2f}"
      )
    elif "lowest" in q and "sale" in q:
      row = df.loc[df[sales_col].idxmin()]
      st.success(
          f"{row[month_col]} has the lowest sales of ₹{row[sales_col]:,.2f}"
      )
    elif "average" in q and "sale" in q:
      st.success(f"Average Sales = ₹{df[sales_col].mean():,.2f}")
    elif "average" in q and "profit" in q:
      st.success(f"Average Profit = ₹{df['Profit'].mean():,.2f}")
    else:
      st.warning(
          "Sorry, I couldn't understand your question. Try rephrasing — e.g."
          " 'sales in June' or 'total profit'."
      )

  st.markdown("### Chat History")
  for entry in reversed(st.session_state["chat_history"]):
    st.markdown(f"**You:** {entry['question']}")
    st.markdown(f"**AI:** {entry['answer']}")
    st.markdown("---")