import streamlit as st
import pandas as pd
import re
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load CSVs
orders_df = pd.read_csv("orders.csv")
products_df = pd.read_csv("products.csv")
faq_df = pd.read_csv("faq.csv")

# Initialize chat history and cancelled order memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "cancelled_orders" not in st.session_state:
    st.session_state.cancelled_orders = set()
if "cancellation_context" not in st.session_state:
    st.session_state.cancellation_context = {}

# Helper functions
def extract_order_id(text):
    match = re.search(r"#?(\d{5})", text)
    return match.group(1) if match else None

def faq_response(query):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(faq_df["question"])
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix)
    max_idx = similarity.argmax()
    if similarity[0, max_idx] > 0.05:
        return faq_df.iloc[max_idx]["answer"]
    return None

def find_closest_product(query):
    all_products = products_df["product_name"].str.lower().tolist()
    match = get_close_matches(query.lower(), all_products, n=1, cutoff=0.5)
    if match:
        prod_row = products_df[products_df["product_name"].str.lower() == match[0]].iloc[0]
        return f"\U0001F6CD\uFE0F Product: {prod_row['product_name']}\n\U0001F4CD Sizes: {prod_row['available_sizes']}\n\U0001F4E6 Stock: {prod_row['stock_status']}"
    return None

def respond_to_query(query):
    query = query.strip()
    order_id = extract_order_id(query)

    # Cancellation follow-up
    if st.session_state.cancellation_context.get("awaiting_reason"):
        current_order_id = st.session_state.cancellation_context.get("order_id")
        if current_order_id:
            st.session_state.cancelled_orders.add(current_order_id)
            st.session_state.cancellation_context = {}
            return f"\u2705 Your order #{current_order_id} has been cancelled successfully. \U0001F4B8 A refund will be processed in 3–5 business days."

    # Cancellation intent
    if "cancel my order" in query.lower() and order_id:
        order_row = orders_df[orders_df["order_id"] == int(order_id)]
        if not order_row.empty:
            if order_id in st.session_state.cancelled_orders:
                return f"❌ Order #{order_id} was already cancelled successfully."
            row = order_row.iloc[0]
            st.session_state.cancellation_context = {"order_id": order_id, "awaiting_reason": True}
            cancel_count = sum(orders_df["order_id"].astype(str).isin(st.session_state.cancelled_orders))
            return f"\U0001F4E6 You're requesting to cancel Order #{order_id} ({row['product_name']}). This is your {cancel_count + 1} cancellation in recent times. Please tell us the reason for cancellation to proceed with refund or exchange options."
        else:
            return f"❗ Order ID not found. Please check again."

    # Refund intent
    if "refund" in query.lower():
        if st.session_state.cancelled_orders:
            last_cancel = list(st.session_state.cancelled_orders)[-1]
            return f"\U0001F4B8 A refund for Order #{last_cancel} will be processed in 3–5 business days."
        return "No recent cancelled orders found to refund."

    # Tracking intent
    if ("track" in query.lower() or "status" in query.lower()) and order_id:
        order_row = orders_df[orders_df["order_id"] == int(order_id)]
        if not order_row.empty:
            row = order_row.iloc[0]
            return f"\U0001F4E6 Order #{row['order_id']} ({row['product_name']}) is currently '{row['status']}' and was placed on {row['delivery_date']}."

    # Handle raw order ID queries after cancellation
    if order_id:
        order_row = orders_df[orders_df["order_id"] == int(order_id)]
        if not order_row.empty:
            if order_id in st.session_state.cancelled_orders:
                return f"❌ Order #{order_id} was already cancelled successfully."
            row = order_row.iloc[0]
            return f"\U0001F4E6 Order #{row['order_id']} ({row['product_name']}) is currently '{row['status']}' and was placed on {row['delivery_date']}."

    # Product Search
    prod_response = find_closest_product(query)
    if prod_response:
        return prod_response

    # FAQ check
    faq_ans = faq_response(query)
    if faq_ans:
        return faq_ans

    return "🤖 I'm sorry, I didn't quite understand that. Could you rephrase your request?"

# UI
st.set_page_config(page_title="🛍️ E-Commerce AI Support Bot", layout="wide")
st.title("🛍️ E-Commerce AI Support Chatbot (Chat Style)")

# Display chat history
for user, bot in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user)
    with st.chat_message("assistant"):
        st.markdown(bot)

# Chat input
user_input = st.chat_input("Type your message...")
if user_input:
    response = respond_to_query(user_input)
    st.session_state.chat_history.append((user_input, response))
    st.experimental_rerun()
