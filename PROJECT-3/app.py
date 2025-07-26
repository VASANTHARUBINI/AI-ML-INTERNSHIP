import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#  Page Setup

st.set_page_config(page_title="🛍️ AI Product Chatbot", layout="wide")
st.title("🛍️ AI-Powered Product Chatbot")
st.caption("Ask about any product — like price, color, availability, etc.")


#  Load product data

@st.cache_data
def load_data():
    df = pd.read_csv("products_50_named.csv")
    return df

df = load_data()


#  Build TF-IDF model

@st.cache_resource
def build_vectorizer(data):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data["description"])
    return vectorizer, vectors

vectorizer, vectors = build_vectorizer(df)


#  Search Function

def search_products(query, vectorizer, vectors, df):
    query_vec = vectorizer.transform([query])
    sim_scores = cosine_similarity(query_vec, vectors).flatten()
    top_index = sim_scores.argmax()  # Get the most relevant product index
    confidence = sim_scores[top_index]

    # Set a confidence threshold (optional)
    if confidence < 0.1:
        return None, 0.0
    else:
        return df.iloc[top_index], confidence


#  User Query Input

query = st.text_input("🔎 Ask about a product:")


#  Response Display

if query:
    result, score = search_products(query, vectorizer, vectors, df)

    if result is not None:
        st.success("✅ Found a product that matches your query!")
        st.subheader(result["name"])
        st.write(result["description"])
        st.markdown(f"💰 **Price:** ₹{result['price']}")
        st.markdown(f"🎨 **Color:** {result['color']} | 📏 **Size:** {result['size']}")
        st.markdown(f"📦 **Availability:** {result['availability']} | 🛡️ **Warranty:** {result['warranty']}")
        st.caption(f"🔍 Confidence Score: {score:.2f}")
    else:
        st.warning("❌ Sorry, I couldn't find a product that matches your query.")
