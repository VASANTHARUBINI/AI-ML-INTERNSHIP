# 🛒 AI E-Commerce Support Chatbot

This project is an **AI-powered chatbot** for E-Commerce platforms that can handle customer queries such as order tracking, cancellation, refunds, product search, and FAQs. It simulates a real-world customer support assistant using NLP and ML techniques like **TF-IDF**, **Cosine Similarity**, and **fuzzy matching**.

---

## ✨ Features

- 📦 Track orders by Order ID  
- ❌ Cancel orders with reason handling  
- 💰 Process refund queries  
- 🔍 Search for products using approximate matches  
- ❓ Answer FAQs using TF-IDF + cosine similarity  
- 🧠 Maintains session chat history  
- 💬 Interactive Streamlit chat UI  

---

## 📂 Project Files

- `orders.csv` – Sample order history  
- `products.csv` – Product catalog  
- `faq.csv` – Predefined frequently asked questions  
- `app.py` – Main chatbot application

---

## 🚀 Tech Stack

- `Python`  
- `Streamlit` – Web interface  
- `scikit-learn` – TF-IDF and cosine similarity  
- `difflib` – For fuzzy product search  
- `Pandas` – Data handling for orders/products

---

## Sample Output

    💬 User: Cancel my order #12345
    
    🤖 Bot: Order #12345 (Smart Watch) has been cancelled. Refund will be processed within 3–5 business days.


## 🔐 Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/VASANTHARUBINI/AI-ML-INTERNSHIP.git
   cd AI-ML-INTERNSHIP/PROJECT-3
