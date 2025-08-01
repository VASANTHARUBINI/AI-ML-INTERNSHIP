# 🎤 Voice-Enabled AI PDF Chatbot

This project is a **voice-interactive AI chatbot** that allows users to upload PDF documents, ask questions using **voice or text**, and receive both **textual and spoken answers**. It is powered by **Google Gemini**, **LangChain**, and speech tools like **SpeechRecognition** and **pyttsx3**.

---

## ✨ Features

- 📄 Upload and process multiple PDF documents
- 🎤 Ask questions via microphone or chat input
- 🧠 Maintains conversation memory across queries
- 🔍 Semantic search using FAISS & Gemini embeddings
- 🤖 Gemini 1.5 Flash LLM for question answering
- 🔊 Answers are spoken aloud (text-to-speech)
- 🧠 Understands tasks like summarize, compare, bullet points
- 🎨 Styled interface using Streamlit

---

## 🚀 Tech Stack

- `Python`
- `Streamlit` – Web interface
- `SpeechRecognition` – Voice input
- `pyttsx3`, `pydub` – Text-to-speech
- `LangChain` – LLM chaining and memory
- `FAISS` – Vector-based document search
- `Google Gemini` – Embedding + chat model
- `PyMuPDF`, `PyPDFLoader` – PDF text extraction

---
## Sample Output

     🎤 You said: “Compare both uploaded documents.”

     🤖 Response: "Document 1 outlines the AI syllabus; Document 2 provides use cases of AI in healthcare and education."

## 🔐 Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/VASANTHARUBINI/AI-ML-INTERNSHIP.git
   cd AI-ML-INTERNSHIP/PROJECT-2
