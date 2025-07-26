import streamlit as st
import os
import time
import platform
import re
import speech_recognition as sr
import pyttsx3
import threading
from dotenv import load_dotenv

import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="📄 PDF Voice Assistant", layout="centered")
st.markdown("<h2 style='text-align: center;'>🎙️ Voice-Enabled AI PDF Chatbot</h2>", unsafe_allow_html=True)
st.caption("Upload PDFs and ask questions using voice or text. Summarize, compare, extract info, and hear the answer spoken back!")

# 🧹 Clean text for TTS

def clean_for_tts(text):
    text = re.sub(r'[_*#<>`~\\\-]', '', text)
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'[^\w\s.,?!]', '', text)
    return text

# 🔊 Speak text in background

def speak_text_async(text):
    def run_tts():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 170)
            engine.setProperty('volume', 1.0)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            st.warning(f"🔊 Text-to-speech error: {e}")
    threading.Thread(target=run_tts).start()

# 🎤 Voice Input

def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Please speak.")
        try:
            audio = recognizer.listen(source, timeout=5)
            query = recognizer.recognize_google(audio)
            st.success(f"🗣️ You said: {query}")
            return query
        except sr.UnknownValueError:
            st.warning("⚠️ Could not understand audio.")
        except sr.RequestError as e:
            st.error(f"❌ Speech recognition error: {e}")
        except sr.WaitTimeoutError:
            st.warning("⏱️ Listening timed out.")
    return ""

# 🔍 Task Detection

def detect_task(user_input):
    task_keywords = {
        "summarize": "summarize",
        "summary": "summarize",
        "bullet": "bullet_points",
        "key points": "bullet_points",
        "compare": "compare",
        "main topic": "main_topic",
    }
    for keyword, task in task_keywords.items():
        if keyword in user_input.lower():
            return task
    return "qa"

# ✨ Prompt Handlers

def handle_summarization(text, llm):
    return llm.invoke("Summarize the document with markdown headings:\n\n" + text[:5000])

def handle_bullet_points(text, llm):
    return llm.invoke("List key points in bullet form:\n\n" + text[:5000])

def handle_comparison(text1, text2, llm):
    return llm.invoke(f"Compare these documents:\n\nDocument 1:\n{text1[:5000]}\n\nDocument 2:\n{text2[:5000]}")

# 📄 Upload PDFs
pdf_files = st.file_uploader("📄 Upload your PDF files", type=["pdf"], accept_multiple_files=True)
voice_enabled = st.toggle("🔈 Enable Voice Output", value=True)

if pdf_files:
    with st.spinner("📚 Processing PDFs..."):
        all_docs = []
        for pdf in pdf_files:
            with open(pdf.name, "wb") as f:
                f.write(pdf.read())
            loader = PyPDFLoader(pdf.name)
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(pages)
            for doc in docs:
                doc.metadata["source"] = pdf.name
            all_docs.extend(docs)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(all_docs, embeddings)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=st.session_state.memory,
            return_source_documents=True,
            output_key="answer"
        )

        if "chat" not in st.session_state:
            st.session_state.chat = []

        # 🎙️ Voice Input
        st.markdown("🎙️ Or ask with your voice")
        if st.button("🎤 Speak Now"):
            voice_query = get_voice_input()
            if voice_query:
                st.session_state.chat.append(("user", voice_query))
                query = voice_query
            else:
                query = None
        else:
            query = None

        # 💬 Text Input
        typed_query = st.chat_input("💬 Type your question here...")
        if typed_query:
            st.session_state.chat.append(("user", typed_query))
            query = typed_query

        # 🧠 Process Query
        if "query" in locals() and query:
            task_type = detect_task(query)
            with st.spinner("🤖 Thinking..."):
                if task_type == "summarize":
                    content = " ".join([doc.page_content for doc in all_docs])
                    answer = handle_summarization(content, llm)
                elif task_type == "bullet_points":
                    content = " ".join([doc.page_content for doc in all_docs])
                    answer = handle_bullet_points(content, llm)
                elif task_type == "compare":
                    if len(pdf_files) >= 2:
                        doc1 = " ".join([doc.page_content for doc in all_docs if doc.metadata["source"] == pdf_files[0].name])
                        doc2 = " ".join([doc.page_content for doc in all_docs if doc.metadata["source"] == pdf_files[1].name])
                        answer = handle_comparison(doc1, doc2, llm)
                    else:
                        answer = "⚠️ Please upload at least 2 PDFs to compare."
                else:
                    result = qa_chain.invoke({"question": query})
                    answer = result["answer"]
                    sources = set([doc.metadata.get("source") for doc in result["source_documents"]])
                    if sources:
                        answer += "\n\n📄 **Source(s)**: " + ", ".join(sources)

            st.session_state.chat.append(("bot", answer))
            if voice_enabled:
                speak_text_async(clean_for_tts(answer))

        # 💬 Show Chat
        for role, msg in st.session_state.chat:
            if role == "user":
                st.markdown(f"<div style='text-align: right; margin: 8px 0;'><div style='display: inline-block; background-color: #f0f2f6; color: black; padding: 10px 15px; border-radius: 20px; max-width: 75%;'>🧑‍🎓 {msg}</div></div>", unsafe_allow_html=True)
            else:
                with st.expander("📄 Click to expand response"):
                    st.markdown(msg.content if hasattr(msg, "content") else msg, unsafe_allow_html=True)

# 🔁 Reset Chat
st.markdown("---")
if st.button("🔁 Reset Chat"):
    st.session_state.chat = []
    if "memory" in st.session_state:
        st.session_state.memory.clear()
    st.success("✅ Chat history cleared!")
    time.sleep(1)
    st.rerun()
