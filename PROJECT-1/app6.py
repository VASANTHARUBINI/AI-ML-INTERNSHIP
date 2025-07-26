import streamlit as st
import os
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

# Streamlit Page Config
st.set_page_config(page_title="PDF-BOT Q&A", page_icon="📄", layout="centered")
st.markdown("<h2 style='text-align: center;'>✨ PDF BASED-AI BOT (Multi-PDF) ✨</h2>", unsafe_allow_html=True)
st.caption("Ask anything across multiple PDFs.✨")

# Show Welcome Message
if "welcomed" not in st.session_state:
    st.session_state.welcomed = True
    st.info("👋 Hello! Upload 1 or more PDFs and ask me anything!")

# Upload Multiple PDFs
pdf_files = st.file_uploader("📄 Upload your PDFs", type=["pdf"], accept_multiple_files=True)

# Small Talk Handler
def handle_small_talk(query):
    query = query.lower().strip()
    if query in ["hi", "hello", "hey"]:
        return "👋 Hello! I'm your AI assistant. How can I help you today?"
    elif query in ["bye", "goodbye", "see you", "exit"]:
        return "👋 Goodbye! Have a great day ahead. 😊"
    elif "thank" in query:
        return "You're welcome! 😊 Let me know if you need anything else."
    elif "who are you" in query:
        return "I'm your PDF AI Assistant. Upload PDFs and ask me anything!"
    elif "how are you" in query:
        return "I'm always learning and ready to help you. 😊"
    return None

# Process PDFs and Setup LLM
if pdf_files:
    with st.spinner("🔍 Processing your documents..."):
        all_docs = []

        for pdf in pdf_files:
            with open(pdf.name, "wb") as f:
                f.write(pdf.read())

            loader = PyPDFLoader(pdf.name)
            pages = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(pages)

            # Add filename as source
            for doc in docs:
                doc.metadata["source"] = pdf.name

            all_docs.extend(docs)

        # Create Vector Store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(all_docs, embeddings)

        # Gemini Chat Model
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

        # Memory Setup
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True, output_key="answer"
            )

        # QA Chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=st.session_state.memory,
            return_source_documents=True,
            output_key="answer"
        )

        # Chat History
        if "chat" not in st.session_state:
            st.session_state.chat = []

        # Chat Input
        query = st.chat_input("Ask your PDFs a question...")

        if query:
            st.session_state.chat.append(("user", query))

            # Handle small talk
            small_talk_response = handle_small_talk(query)
            if small_talk_response:
                answer = small_talk_response
            else:
                with st.spinner("🤖 BOT is thinking..."):
                    result = qa_chain.invoke({"question": query})
                    answer = result["answer"]

                    # Optional: Source document tracking
                    sources = set([doc.metadata.get("source") for doc in result["source_documents"]])
                    if sources:
                        answer += "\n\n📄 **Source(s)**: " + ", ".join(sources)

            st.session_state.chat.append(("bot", answer))

        # Display Chat History
        for role, msg in st.session_state.chat:
            if role == "user":
                st.markdown(
                    f"""
                    <div style='text-align: right; margin: 8px 0;'>
                        <div style='display: inline-block; background-color: #f0f2f6; color: black;
                                    padding: 10px 15px; border-radius: 20px; max-width: 75%;'>
                            🧑‍🎓 {msg}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style='text-align: left; margin: 8px 0;'>
                        <div style='display: inline-block; background-color: #262730; color: white;
                                    padding: 10px 15px; border-radius: 20px; max-width: 75%;'>
                            🤖 {msg}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# Optional: Reset button
st.markdown("---")
if st.button("🔁 Reset Chat"):
    st.session_state.chat = []
    st.session_state.memory.clear()
    st.success("Chat history cleared!")
