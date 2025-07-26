import streamlit as st
import fitz  # PyMuPDF
import os
import google.generativeai as genai
import speech_recognition as sr
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load Gemini Pro model
model = genai.GenerativeModel("gemini-2.5-pro")

# ====================== UTILS ======================

@st.cache_data
def extract_text(file_path):
    doc = fitz.open(file_path)
    return " ".join([page.get_text() for page in doc])

def chunk_text(text, max_len=1500):
    words = text.split()
    return [" ".join(words[i:i+max_len]) for i in range(0, len(words), max_len)]

def summarize(text):
    chunks = chunk_text(text)
    summaries = []
    for chunk in chunks:
        prompt = f"""
Act as a professional teacher. Summarize the following class notes in simple language with bullet points:

{chunk}
"""
        response = model.generate_content(prompt)
        summaries.append(response.text)
    return "\n".join(summaries)

def generate_structured_quiz(text):
    prompt = f"""
Generate 3 multiple choice questions with 4 options. Format each like:
Q: Question?
a) Option A
b) Option B
c) Option C
d) Option D
Answer: b
From this content:
{text}
"""
    raw = model.generate_content(prompt).text
    pattern = r"Q:\s*(.*?)\n\s*a\)\s*(.*?)\n\s*b\)\s*(.*?)\n\s*c\)\s*(.*?)\n\s*d\)\s*(.*?)\n\s*Answer:\s*([abcd])"
    matches = re.findall(pattern, raw, re.DOTALL)

    quiz = []
    for match in matches:
        q = match[0].strip()
        options = {
            'a': match[1].strip(),
            'b': match[2].strip(),
            'c': match[3].strip(),
            'd': match[4].strip()
        }
        answer = match[5].strip()
        quiz.append({"question": q, "options": options, "answer": answer})
    return quiz

def generate_flashcards(text):
    prompt = f"""
From these notes, create 5 flashcards in Q&A format to help students learn:

{text}
"""
    return model.generate_content(prompt).text

def answer_question(text, query):
    prompt = f"""
You are an expert AI teacher. Use the notes below to answer the student's question. Give the answer clearly in 3 bullet points.

Notes:
{text}

Question:
{query}
"""
    return model.generate_content(prompt).text

def save_bookmark(q, a):
    os.makedirs("data", exist_ok=True)
    with open("data/bookmarks.txt", "a", encoding="utf-8") as f:
        f.write(f"🔖 Q: {q}\nA: {a}\n\n")

def load_bookmarks():
    path = "data/bookmarks.txt"
    if not os.path.exists(path):
        return "No bookmarks yet. Ask and save a question to bookmark it."
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        return content if content else "No bookmarks yet. Ask and save a question to bookmark it."

def voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak your question.")
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except:
        return "❌ Could not recognize your voice."

# ====================== UI ======================

st.set_page_config(page_title="📚 AI Study Assistant", layout="centered")
st.title("📚 AI Study Assistant (Gemini 2.5 Pro)")

uploaded_file = st.file_uploader("📂 Upload your PDF notes", type="pdf")

if uploaded_file:
    st.info("ℹ️ Large files may take longer. Use slider to control speed.")
    word_limit = st.slider("📏 Max words", 500, 4000, 800, step=100)

    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    full_text = extract_text(file_path)
    text = " ".join(full_text.split()[:word_limit])

    mode = st.selectbox("🧠 Choose Task", ["Summary", "Quiz", "Flashcards"])

    if mode == "Summary":
        if st.button("📝 Generate Summary"):
            with st.spinner("Working..."):
                st.write(summarize(text))

    elif mode == "Quiz":
        if st.button("🧠 Generate Interactive Quiz"):
            with st.spinner("Generating..."):
                quiz = generate_structured_quiz(text)
                st.session_state.quiz = quiz
                st.session_state.submitted = False

        if "quiz" in st.session_state:
            quiz = st.session_state.quiz
            st.subheader("📋 Answer the Quiz:")

            answers = {}
            for i, q in enumerate(quiz):
                st.write(f"**Q{i+1}: {q['question']}**")
                selected = st.radio(
                    f"Choose your answer for Q{i+1}:", 
                    options=[f"{k}) {v}" for k, v in q["options"].items()],
                    key=f"q{i}"
                )
                answers[f"q{i}"] = selected[0]
                st.write("---")

            if st.button("✅ Submit Quiz"):
                score = 0
                st.session_state.submitted = True
                for i, q in enumerate(quiz):
                    correct = q["answer"]
                    selected = answers[f"q{i}"]
                    if selected == correct:
                        st.success(f"✔️ Q{i+1}: Correct!")
                        score += 1
                    else:
                        st.error(f"❌ Q{i+1}: Wrong. Correct answer was: {correct}) {q['options'][correct]}")
                st.info(f"🧠 Your score: {score} / {len(quiz)}")

                if st.button("🔖 Bookmark All Correct Q&A"):
                    for i, q in enumerate(quiz):
                        selected = answers[f"q{i}"]
                        if selected == q["answer"]:
                            save_bookmark(q["question"], q["options"][selected])
                    st.success("✅ Bookmarked correct answers.")

    elif mode == "Flashcards":
        if st.button("📇 Create Flashcards"):
            with st.spinner("Generating..."):
                st.write(generate_flashcards(text))

    st.subheader("💬 Ask a Question")
    q = st.text_input("Type your question here")
    if st.button("🔍 Answer"):
        with st.spinner("Gemini is thinking..."):
            a = answer_question(text, q)
            st.success("Done")
            st.write(a)
            if st.button("🔖 Save this"):
                save_bookmark(q, a)

    if st.button("🎤 Ask by Voice"):
        with st.spinner("Listening..."):
            voice_q = voice_input()
        st.write(f"🗣️ You said: {voice_q}")
        with st.spinner("Answering..."):
            a = answer_question(text, voice_q)
            st.success("Answer ready")
            st.write(a)
            if st.button("🔖 Save voice Q&A"):
                save_bookmark(voice_q, a)

    if st.checkbox("📑 View Saved Bookmarks"):
        st.code(load_bookmarks())
