# ========== main.py ==========
import streamlit as st
import fitz  # PyMuPDF
import os
import google.generativeai as genai
import speech_recognition as sr
from dotenv import load_dotenv
import re

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load Gemini 2.5 Pro
model = genai.GenerativeModel("gemini-2.5-pro")

# ========== UTILS ==========
@st.cache_data
def extract_text(file_path):
    doc = fitz.open(file_path)
    return " ".join([page.get_text() for page in doc])

def chunk_text(text, max_len=1500):
    words = text.split()
    return [" ".join(words[i:i+max_len]) for i in range(0, len(words), max_len)]

def summarize(text):
    chunks = chunk_text(text)
    all = []
    for chunk in chunks:
        prompt = f"""
Act as a professional teacher. Summarize the following class notes in simple language with bullet points:

{chunk}
"""
        response = model.generate_content(prompt)
        all.append(response.text)
    return "\n".join(all)

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

def voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak your question.")
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except:
        return "❌ Could not recognize your voice."

# ========== UI ==========
st.set_page_config(page_title="📚 AI Study Assistant", layout="centered")
st.title("📚 AI Study Assistant")

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
                option_labels = [f"{opt}) {text}" for opt, text in q['options'].items()]
                selected_label = st.radio(
                    f"Choose your answer for Q{i+1}:",
                    options=option_labels,
                    key=f"q{i}"
                )
                selected_key = selected_label[0]
                answers[f"q{i}"] = selected_key
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

    elif mode == "Flashcards":
        if st.button("📇 Create Flashcards"):
            with st.spinner("Generating..."):
                st.write(generate_flashcards(text))

    st.subheader("💬 Ask a Question")
    q = st.text_input("Type your question here")

    if st.button("🔍 Answer"):
        with st.spinner("Gemini is thinking..."):
            a = answer_question(text, q)
            st.session_state.last_q = q
            st.session_state.last_a = a
            st.success("Done")
            st.write(a)

    if st.button("🎤 Ask by Voice"):
        with st.spinner("Listening..."):
            voice_q = voice_input()
        st.write(f"🗣️ You said: {voice_q}")
        with st.spinner("Answering..."):
            a = answer_question(text, voice_q)
            st.session_state.voice_q = voice_q
            st.session_state.voice_a = a
            st.success("Answer ready")
            st.write(a)