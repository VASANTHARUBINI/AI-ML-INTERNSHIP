import streamlit as st
import fitz  # PyMuPDF
import os
import pyttsx3
from pydub import AudioSegment
from dotenv import load_dotenv
import google.generativeai as genai

# Load Gemini API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-pro")


# ----------------- Helper Functions -----------------

def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            page_text = page.get_text()
            if len(page_text.strip()) > 20:
                text += page_text
    return text.strip()


def gemini_generate(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[ERROR] Gemini failed: {e}"


def generate_title_and_summary(text):
    prompt = f"""
    You're an AI podcast assistant. Read the content below and generate ONLY:
    1. A podcast title
    2. A short episode summary

    Respond in this format:
    Title: <your title>
    Summary: <your summary>

    CONTENT:
    {text}
    """
    response = gemini_generate(prompt)

    # Parse output
    title, summary = "Untitled Podcast", "No summary available."
    lines = response.splitlines()
    for line in lines:
        if "Title:" in line:
            title = line.replace("Title:", "").strip()
        elif "Summary:" in line:
            summary = line.replace("Summary:", "").strip()
    return title, summary


def generate_podcast_script(text):
    prompt = (
        "You're an intelligent podcast narrator. Based ONLY on the content below, create a 3-minute script.\n"
        "If it's a story, retell it naturally. If it's educational or technical, explain the concepts clearly.\n"
        "Avoid reading code or special characters. Do NOT add extra topics.\n\n"
        f"CONTENT:\n{text}"
    )
    body = gemini_generate(prompt)

    # Optional smart intro based on content type
    if any(keyword in text.lower() for keyword in ["api", "machine learning", "gemini", "ai", "model", "neural"]):
        intro = "Welcome to AI Podcast – simplifying the world of technology, one episode at a time.\n\n"
    else:
        intro = "Welcome to AI Podcast – where we bring timeless stories and smart ideas to life.\n\n"

    outro = "\n\nThanks for listening. Stay inspired with AI Podcast!"

    return intro + body.strip() + outro


def text_to_speech(text, output_path="output/podcast.mp3"):
    if not os.path.exists("output"):
        os.makedirs("output")

    engine = pyttsx3.init()

    # Set female voice
    voices = engine.getProperty('voices')
    for voice in voices:
        if "female" in voice.name.lower() or "zira" in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break

    engine.save_to_file(text, "output/temp.wav")
    engine.runAndWait()

    sound = AudioSegment.from_wav("output/temp.wav")
    sound.export(output_path, format="mp3")
    return output_path

# ----------------- Streamlit App -----------------

st.set_page_config(page_title="🎙️ AI Podcast Generator")
st.title("🎙️ AI-Powered Podcast Generator")
st.markdown("Upload any PDF – story, tech, blog, notes – and get a podcast with intelligent narration.")

uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("📖 Reading your PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    if not text or len(text.split()) < 30:
        st.error("❌ This file doesn't have enough readable content. Please upload a different PDF.")
        st.stop()

    with st.spinner("🧠 Generating title and summary..."):
        title, summary = generate_title_and_summary(text)

    st.subheader("🎧 Podcast Title")
    st.success(title)

    st.subheader("📝 Episode Summary")
    st.info(summary)

    with st.spinner("🎙️ Writing podcast script..."):
        script = generate_podcast_script(text)

    with st.spinner("🔊 Converting to female voice..."):
        audio_path = text_to_speech(script)

    st.subheader("🎵 Listen to Your Podcast")
    st.audio(audio_path)

    with open(audio_path, "rb") as file:
        st.download_button("📥 Download Podcast", file, file_name="ai_podcast.mp3", mime="audio/mpeg")
