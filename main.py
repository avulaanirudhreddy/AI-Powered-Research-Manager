import streamlit as st
from google import genai
import fitz
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import pandas as pd
import pytesseract
from pdf2image import convert_from_bytes
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer

# ==============================
# Tesseract Setup
# ==============================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ==============================
# Load Environment Variables
# ==============================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("🚨 GOOGLE_API_KEY not found in .env file")
    st.stop()

client = genai.Client(api_key=api_key)

# ==============================
# Streamlit Config
# ==============================
st.set_page_config(page_title="AI Research Manager", page_icon="📄", layout="wide")
st.title("🤖 AI-Powered Research Manager")
st.markdown("Multi-Source AI Research System (PDF + OCR + Website + YouTube + RAG)")

# ==============================
# Mode Selector
# ==============================
mode = st.sidebar.radio(
    "Select Mode",
    ["Research Mode", "Chat Mode"]
)

# ==============================
# Load Local Embedding Model
# ==============================
@st.cache_resource
def load_local_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

local_embedding_model = load_local_model()

class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return local_embedding_model.encode(texts).tolist()

    def embed_query(self, text):
        return local_embedding_model.encode([text])[0].tolist()

# ==============================
# PDF Extraction
# ==============================
def get_pdf_text(uploaded_file):
    try:
        uploaded_file.seek(0)
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "".join(page.get_text() for page in pdf_document)
        pdf_document.close()
        return text.strip()
    except Exception as e:
        st.error(f"PDF error: {e}")
        return None

# ==============================
# OCR Fallback
# ==============================
def ocr_pdf_with_tesseract(uploaded_file):
    try:
        uploaded_file.seek(0)
        try:
            images = convert_from_bytes(uploaded_file.read())
        except:
            images = convert_from_bytes(
                uploaded_file.read(),
                poppler_path=r"C:\poppler\Library\bin"
            )

        extracted_text = ""
        for image in images:
            text = pytesseract.image_to_string(image)
            extracted_text += text + "\n"

        return extracted_text.strip()
    except Exception as e:
        st.error(f"OCR Error: {e}")
        return None

# ==============================
# Website Extraction
# ==============================
def extract_text_from_website(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        for script in soup(["script", "style", "noscript"]):
            script.extract()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        clean_text = "\n".join(line for line in lines if line)

        return clean_text
    except Exception as e:
        st.error(f"Website extraction error: {e}")
        return None

# ==============================
# YouTube Transcript Extraction
# ==============================
def extract_text_from_youtube(url):
    try:
        video_id = None

        if "watch?v=" in url:
            video_id = url.split("watch?v=")[1].split("&")[0]

        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]

        elif "/shorts/" in url:
            video_id = url.split("/shorts/")[1].split("?")[0]

        if not video_id:
            st.error("Invalid YouTube URL format.")
            return None

        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)

        transcript = None

        # Try manually created English transcript
        try:
            transcript = transcript_list.find_manually_created_transcript(["en"])
        except:
            pass

        # Try generated English transcript
        if not transcript:
            try:
                transcript = transcript_list.find_generated_transcript(["en"])
            except:
                pass

        # Fallback: first available transcript
        if not transcript:
            available = list(transcript_list)
            if available:
                transcript = available[0]
            else:
                st.error("No transcripts available.")
                return None

        transcript_data = transcript.fetch()

        full_text = ""
        for entry in transcript_data:
            full_text += entry.text + " "

        return full_text.strip()

    except Exception as e:
        st.error(f"YouTube Transcript Error: {e}")
        return None

# ==============================
# Gemini Summary
# ==============================
def analyze_full_summary(text_content):
    prompt = f"""
    Provide:
    - A detailed academic summary (minimum 4 paragraphs)
    - Key findings (bullet points)
    - Main contributions (bullet points)

    CONTENT:
    {text_content[:12000]}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

# ==============================
# Gemini RAG Q&A
# ==============================
def analyze_question_with_rag(context, query):
    prompt = f"""
    Answer the question using ONLY the context below.
    Be detailed and extract exact information from context.

    CONTEXT:
    {context}

    QUESTION:
    {query}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


# =====================================================
# ================= RESEARCH MODE =====================
# =====================================================
if mode == "Research Mode":

    source_option = st.radio(
        "Select Source",
        ["Upload PDF", "Website URL", "YouTube Video"]
    )

    uploaded_file = None
    website_url = None
    youtube_url = None

    if source_option == "Upload PDF":
        uploaded_file = st.file_uploader("📁 Upload PDF", type="pdf")

    elif source_option == "Website URL":
        website_url = st.text_input("🌐 Enter Website URL")

    elif source_option == "YouTube Video":
        youtube_url = st.text_input("🎥 Enter YouTube URL")

    if uploaded_file or website_url or youtube_url:

        with st.spinner("Processing content..."):

            if uploaded_file:
                text_content = get_pdf_text(uploaded_file)
                if not text_content or len(text_content.split()) < 50:
                    st.warning("Applying OCR...")
                    text_content = ocr_pdf_with_tesseract(uploaded_file)

            elif website_url:
                text_content = extract_text_from_website(website_url)

            elif youtube_url:
                text_content = extract_text_from_youtube(youtube_url)

            if not text_content:
                st.error("Failed to extract content.")
                st.stop()

            st.success(f"✅ Extracted {len(text_content.split())} words")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=150
            )

            chunks = splitter.split_text(text_content)

            embeddings = LocalEmbeddings()
            vectorstore = FAISS.from_texts(chunks, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

            st.session_state.retriever = retriever
            st.session_state.text_content = text_content

        query = st.text_input("💬 Type 'summarize' or ask a question")

        if query:
            with st.spinner("Analyzing..."):

                if query.lower() in ["summarize", "summary", "give summary"]:
                    analysis = analyze_full_summary(st.session_state.text_content)
                else:
                    relevant_docs = st.session_state.retriever.invoke(query)
                    context = "\n\n".join(doc.page_content for doc in relevant_docs)
                    analysis = analyze_question_with_rag(context, query)

            st.subheader("📋 AI Analysis")
            st.markdown(analysis)

            df = pd.DataFrame([{"Query": query, "Response": analysis}])
            st.download_button(
                "📥 Download CSV",
                df.to_csv(index=False),
                "analysis.csv",
                "text/csv"
            )

    else:
        st.info("👆 Select a source and provide content.")


# =====================================================
# =================== CHAT MODE =======================
# =====================================================
if mode == "Chat Mode":

    st.header("💬 Gemini Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask me anything...")

    if user_input:

        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )

        conversation_text = ""
        for msg in st.session_state.chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n"

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=conversation_text
        )

        bot_reply = response.text

        with st.chat_message("assistant"):
            st.markdown(bot_reply)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": bot_reply}
        )

st.markdown("Powered by Gemini 2.5 Flash + Local Embeddings")