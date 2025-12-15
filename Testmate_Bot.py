import os
from pathlib import Path

import streamlit as st
from langdetect import detect

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA


# ----------------------------
# Helpers
# ----------------------------
def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"


def load_documents_from_folder(folder: str):
    """Loads .txt and .pdf files from a folder (recursively) into LangChain Documents."""
    docs = []
    base = Path(folder)

    if not base.exists():
        return docs

    # Load TXT files
    for p in base.rglob("*.txt"):
        try:
            docs.extend(TextLoader(str(p), encoding="utf-8").load())
        except Exception:
            # Fallback encoding
            docs.extend(TextLoader(str(p), encoding="latin-1").load())

    # Load PDF files
    for p in base.rglob("*.pdf"):
        docs.extend(PyPDFLoader(str(p)).load())

    return docs


@st.cache_resource
def build_vectorstore(folder: str):
    """Build and cache a FAISS vectorstore for a given folder path."""
    docs = load_documents_from_folder(folder)

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.from_documents(chunks, embeddings)
    return vs


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Finio AI", page_icon="✨")

# Optional styling (keep your black theme if you want)
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stToolbar"] {display: none !important;}
[data-testid="collapsedControl"] {display: none !important;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

black_background = """
<style>
.stApp { background-color: #000000 !important; }
html, body, [class*="css"] {
    background-color: #000000 !important;
    color: white !important;
}
</style>
"""
st.markdown(black_background, unsafe_allow_html=True)

# ----------------------------
# API Key
# ----------------------------
# Only set this ONCE
openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
if not openai_api_key:
    st.error("Missing OPENAI_API_KEY in Streamlit Secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key


# ----------------------------
# Header / Logo
# ----------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Make sure your logo file exists in the repo root
    if Path("logo.png.png").exists():
        st.image("logo.png.png", width=500)
    st.markdown("<h3 style='text-align:center;'>I am here to help with your Finance queries</h3>", unsafe_allow_html=True)

# ----------------------------
# Build stores (cached)
# ----------------------------
english_store = build_vectorstore("docs/en")
greek_store = build_vectorstore("docs/gr")


if english_store is None and greek_store is None:
    st.warning("No documents found. Add .txt/.pdf files to docs/english and/or docs/greek.")
    st.stop()

# ----------------------------
# Chat UI
# ----------------------------
user_question = st.text_input("Input query:")

if user_question:
    with st.spinner("Analysing..."):
        lang = detect_language(user_question)

        # Choose vectorstore based on language (Greek = 'el')
        if lang == "el" and greek_store is not None:
            vectorstore = greek_store
        else:
            # Default to English store if available, otherwise Greek
            vectorstore = english_store if english_store is not None else greek_store

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False
        )

        response = qa_chain.run(user_question)

    # Display
    if not response or any(p in response.lower() for p in ["i don't know", "not sure", "cannot find", "no information"]):
        st.warning("⚠️ Sorry, I can't find that answer within the company information.")
    else:
        st.success("✅ Answer:")
        st.write(response)
