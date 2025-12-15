import os
from pathlib import Path

import streamlit as st
from langdetect import detect

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


# ----------------------------
# Helpers
# ----------------------------
def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"


def load_documents_from_folder(folder: str):
    """Load .txt and .pdf documents recursively from a folder."""
    docs = []
    base = Path(folder)

    if not base.exists():
        return docs

    # TXT
    for p in base.rglob("*.txt"):
        # Skip temp/lock files e.g. "~$something.txt"
        if p.name.startswith("~$"):
            continue
        try:
            docs.extend(TextLoader(str(p), encoding="utf-8").load())
        except Exception:
            docs.extend(TextLoader(str(p), encoding="latin-1").load())

    # PDF
    for p in base.rglob("*.pdf"):
        docs.extend(PyPDFLoader(str(p)).load())

    return docs


@st.cache_resource
def build_vectorstore(folder: str):
    """Build and cache a FAISS vectorstore for a folder."""
    documents = load_documents_from_folder(folder)
    if not documents:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)


# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="Finio AI", page_icon="✨")

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
# OpenAI API Key
# ----------------------------
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
    # Your repo shows: logo.png.png
    if Path("logo.png.png").exists():
        st.image("logo.png.png", width=500)

st.markdown(
    "<h2 style='text-align:center;'>I am here to help with your Finance queries</h2>",
    unsafe_allow_html=True
)


# ----------------------------
# Build vector stores (your folders)
# ----------------------------

st.write("BASE_DIR:", str(BASE_DIR))
st.write("EN_FOLDER exists:", EN_FOLDER.exists(), "->", str(EN_FOLDER))

if EN_FOLDER.exists():
    st.write("Files in docs/en:", [p.name for p in EN_FOLDER.rglob("*")])

BASE_DIR = Path(__file__).resolve().parent
EN_FOLDER = BASE_DIR / "docs" / "en"


vectorstore = build_vectorstore(str(EN_FOLDER))


if vectorstore is None:
    st.warning(f"No documents found. Add .txt/.pdf files to {EN_FOLDER}.")
    st.stop()


# ----------------------------
# Prompt (forces answers from context only)
# ----------------------------
STRICT_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are Finio AI, a company finance assistant.\n"
        "Answer ONLY using the company documents provided in the CONTEXT.\n"
        "If the answer is present, explain it clearly and step-by-step.\n"
        "If the context does not contain the answer, reply exactly:\n"
        "\"Sorry, I can't find that answer within the company information.\"\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION:\n{question}\n\n"
        "ANSWER:"
    ),
)


# ----------------------------
# Chat UI
# ----------------------------
user_question = st.text_input("Input query:")

if user_question:
    with st.spinner("Analysing..."):
        lang = detect_language(user_question)

        # Choose store based on language
        if lang == "el" and greek_store is not None:
            vectorstore = greek_store
        else:
            vectorstore = english_store if english_store is not None else greek_store

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": STRICT_PROMPT}
        )

        response = qa_chain.run(user_question)

    # Display (simple and correct)
    if response and response.strip():
        st.success("✅ Answer:")
        st.write(response)
    else:
        st.warning("⚠️ Sorry, I can't find that answer within the company information.")
