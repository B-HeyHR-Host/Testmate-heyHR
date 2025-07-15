import os
import streamlit as st
from PIL import Image
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langdetect import detect

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# ----------------------
# 1. Page Setup
# ----------------------
st.set_page_config(page_title="Pharmathen - heyHR", page_icon="✨")

# Load OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key
# ----------------------------------
# 2. Load and Embed All Documents
# ----------------------------------
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_language_documents():
    english_docs = []
    greek_docs = []

    # Folder paths
    english_folder = "docs/en"
    greek_folder = "docs/gr"

    # Load English .txt
    for file in os.listdir(english_folder):
        if file.endswith(".txt"):
            try:
                loader = TextLoader(os.path.join(english_folder, file), encoding="utf-8")
                english_docs.extend(loader.load())
            except Exception as e:
                print(f"⚠️ Skipped EN file: {file} – {e}")

    # Load Greek .txt
    for file in os.listdir(greek_folder):
        if file.endswith(".txt"):
            try:
                loader = TextLoader(os.path.join(greek_folder, file), encoding="utf-8")
                greek_docs.extend(loader.load())
            except Exception as e:
                print(f"⚠️ Skipped GR file: {file} – {e}")

    return english_docs, greek_docs

# Load and embed
english_docs, greek_docs = load_language_documents()
all_documents = english_docs + greek_docs
print(f"✅ English documents loaded: {len(english_docs)}")
print(f"✅ Greek documents loaded: {len(greek_docs)}")


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
split_docs = splitter.split_documents(all_documents)

# Load or create vector store
embedding = OpenAIEmbeddings()

from pathlib import Path
import shutil

# Set vectorstore paths
english_store_path = "vector_store_en"
greek_store_path = "vector_store_gr"

# Delete existing vector stores to ensure fresh indexing
if Path(english_store_path).exists():
    shutil.rmtree(english_store_path)

if Path(greek_store_path).exists():
    shutil.rmtree(greek_store_path)

# Rebuild and save new vector stores
embedding = OpenAIEmbeddings()

english_store = FAISS.from_documents(english_docs, embedding)
english_store.save_local(english_store_path)

greek_store = FAISS.from_documents(greek_docs, embedding)
greek_store.save_local(greek_store_path)
print("✅ English vector store saved to:", english_store_path)
print("✅ Greek vector store saved to:", greek_store_path)



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

# Display logo
logo = Image.open("logo.png.png")
# Create 3 columns: left (1), center (2), right (1)
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image(logo, width=300)
st.markdown("<h2 style='text-align: center;'>heyHR Lex</h3>", unsafe_allow_html=True)
st.markdown("Ask us anything related to the company policies and procedures")

# ----------------------
# 2. Load Environment Variables
# ----------------------
openai_api_key = st.secrets["OPENAI_API_KEY"]

# ----------------------
# 5. Chat Interface
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
# Load vector stores before any questions are asked
embedding = OpenAIEmbeddings()
english_store = FAISS.load_local("vector_store_en", embedding, allow_dangerous_deserialization=True)
greek_store = FAISS.load_local("vector_store_gr", embedding, allow_dangerous_deserialization=True)

user_question = st.text_input("Ask me anything:")

if user_question:
    with st.spinner("Thinking..."):
        lang = detect_language(user_question)

        retriever = VectorStoreRetriever(
            vectorstore=greek_store if lang == "el" else english_store
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(),
            retriever=retriever,
            return_source_documents=False
        )

        response = qa_chain.run(user_question)  # ✅ Make sure this is inside the `if user_question:` block

    # ✅ Response exists here now
    if not response.strip() or any(phrase in response.lower() for phrase in [
        "i don't know", "not sure", "cannot find", "no information"
    ]):
        st.warning("⚠️ Sorry, I can't find that answer within the Pharmathen company information.")
    else:
        st.success("✅ Answer:")
        st.write(response)


