import os
import streamlit as st
import pandas as pd
from PIL import Image
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
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
st.set_page_config(page_title="heyHR AI", page_icon="✨")

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
    english_folder = "docs/en"
    greek_folder = "docs/gr"

    # Load English .txt and .pdf
    for file in os.listdir(english_folder):
        full_path = os.path.join(english_folder, file)
        try:
            if file.endswith(".txt"):
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if content.strip():
                    english_docs.append(Document(page_content=content, metadata={"source": file}))
                    print(f"✅ Loaded .txt file: {file}")
                    print("✅ English document added:", file)

                    print("------ Preview ------")
                    print(content[:200])
                else:
                    print(f"⚠️ File {file} is empty.")
            elif file.endswith(".pdf"):
                loader = PyPDFLoader(full_path)
                docs = loader.load()
                english_docs.extend(docs)
                print(f"✅ Loaded PDF: {file}")
                if docs:
                 print(docs[0].page_content[:200])
            else:
                print("⚠️ PDF is empty or failed to load.")
        except Exception as e:
            print(f"❌ Error loading {full_path}: {e}")

    # Load Greek .txt and .pdf
    for file in os.listdir(greek_folder):
        full_path = os.path.join(greek_folder, file)
        try:
            if file.endswith(".txt"):
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if content.strip():
                    greek_docs.append(Document(page_content=content, metadata={"source": file}))
            elif file.endswith(".pdf"):
                loader = PyPDFLoader(full_path)
                docs = loader.load()
                greek_docs.extend(docs)
        except Exception as e:
            print(f"⚠️ Skipped GR file: {file} ❌ {e}")

    return english_docs, greek_docs

# Load and embed
english_docs, greek_docs = load_language_documents()
print(f"📄 Total English documents loaded: {len(english_docs)}")


all_documents = english_docs + greek_docs
print(f"✅ English documents loaded: {len(english_docs)}")
print(f"✅ Greek documents loaded: {len(greek_docs)}")


splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
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

if english_docs:
    english_store = FAISS.from_documents(english_docs, embedding)
    english_store.save_local(english_store_path)
    print(f"✅ English vector store saved to: {english_store_path}")
else:
    print("❌ No English documents found. Skipping English vector store creation.")

if greek_docs:
    greek_store = FAISS.from_documents(greek_docs, embedding)
    greek_store.save_local(greek_store_path)
    print(f"✅ Greek vector store saved to: {greek_store_path}")
else:
    print("❌ No Greek documents found. Skipping Greek vector store creation.")

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
    st.image(logo, width=500)
st.markdown("<h2 style='text-align: center;'></h3>", unsafe_allow_html=True)
st.markdown("I am here to help with your queries")

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
from pathlib import Path

if Path("vector_store_en/index.faiss").exists():
    english_store = FAISS.load_local("vector_store_en", embedding, allow_dangerous_deserialization=True)
else:
    st.error("❌ English vector store not found. Please run the app with at least one English document to generate it.")
    st.stop()
greek_store = FAISS.load_local("vector_store_gr", embedding, allow_dangerous_deserialization=True)

user_question = st.text_input("Input Query:")

if user_question:
    with st.spinner("Analysing..."):
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
        st.warning("⚠️ Sorry, I can't find that answer within the Symphony.is company information.")
    else:
        st.success("✅ Answer:")
        st.write(response)
# Convert response to CSV if needed
try:
    # Example: Assume the response is tabular-like text or key-value pairs
    if isinstance(response, str) and "," in response:
        # Split into lines and columns
        data = [row.split(",") for row in response.strip().split("\n")]
        df = pd.DataFrame(data[1:], columns=data[0])  # assumes first row is header
    elif isinstance(response, list):
        df = pd.DataFrame(response)
    else:
        # fallback: single column CSV
        df = pd.DataFrame({"Result": [response]})
    
    csv_data = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download as CSV",
        data=csv_data,
        file_name="query_results.csv",
        mime="text/csv"
    )
except Exception as e:
    st.error(f"⚠ Could not generate CSV: {e}")

