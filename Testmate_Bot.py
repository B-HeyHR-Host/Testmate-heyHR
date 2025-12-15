import os
from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image
from langdetect import detect
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"
    import re
import io

# ----------------------
# Page Setup
# ----------------------
st.set_page_config(page_title="Humanio.AI", page_icon="✨")

# Load OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key


# ----------------------
# Styling
# ----------------------
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
.stApp {
background-color: #000000 !important;
}
html, body, [class*="css"] {
background-color: #000000 !important;
color: white !important;
}
</style>
"""
st.markdown(black_background, unsafe_allow_html=True)


# ----------------------
# Header / Logo
# ----------------------
logo_path = Path("logo.png.png")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
 st.image(logo,width=500)

st.markdown("<h2 style='text-align: center;'></h3>", unsafe_allow_html=True)
st.markdown("I am here to help with your HR queries")

# ----------------------
# 2. Load Environment Variables
# ----------------------
openai_api_key = st.secrets["OPENAI_API_KEY"]

# ----------------------
# Chat UI
# ----------------------
user_question = st.text_input("Input query:")
response = ""   # pre-declare so it's always defined

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

        response = qa_chain.run(user_question)  # Get answer

        # Show the response only once, in the green box
        if not response.strip() or any(p in response.lower() for p in ["i don't know", "not sure", "cannot find", "no information"]):
            st.warning("⚠ Sorry, I can't find that answer within the company information.")
        else:
            st.success("✅ Answer:")
            st.write(response)