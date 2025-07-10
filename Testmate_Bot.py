import os
import streamlit as st
from PIL import Image
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
# ----------------------
# 1. Page Setup
# ----------------------
st.set_page_config(page_title="Trident Group - heyHR", page_icon="✨")

# Load OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key

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
st.markdown("<h2 style='text-align: center;'>Trident Group</h3>", unsafe_allow_html=True)
st.markdown("Ask me anything about Your Companies policies, processes, or the employee handbook.")

# ----------------------
# 2. Load Environment Variables
# ----------------------
openai_api_key = st.secrets["OPENAI_API_KEY"]

# ----------------------
# 3. Load & Split Documents
# ----------------------
@st.cache_resource
def load_docs():
    doc_files = [
        "Testmate-heyHR.txt",
        "Testmate-heyHR1.txt",
        "Testmate-heyHR2.txt"
    ]

    all_docs = []
    for file in doc_files:
        if os.path.exists(file):
            loader = TextLoader(file)
            docs = loader.load()
            all_docs.extend(docs)
        else:
            st.warning(f"⚠️ Missing document: {file}")

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings(openai_api_key =os.getenv ("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    return vectorstore

# ----------------------
# 4. Create QA Chain
# ----------------------
vectorstore = load_docs()
retriever = vectorstore.as_retriever()
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# ----------------------
# 5. Chat Interface
# ----------------------
user_question = st.text_input("💬 Ask Trident Group a question")

if user_question:
    with st.spinner("Thinking..."):
        response = qa_chain.run(user_question)

    # Check for empty or unclear response
    if not response.strip() or any(phrase in response.lower() for phrase in [
        "i don't know", "i'm not sure", "cannot find", "no information", "sorry"
    ]):
        st.warning("⚠️ Sorry, I can’t find that answer within the Trident Group information.")
    else:
        st.success("✅ Answer:")
        st.write(response)

