import os
import streamlit as st


UPLOAD_FOLDER = "uploads/"
PERSIST_DIRECTORY = "chroma_db/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "workflow" not in st.session_state:
    st.session_state.workflow = None

st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide",
    page_icon="📘",
)
st.title("Agentic RAG Chatbot")
