import os
import streamlit as st
from src.indexing.document_processing import DocumentProcessor
from src.indexing.vectore_store import VectorStoreManager
from src.tools_retrieval.retriever import RetrieverManager
from src.workflow import RAGWorkflow


UPLOAD_FOLDER = "uploads/"
PERSIST_DIRECTORY = "qdrant_db/"
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

with st.sidebar:
    st.header("PDF Upload")
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
    st.info("Supported file type: PDF")

    if uploaded_file:
        with st.spinner("Processing PDF..."):
            
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            doc_processor = DocumentProcessor()
            chunks = doc_processor.load_and_split_pdf(file_path)

            vector_store_manager = VectorStoreManager(collection_name=uploaded_file.name, persist_directory=PERSIST_DIRECTORY)
            vector_store = vector_store_manager.index_documents(
                documents=chunks,
                collection_name=uploaded_file.name,
                
            )
            st.session_state.vector_store = vector_store
            st.success("PDF processed and indexed successfully!")
            
            retriever_manager = RetrieverManager(vector_store)
            retriever_tool = retriever_manager.create_retriever()
            st.session_state.retriever = retriever_tool
            st.success("Retriever tool created successfully!")
            
# Chat interface
# Main chat interface
st.divider()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)    

    # Generate response
    with st.chat_message("assistant"):
        if st.session_state.retriever is None:
            final_response = "Please upload a PDF document first."
        else:
            with st.spinner("Thinking..."):
                # Retrieve relevant documents
                rag_workflow = RAGWorkflow(retriever_tool)
                workflow = rag_workflow.compile()
                st.session_state.workflow = workflow
                inputs = {
                    "messages": [
                        ("user", prompt),
                    ]
                }

                # Generate response using workflow
                if st.session_state.workflow is not None:
                    response = st.session_state.workflow.invoke(inputs)
                    final_response = response["messages"][-1].content
                else:
                    final_response = "Please upload a PDF document first."

        st.markdown(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})

# Add clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []    
