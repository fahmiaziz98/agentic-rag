from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )

    def load_and_split_pdf(self, file_path: str):
        """Load PDF and split into chunks"""
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        chunks = self.text_splitter.split_documents(docs)
        return chunks