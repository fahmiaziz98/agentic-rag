from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class VectorStoreManager:
    def __init__(self, embedding_model="intfloat/multilingual-e5-small"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
    def create_vector_store(self, collection_name="my_collection", persist_directory=None):
        """Create a new vector store"""
        store_params = {
            "collection_name": collection_name,
            "embedding_function": self.embeddings,
        }
        if persist_directory:
            store_params["persist_directory"] = persist_directory
            
        return Chroma(**store_params)
    
    def index_documents(self, documents, collection_name="my_collection", persist_directory=None):
        """Index documents into vector store"""
        vector_store = self.create_vector_store(collection_name, persist_directory)
        vector_store.add_documents(documents=documents)
        return vector_store