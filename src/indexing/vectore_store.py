from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore

class VectorStoreManager:
    def __init__(self, embedding_model="intfloat/multilingual-e5-small"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
    def create_vector_store(self, documents):
        """Create a new vector store"""
        vector_store = SKLearnVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings,
        )
        return vector_store
    
    # def index_documents(self, documents):
    #     """Index documents into vector store"""
    #     vector_store = self.create_vector_store()
    #     vector_store.add_documents(documents=documents)
    #     return vector_store 