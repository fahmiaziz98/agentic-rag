from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

class VectorStoreManager:
    def __init__(self, embedding_model="intfloat/multilingual-e5-small", collection_name="my_collection", persist_directory=None):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        self.collection_name = collection_name
        if persist_directory:
            self.client = QdrantClient(path=persist_directory)
        else:
            self.client = QdrantClient(":memory:")
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={"dense": VectorParams(size=3072, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))},
        )
        

    def create_vector_store(self):
        """Create a new vector store"""
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_embedding=self.sparse_embeddings,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
    
    def index_documents(self, documents):
        """Index documents into vector store"""
        vector_store = self.create_vector_store()
        vector_store.add_documents(documents=documents)
        return vector_store