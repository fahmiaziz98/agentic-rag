from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank


class RetrieverManager:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        
    def create_base_retriever(self, search_type="similarity", k=3):
        """Create basic vector store retriever"""
        return self.vector_store.as_retriever(
            search_type=search_type, 
            search_kwargs={"k": k}
        )
    
    def create_ensemble_retriever(self, texts, vector_weight=0.5, keyword_weight=0.5):
        """Create ensemble retriever combining vector and keyword search"""
        vector_retriever = self.create_base_retriever()
        keyword_retriever = BM25Retriever.from_documents(texts)
        keyword_retriever.k = 3
        
        return EnsembleRetriever(
            retrievers=[vector_retriever, keyword_retriever],
            weights=[vector_weight, keyword_weight]
        )
    
    def create_compression_retriever(self, base_retriever, top_n=5):
        """Create compression retriever with reranking"""
        compressor = FlashrankRerank(top_n=top_n)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )