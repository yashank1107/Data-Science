from typing import List, Dict, Any, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json
import logging
from config import settings

logger = logging.getLogger(__name__)

class BaseRAG:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_store_path = settings.VECTOR_STORE_PATH
        self.index = None
        self.documents = []
        
    def load_documents(self, document_names: List[str]):
        """Load selected documents into memory"""
        self.documents = []
        # Implementation for loading documents would go here
        logger.info(f"Loading documents: {document_names}")
        
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Basic semantic search"""
        if not self.documents:
            return []
            
        try:
            # Create a simple FAISS index for demonstration
            if self.index is None:
                # For demo purposes, create dummy embeddings
                doc_embeddings = np.random.random((len(self.documents), 384)).astype('float32')
                self.index = faiss.IndexFlatIP(384)
                self.index.add(doc_embeddings)
            
            query_embedding = self.encoder.encode([query])
            distances, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    results.append({
                        "content": self.documents[idx],
                        "score": float(distances[0][i]),
                        "type": "semantic",
                        "source": "document"
                    })
            return results
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []

class KnowledgeGraphRAG(BaseRAG):
    def __init__(self):
        super().__init__()
        self.knowledge_graph = {}
    
    def build_knowledge_graph(self):
        """Build knowledge graph from documents"""
        # Implementation for knowledge graph construction
        logger.info("Building knowledge graph from documents")
        pass
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Knowledge graph enhanced search"""
        semantic_results = super().search(query, k)
        # Add knowledge graph reasoning here
        logger.info("Using knowledge graph enhanced search")
        return semantic_results

class HybridRAG(BaseRAG):
    def __init__(self, search_service):
        super().__init__()
        self.search_service = search_service
    
    async def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Hybrid search combining semantic and internet search"""
        logger.info(f"Performing hybrid search for: {query}")
        
        # Get semantic results from documents
        semantic_results = super().search(query, k)
        
        # Get internet search results
        internet_results = await self.search_service.search(query)
        
        # Combine and rank results
        combined_results = semantic_results + internet_results
        
        # Simple ranking: semantic results first, then internet results
        ranked_results = sorted(
            semantic_results, 
            key=lambda x: x.get('score', 0), 
            reverse=True
        ) + internet_results
        
        logger.info(f"Hybrid search returned {len(ranked_results)} total results "
                   f"({len(semantic_results)} semantic, {len(internet_results)} internet)")
        
        return ranked_results[:k]

class RAGFactory:
    @staticmethod
    def create_rag(variant: str, search_service=None) -> BaseRAG:
        logger.info(f"Creating RAG variant: {variant}")
        if variant == "knowledge_graph":
            return KnowledgeGraphRAG()
        elif variant == "hybrid":
            if search_service is None:
                logger.warning("No search service provided for hybrid RAG, falling back to basic")
                return BaseRAG()
            return HybridRAG(search_service)
        else:
            return BaseRAG()