"""
Vector Store using Universal Sentence Encoder instead of sentence-transformers
"""

import logging
from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

import config
from src.utils import clean_text, Timer

logger = logging.getLogger(__name__)


class USEVectorStore:
    """Vector database using Universal Sentence Encoder for embeddings."""
    
    def __init__(self, collection_name: str = "skill_assessment_kb"):
        """Initialize ChromaDB and Universal Sentence Encoder."""
        
        logger.info(f"Initializing USEVectorStore with collection: {collection_name}")
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.Client(Settings(
            persist_directory=config.CHROMA_PERSIST_DIRECTORY,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Educational content using USE embeddings"}
        )
        
        # Initialize Universal Sentence Encoder
        logger.info("Loading Universal Sentence Encoder from TensorFlow Hub...")
        self.model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        
        try:
            self.embedding_model = hub.load(self.model_url)
            logger.info("✓ Universal Sentence Encoder loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load USE: {e}")
            raise
        
        logger.info(f"✓ USEVectorStore initialized with {self.collection.count()} documents")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using Universal Sentence Encoder."""
        # USE returns embeddings as TensorFlow tensors
        embeddings = self.embedding_model(texts)
        # Convert to numpy array
        return embeddings.numpy()
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of text documents
            metadatas: Optional metadata for each document
            ids: Optional custom IDs for documents
        
        Returns:
            bool: Success status
        """
        try:
            if not documents:
                logger.warning("No documents provided")
                return False
            
            # Clean documents
            cleaned_docs = [clean_text(doc) for doc in documents]
            
            # Generate embeddings using USE
            with Timer("USE embedding generation"):
                embeddings = self.encode(cleaned_docs)
                embeddings_list = embeddings.tolist()
            
            # Generate IDs if not provided
            if ids is None:
                existing_count = self.collection.count()
                ids = [f"doc_{existing_count + i}" for i in range(len(documents))]
            
            # Generate default metadata if not provided
            if metadatas is None:
                metadatas = [{"source": "manual"} for _ in documents]
            
            # Add to collection
            self.collection.add(
                documents=cleaned_docs,
                embeddings=embeddings_list,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"✓ Added {len(documents)} documents to vector store")
            return True
        
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
        
        Returns:
            Dictionary with search results
        """
        try:
            if not query:
                logger.warning("Empty query provided")
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
            
            # Generate query embedding
            with Timer(f"Search: '{query[:50]}...'"):
                query_embedding = self.encode([query])[0].tolist()
                
                # Perform search
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(n_results, self.collection.count()),
                    where=filter_metadata
                )
            
            logger.info(f"✓ Found {len(results['documents'][0])} results")
            return results
        
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def get_all_documents(self) -> Dict[str, Any]:
        """Retrieve all documents from the collection."""
        try:
            results = self.collection.get()
            logger.info(f"Retrieved {len(results['documents'])} documents")
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return {"documents": [], "metadatas": [], "ids": []}
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(name=self.collection.name)
            logger.info(f"✓ Deleted collection: {self.collection.name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "total_documents": self.collection.count(),
            "collection_name": self.collection.name,
            "embedding_model": "Universal Sentence Encoder v4",
            "embedding_dimension": 512  # USE produces 512-dim embeddings
        }


def initialize_sample_data(vector_store: USEVectorStore):
    """Initialize vector store with sample educational content."""
    
    sample_documents = [
        # Python Programming
        "Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        
        "Object-oriented programming (OOP) in Python involves creating classes and objects. Key concepts include encapsulation, inheritance, and polymorphism. Classes are blueprints for objects.",
        
        "Python data structures include lists, tuples, dictionaries, and sets. Lists are mutable ordered collections, tuples are immutable, dictionaries store key-value pairs, and sets contain unique elements.",
        
        # Machine Learning
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed. It includes supervised learning, unsupervised learning, and reinforcement learning.",
        
        "Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes (neurons) that process information through weighted connections.",
        
        "Supervised learning involves training a model on labeled data. Common algorithms include linear regression, logistic regression, decision trees, and support vector machines.",
        
        # Data Science
        "Data science combines statistics, mathematics, programming, and domain expertise to extract insights from data. Key skills include data cleaning, exploratory analysis, and visualization.",
        
        "Pandas is a Python library for data manipulation and analysis. It provides DataFrame structures for handling tabular data and offers powerful tools for data cleaning and transformation.",
        
        # Web Development
        "Web development involves creating websites and web applications. Frontend development focuses on user interface using HTML, CSS, and JavaScript, while backend handles server-side logic.",
        
        "REST APIs (Representational State Transfer) are architectural styles for building web services. They use HTTP methods (GET, POST, PUT, DELETE) to perform CRUD operations.",
        
        # Algorithms
        "Algorithm complexity is measured using Big O notation. Common complexities include O(1) constant, O(log n) logarithmic, O(n) linear, O(n log n) linearithmic, and O(n²) quadratic.",
        
        "Sorting algorithms arrange data in a specific order. Common algorithms include bubble sort, merge sort, quick sort, and heap sort, each with different time and space complexities.",
    ]
    
    sample_metadata = [
        {"topic": "python", "difficulty": "beginner", "category": "programming"},
        {"topic": "python", "difficulty": "intermediate", "category": "programming"},
        {"topic": "python", "difficulty": "beginner", "category": "programming"},
        {"topic": "machine_learning", "difficulty": "beginner", "category": "ai"},
        {"topic": "machine_learning", "difficulty": "intermediate", "category": "ai"},
        {"topic": "machine_learning", "difficulty": "beginner", "category": "ai"},
        {"topic": "data_science", "difficulty": "beginner", "category": "data"},
        {"topic": "data_science", "difficulty": "intermediate", "category": "data"},
        {"topic": "web_development", "difficulty": "beginner", "category": "programming"},
        {"topic": "web_development", "difficulty": "intermediate", "category": "programming"},
        {"topic": "algorithms", "difficulty": "intermediate", "category": "computer_science"},
        {"topic": "algorithms", "difficulty": "intermediate", "category": "computer_science"},
    ]
    
    logger.info("Adding sample educational content...")
    vector_store.add_documents(
        documents=sample_documents,
        metadatas=sample_metadata
    )
    
    logger.info("✓ Sample data initialized")


# Example usage and testing
if __name__ == "__main__":
    print("Testing USE Vector Store...\n")
    
    # Initialize vector store
    vs = USEVectorStore(collection_name="test_use_collection")
    
    # Add sample data if empty
    if vs.collection.count() == 0:
        print("Adding sample data...")
        initialize_sample_data(vs)
    
    # Show stats
    stats = vs.get_stats()
    print(f"\nVector Store Stats:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Embedding model: {stats['embedding_model']}")
    print(f"  Embedding dimension: {stats['embedding_dimension']}")
    
    # Test search
    print("\nTesting search...")
    test_queries = [
        "What is Python?",
        "Explain machine learning",
        "How do I sort data?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = vs.search(query, n_results=3)
        
        for i, (doc, meta) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0]
        ), 1):
            print(f"  {i}. [{meta.get('topic', 'N/A')}] {doc[:100]}...")
    
    print("\n✓ USE Vector Store module ready!")