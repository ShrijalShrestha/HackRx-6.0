"""
Simple vector store management module for embedding and vector database operations.
Handles Pinecone integration without multi-user functionality - based on working notebook.
"""
import logging
import time
from typing import List, Optional, Dict, Any, Tuple
import uuid

from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from config import config

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages vector store operations including embedding and Pinecone integration."""
    
    def __init__(self):
        self.embedding_model = None
        self.pinecone_client = None
        self.pinecone_index = None
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize embedding model and Pinecone client."""
        try:
            # Initialize embedding model
            logger.info(f"Initializing SentenceTransformer embedding model: {config.EMBEDDING_MODEL_NAME}")
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
            
            # Initialize Pinecone client
            logger.info(f"Initializing Pinecone client for region: {config.PINECONE_REGION}")
            self.pinecone_client = Pinecone(api_key=config.PINECONE_API_KEY)
            
            # Connect to or create the index
            self._ensure_index_exists()
            self.pinecone_index = self.pinecone_client.Index(config.PINECONE_INDEX_NAME)
            
            logger.info("Vector store manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store manager: {e}")
            raise
    
    def _ensure_index_exists(self):
        """Ensure the Pinecone index exists, create if it doesn't."""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pinecone_client.list_indexes()]
            
            if config.PINECONE_INDEX_NAME not in existing_indexes:
                logger.info(f"Creating Pinecone index: {config.PINECONE_INDEX_NAME}")
                self.pinecone_client.create_index(
                    name=config.PINECONE_INDEX_NAME,
                    dimension=config.EMBEDDING_DIMENSION,
                    metric=config.PINECONE_METRIC,
                    spec=ServerlessSpec(
                        cloud=config.PINECONE_CLOUD,
                        region=config.PINECONE_REGION
                    )
                )
                # Wait for index to be ready
                time.sleep(10)
                logger.info("Index created successfully")
            else:
                logger.info(f"Index '{config.PINECONE_INDEX_NAME}' already exists")
                
        except Exception as e:
            logger.error(f"Failed to ensure index exists: {e}")
            raise
    
    def get_existing_vector_store(self) -> bool:
        """
        Check if a vector store already exists and has vectors.
        
        Returns:
            True if exists and has vectors, False otherwise
        """
        try:
            # Check if index exists and has vectors
            index_stats = self.pinecone_index.describe_index_stats()
            total_vectors = index_stats.get('total_vector_count', 0)
            
            if total_vectors > 0:
                logger.info(f"Found existing vector store with {total_vectors} vectors")
                return True
            else:
                logger.info("Vector store exists but is empty")
                return False
                
        except Exception as e:
            logger.error(f"Error checking existing vector store: {e}")
            return False
    
    def create_vector_store(self, documents: List[Document]) -> bool:
        """
        Create vectors from documents and store them in Pinecone.
        
        Args:
            documents: List of Document objects to embed and store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Creating vector store with {len(documents)} documents")
            
            # Prepare data for upsert
            vectors_to_upsert = []
            
            for i, doc in enumerate(documents):
                # Generate embedding
                embedding = self.embedding_model.encode(doc.page_content).tolist()
                
                # Create unique ID
                vector_id = str(uuid.uuid4())
                
                # Prepare metadata
                metadata = {
                    'text': doc.page_content,
                    'source': doc.metadata.get('source', 'unknown'),
                    'chunk_id': doc.metadata.get('chunk_id', f'chunk_{i}'),
                    'chunk_index': doc.metadata.get('chunk_index', i)
                }
                
                # Add other metadata if available
                for key, value in doc.metadata.items():
                    if key not in ['text', 'source', 'chunk_id', 'chunk_index']:
                        if isinstance(value, (str, int, float, bool)):
                            metadata[key] = value
                
                vectors_to_upsert.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': metadata
                })
            
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i+batch_size]
                self.pinecone_index.upsert(vectors=batch)
                logger.info(f"Upserted batch {i//batch_size + 1}/{(len(vectors_to_upsert)-1)//batch_size + 1}")
            
            logger.info("Vector store created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            return False
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add new documents to the existing vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Adding {len(documents)} documents to vector store")
            return self.create_vector_store(documents)
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 5, score_threshold: float = 0.1) -> List[Document]:
        """
        Perform similarity search in the vector store.
        
        Args:
            query: Query string
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of relevant Document objects
        """
        try:
            logger.info(f"Performing similarity search for: {query[:50]}...")

            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()

            # Retry with backoff to allow for eventual consistency after upserts
            attempts = 5
            backoff = 1.0
            last_documents: List[tuple] = []

            for attempt in range(1, attempts + 1):
                # Query Pinecone
                results = self.pinecone_index.query(
                    vector=query_embedding,
                    top_k=k,
                    include_metadata=True,
                )

                # Handle response for both dict-like and attr-like access
                matches = []
                try:
                    if isinstance(results, dict):
                        matches = results.get('matches', [])
                    else:
                        matches = getattr(results, 'matches', []) or []
                except Exception:
                    matches = []

                # Convert results to Document objects
                documents: List[tuple] = []
                for m in matches:
                    try:
                        score = m.get('score') if isinstance(m, dict) else getattr(m, 'score', 0.0)
                        if score is None:
                            score = 0.0
                        if float(score) < float(score_threshold):
                            continue

                        metadata = m.get('metadata') if isinstance(m, dict) else getattr(m, 'metadata', {})
                        metadata = metadata or {}
                        # Extract text from metadata (stored during upsert)
                        text = metadata.pop('text', '')

                        doc = Document(
                            page_content=text,
                            metadata=metadata,
                        )
                        documents.append((doc, float(score)))
                    except Exception:
                        continue

                if documents:
                    logger.info(f"Found {len(documents)} results above threshold {score_threshold}")
                    return documents

                last_documents = documents
                if attempt < attempts:
                    time.sleep(backoff)
                    backoff *= 1.5  # exponential backoff

            logger.info(f"Found 0 results above threshold {score_threshold} after {attempts} attempts")
            return last_documents

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            if not self.pinecone_index:
                return {"error": "Pinecone index not initialized"}
            
            # Get index statistics
            index_stats = self.pinecone_index.describe_index_stats()
            
            return {
                "index_name": config.PINECONE_INDEX_NAME,
                "total_vectors": index_stats.get('total_vector_count', 0),
                "dimension": index_stats.get('dimension', 0),
                "embedding_model": config.EMBEDDING_MODEL_NAME,
                "index_fullness": index_stats.get('index_fullness', 0.0),
                "namespaces": index_stats.get('namespaces', {})
            }
            
        except Exception as e:
            logger.error(f"Failed to get vector store stats: {e}")
            return {"error": str(e)}
    
    def check_vector_store_health(self) -> bool:
        """Check if the vector store is healthy and accessible."""
        try:
            if not self.pinecone_index:
                return False
            
            # Try to get index stats
            stats = self.pinecone_index.describe_index_stats()
            return True
            
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return False
    
    def get_total_vector_count(self) -> int:
        """Get the total number of vectors in the index."""
        try:
            if not self.pinecone_index:
                return 0
            
            index_stats = self.pinecone_index.describe_index_stats()
            return index_stats.get('total_vector_count', 0)
            
        except Exception as e:
            logger.error(f"Failed to get vector count: {e}")
            return 0
    
    def reset_index(self) -> bool:
        """Reset (clear) the entire vector index."""
        try:
            logger.warning("Resetting vector index - all data will be deleted!")
            
            # Delete all vectors in the index
            self.pinecone_index.delete(delete_all=True)
            
            logger.info("Vector index reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset vector index: {e}")
            return False
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed statistics including performance metrics."""
        try:
            start_time = time.time()
            
            # Basic stats
            stats = self.get_stats()
            
            # Add performance metrics
            stats.update({
                "health_check_time": time.time() - start_time,
                "embeddings_initialized": self.embedding_model is not None,
                "pinecone_connected": self.pinecone_index is not None
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get detailed stats: {e}")
            return {"error": str(e)}
