"""
Simple RAG system orchestrator without multi-user functionality.
Coordinates document processing, vector store management, and query processing.
"""
import logging
import os
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from config import config
from document_processor import DocumentProcessor
from vector_store_manager import VectorStoreManager
from query_processor import QueryProcessor

logger = logging.getLogger(__name__)

class RAGSystem:
    """Simple RAG system that orchestrates all components without multi-user support."""
    
    def __init__(self):
        """Initialize the RAG system components."""
        logger.info("Initializing RAG system...")
        
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.query_processor = QueryProcessor(self.vector_store_manager)
        
        self._system_ready = True
        logger.info("RAG system initialized successfully")
    
    def setup_system(self, data_directory: str = None, force_rebuild: bool = False) -> bool:
        """
        Setup the RAG system by processing documents and creating vector store.
        
        Args:
            data_directory: Path to directory containing documents
            force_rebuild: Whether to rebuild vector store even if it exists
            
        Returns:
            True if setup successful, False otherwise
        """
        try:
            data_directory = data_directory or config.DATA_DIRECTORY
            
            logger.info(f"Setting up RAG system with data from: {data_directory}")
            
            # Check if data directory exists and has files
            if not self._validate_data_directory(data_directory):
                return False
            
            # Try to connect to existing vector store first (unless force rebuild)
            if not force_rebuild:
                existing_store = self.vector_store_manager.get_existing_vector_store()
                if existing_store:
                    logger.info("Connected to existing vector store")
                    self._system_ready = True
                    return True
            
            # Process documents
            logger.info("Processing documents...")
            
            # Load documents from directory
            documents = self.document_processor.load_documents_from_directory(data_directory)
            
            if not documents:
                logger.error("No documents were processed successfully")
                return False
            
            # Chunk the documents
            logger.info("Chunking documents...")
            chunked_documents = self.document_processor.chunk_documents(documents)
            
            if not chunked_documents:
                logger.error("No chunks were created from documents")
                return False
            
            # Create vector store
            logger.info("Creating vector store...")
            self.vector_store_manager.create_vector_store(chunked_documents)
            
            self._system_ready = True
            logger.info("RAG system setup completed successfully")
            
            # Print system statistics
            self._print_system_stats()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup RAG system: {e}")
            return False
    
    def add_documents_from_directory(self, directory_path: str) -> bool:
        """
        Add new documents from a directory to the existing vector store.
        
        Args:
            directory_path: Path to directory containing new documents
            
        Returns:
            True if documents added successfully, False otherwise
        """
        try:
            logger.info(f"Adding documents from: {directory_path}")
            
            if not self._validate_data_directory(directory_path):
                return False
            
            # Process new documents
            documents = self.document_processor.load_documents_from_directory(directory_path)
            
            if not documents:
                logger.warning("No new documents to process")
                return True
            
            # Chunk the documents
            chunked_documents = self.document_processor.chunk_documents(documents)
            
            if not chunked_documents:
                logger.warning("No chunks created from new documents")
                return True
            
            # Add to existing vector store
            self.vector_store_manager.add_documents(chunked_documents)
            
            logger.info(f"Successfully added {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def query(self, query: str, max_results: int = 5, score_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Process a query and return results.
        
        Args:
            query: The query string
            max_results: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            Dictionary containing answer and sources
        """
        if not self._system_ready:
            logger.error("System not ready for queries")
            return None
        
        try:
            return self.query_processor.process_query(
                query=query,
                max_results=max_results,
                score_threshold=score_threshold
            )
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return None
    
    def reset_system(self) -> bool:
        """
        Reset the vector store and system state.
        
        Returns:
            True if reset successful, False otherwise
        """
        try:
            success = self.vector_store_manager.reset_index()
            if success:
                self._system_ready = False
                logger.info("System reset successfully")
            return success
        except Exception as e:
            logger.error(f"Failed to reset system: {e}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            Dictionary containing system statistics
        """
        stats = {
            "system_ready": self._system_ready,
            "vector_store_ready": self.vector_store_manager.pinecone_index is not None,
            "total_documents": 0,
            "total_chunks": 0
        }
        
        if self.vector_store_manager.pinecone_index:
            try:
                # Get vector store statistics
                index_stats = self.vector_store_manager.pinecone_index.describe_index_stats()
                stats["total_vectors"] = index_stats.get('total_vector_count', 0)
                stats["index_name"] = config.PINECONE_INDEX_NAME
                stats["embedding_model"] = config.EMBEDDING_MODEL_NAME
            except Exception as e:
                logger.warning(f"Failed to get vector store stats: {e}")
        
        return stats
    
    def _validate_data_directory(self, data_directory: str) -> bool:
        """
        Validate that the data directory exists and contains supported files.
        
        Args:
            data_directory: Path to data directory
            
        Returns:
            True if valid, False otherwise
        """
        data_path = Path(data_directory)
        
        if not data_path.exists():
            logger.error(f"Data directory does not exist: {data_directory}")
            return False
        
        if not data_path.is_dir():
            logger.error(f"Data path is not a directory: {data_directory}")
            return False
        
        # Check for supported files
        supported_files = []
        for file_path in data_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in config.SUPPORTED_EXTENSIONS:
                supported_files.append(file_path)
        
        if not supported_files:
            logger.warning(f"No supported files found in: {data_directory}")
            logger.info(f"Supported extensions: {config.SUPPORTED_EXTENSIONS}")
            return False
        
        logger.info(f"Found {len(supported_files)} supported files")
        return True
    
    def _print_system_stats(self):
        """Print system statistics."""
        try:
            stats = self.get_system_stats()
            logger.info("=== RAG System Statistics ===")
            logger.info(f"System Ready: {stats['system_ready']}")
            logger.info(f"Vector Store Ready: {stats['vector_store_ready']}")
            if 'total_vectors' in stats:
                logger.info(f"Total Vectors: {stats['total_vectors']}")
                logger.info(f"Index Name: {stats['index_name']}")
                logger.info(f"Embedding Model: {stats['embedding_model']}")
            logger.info("============================")
        except Exception as e:
            logger.warning(f"Failed to print system stats: {e}")
    
    @property
    def is_ready(self) -> bool:
        """Check if the system is ready for queries."""
        return self._system_ready
