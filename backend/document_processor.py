"""
Document processing module for parsing, chunking, and tokenizing documents.
Handles various file formats with robust parsing and fallback mechanisms.
"""
import logging
import time
from pathlib import Path
from typing import List, Optional
import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredEmailLoader
)

from config import config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading, parsing, and chunking operations."""
    
    def __init__(self):
        self.llama_parser = None
        self.text_splitter = None
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize document processing components."""
        try:
            # Initialize LlamaParse with timeout and retry settings
            if config.LLAMA_CLOUD_API_KEY:
                try:
                    from llama_parse import LlamaParse
                    
                    self.llama_parser = LlamaParse(
                        api_key=config.LLAMA_CLOUD_API_KEY,
                        result_type="markdown",
                        verbose=False,
                        language="en",
                        # Add timeout and retry settings
                        max_timeout=30,  # 30 second timeout
                        num_workers=1,   # Single worker to avoid conflicts
                        check_interval=2, # Check every 2 seconds
                    )
                    logger.info("LlamaParse initialized successfully with timeout settings")
                except Exception as e:
                    logger.warning(f"Failed to initialize LlamaParse: {e}. Will use fallback parsers only.")
                    self.llama_parser = None
            else:
                logger.info("No LlamaParse API key provided, using community loaders only")
            
            # Initialize text splitter for chunking
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            logger.info("Document processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize document processor: {e}")
            raise
    
    def load_documents_from_directory(self, directory_path: str = None) -> List[Document]:
        """
        Load and parse all supported documents from a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of parsed Document objects
        """
        directory_path = Path(directory_path or config.DATA_DIRECTORY)
        
        if not directory_path.exists():
            logger.error(f"Directory not found: {directory_path}")
            return []
        
        # Find supported files
        supported_files = [
            f for f in directory_path.iterdir() 
            if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS
        ]
        
        if not supported_files:
            logger.warning(f"No supported files found in {directory_path}")
            logger.info(f"Supported extensions: {config.SUPPORTED_EXTENSIONS}")
            return []
        
        logger.info(f"Found {len(supported_files)} supported files")
        
        all_documents = []
        
        for file_path in supported_files:
            try:
                logger.info(f"Processing file: {file_path.name}")
                documents = self._process_single_file(file_path)
                
                if documents:
                    all_documents.extend(documents)
                    logger.info(f"✅ Successfully processed {file_path.name}: {len(documents)} documents")
                else:
                    logger.warning(f"⚠️ No content extracted from {file_path.name}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to process {file_path.name}: {e}")
                continue
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def _process_single_file(self, file_path: Path) -> List[Document]:
        """
        Process a single file with timeout and fallback handling.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of Document objects
        """
        file_extension = file_path.suffix.lower()
        documents = []
        
        # For PDFs and complex documents, try LlamaParse with timeout
        if self.llama_parser and file_extension in [".pdf", ".docx", ".eml"]:
            try:
                logger.info(f"Attempting LlamaParse for {file_path.name}")
                documents = self._parse_with_llama_timeout(file_path)
                if documents:
                    logger.info(f"✅ LlamaParse successful for {file_path.name}")
                    return documents
                else:
                    logger.warning(f"⚠️ LlamaParse returned no content for {file_path.name}")
            except Exception as e:
                logger.warning(f"❌ LlamaParse failed for {file_path.name}: {e}")
        
        # Fallback to community loaders
        try:
            logger.info(f"Using community loaders for {file_path.name}")
            documents = self._parse_with_community_loaders(file_path)
            if documents:
                logger.info(f"✅ Community loader successful for {file_path.name}")
            else:
                logger.warning(f"⚠️ Community loader returned no content for {file_path.name}")
        except Exception as e:
            logger.error(f"❌ All parsing methods failed for {file_path.name}: {e}")
            
        return documents
    
    def _parse_with_llama_timeout(self, file_path: Path) -> List[Document]:
        """Parse document using LlamaParse with timeout handling."""
        try:
            logger.debug(f"Starting LlamaParse for {file_path.name}")
            
            # Set a reasonable timeout for the parsing operation
            start_time = time.time()
            timeout_seconds = 60  # 1 minute timeout
            
            # Load with timeout monitoring
            llama_docs = []
            try:
                # Use async loading with timeout if available
                llama_docs = self.llama_parser.load_data(str(file_path))
                
                # Check if we've exceeded timeout
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    logger.warning(f"LlamaParse timeout ({elapsed:.1f}s) for {file_path.name}")
                    return []
                    
            except Exception as e:
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    logger.warning(f"LlamaParse timeout during processing: {e}")
                    return []
                raise e
            
            # Convert to LangChain documents
            documents = []
            for i, li_doc in enumerate(llama_docs):
                if not li_doc.text.strip():  # Skip empty documents
                    continue
                    
                doc = Document(
                    page_content=li_doc.text,
                    metadata={
                        "source": file_path.name,
                        "file_path": str(file_path),
                        "page_label": li_doc.metadata.get("page_label", f"page_{i+1}"),
                        "parser": "llama_parse",
                        "file_type": file_path.suffix.lower(),
                        "processing_time": time.time() - start_time,
                        **li_doc.metadata
                    }
                )
                documents.append(doc)
            
            logger.info(f"LlamaParse completed for {file_path.name} in {time.time() - start_time:.1f}s")
            return documents
            
        except Exception as e:
            logger.error(f"LlamaParse error for {file_path.name}: {e}")
            return []
    
    def _parse_with_community_loaders(self, file_path: Path) -> List[Document]:
        """Parse document using community loaders as fallback."""
        file_extension = file_path.suffix.lower()
        documents = []
        
        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif file_extension == ".docx":
                loader = Docx2txtLoader(str(file_path))
            elif file_extension == ".eml":
                loader = UnstructuredEmailLoader(str(file_path))
            elif file_extension in [".txt", ".md"]:
                loader = TextLoader(str(file_path), encoding='utf-8')
            else:
                logger.error(f"Unsupported file type: {file_extension}")
                return []
            
            # Load documents
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "source": file_path.name,
                    "file_path": str(file_path),
                    "parser": "community_loader",
                    "file_type": file_extension,
                })
            
            # Filter out empty documents
            documents = [doc for doc in documents if doc.page_content.strip()]
            
        except Exception as e:
            logger.error(f"Community loader failed for {file_path.name}: {e}")
            return []
        
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents into smaller pieces for better embedding quality.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of chunked Document objects with metadata
        """
        if not documents:
            logger.warning("No documents provided for chunking")
            return []
        
        logger.info(f"Chunking {len(documents)} documents")
        
        chunked_documents = []
        
        for doc_idx, document in enumerate(documents):
            try:
                # Split the document
                chunks = self.text_splitter.split_text(document.page_content)
                
                if not chunks:
                    logger.warning(f"No chunks created for document {doc_idx}")
                    continue
                
                # Create Document objects for each chunk
                for chunk_idx, chunk_text in enumerate(chunks):
                    if not chunk_text.strip():  # Skip empty chunks
                        continue
                    
                    # Generate unique chunk ID
                    chunk_id = self._generate_chunk_id(document, chunk_idx)
                    
                    chunk_doc = Document(
                        page_content=chunk_text,
                        metadata={
                            **document.metadata,  # Copy original metadata
                            "chunk_id": chunk_id,
                            "chunk_index": chunk_idx,
                            "document_index": doc_idx,
                            "total_chunks": len(chunks),
                            "chunk_size": len(chunk_text),
                            "word_count": len(chunk_text.split()),
                            "coverage_type": document.metadata.get('coverage_type', "") if document.metadata.get('coverage_type') else "",
                            "document_section": document.metadata.get('document_section', "") if document.metadata.get('document_section') else "",
                        }
                    )
                    
                    chunked_documents.append(chunk_doc)
            
            except Exception as e:
                logger.error(f"Failed to chunk document {doc_idx}: {e}")
                continue
        
        logger.info(f"Created {len(chunked_documents)} chunks from {len(documents)} documents")
        return chunked_documents
    
    def _generate_chunk_id(self, document: Document, chunk_index: int) -> str:
        """Generate a unique ID for a document chunk."""
        source = document.metadata.get('source', 'unknown')
        file_path = document.metadata.get('file_path', '')
        
        # Create a hash of the source and file path for uniqueness
        content_hash = hashlib.md5(f"{file_path}_{source}".encode()).hexdigest()[:8]
        
        return f"{source}_{content_hash}_chunk_{chunk_index}"
    
    def get_processing_stats(self, documents: List[Document]) -> dict:
        """Get statistics about processed documents."""
        if not documents:
            return {"total_documents": 0}
        
        # Group by source file
        files = {}
        total_chunks = 0
        total_content_length = 0
        
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            if source not in files:
                files[source] = {
                    'chunks': 0,
                    'content_length': 0,
                    'parser': doc.metadata.get('parser', 'unknown')
                }
            
            files[source]['chunks'] += 1
            files[source]['content_length'] += len(doc.page_content)
            total_chunks += 1
            total_content_length += len(doc.page_content)
        
        return {
            "total_documents": len(documents),
            "total_files": len(files),
            "total_chunks": total_chunks,
            "total_content_length": total_content_length,
            "average_chunk_size": total_content_length / total_chunks if total_chunks > 0 else 0,
            "files": files
        }
