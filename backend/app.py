"""
Simple FastAPI Backend for RAG System
====================================

A basic FastAPI backend for document processing, vectorization, and semantic search
without multi-user functionality - single user/single namespace.
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import shutil
from datetime import datetime
import requests
from urllib.parse import urlparse
import mimetypes
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from config import config
from rag_system import RAGSystem

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables
rag_system = None
system_stats = {
    "startup_time": datetime.now(),
    "documents_processed": 0,
    "queries_processed": 0,
    "total_chunks": 0,
    "last_activity": None,
    "errors": 0
}

# Pydantic models (keeping all existing models unchanged)
class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask")
    max_results: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.1, ge=0.0, le=1.0)

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    processing_time: float
    metadata: Optional[Dict[str, Any]] = None

class UploadResponse(BaseModel):
    success: bool
    message: str
    files_processed: int
    chunks_created: int
    processing_time: float

class URLUploadRequest(BaseModel):
    urls: List[str] = Field(..., description="List of URLs to download and process")
    timeout: int = Field(default=30, ge=5, le=300, description="Download timeout in seconds")

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    documents_processed: int
    queries_processed: int
    total_chunks: int
    last_activity: Optional[str]
    system_ready: bool

class StatsResponse(BaseModel):
    system_ready: bool
    total_documents: int
    total_chunks: int
    vector_count: int
    index_name: str
    embedding_model: str

# Initialize RAG system
def initialize_rag_system():
    """Initialize the RAG system without auto-processing existing documents."""
    global rag_system
    
    try:
        logger.info("Initializing RAG system...")
        rag_system = RAGSystem()
        
        # Just initialize the system without processing existing documents
        # Documents will only be processed when explicitly uploaded or from URLs
        logger.info("RAG system initialized successfully (ready for document upload)")
            
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting up RAG System...")
    initialize_rag_system()
    logger.info("RAG System startup completed")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG System...")
    # Add any cleanup logic here if needed
    logger.info("RAG System shutdown completed")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="RAG System API",
    description="A simple Retrieval-Augmented Generation system for document Q&A",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    Server is running!
    """

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health."""
    global system_stats
    
    uptime = (datetime.now() - system_stats["startup_time"]).total_seconds()
    
    return HealthResponse(
        status="healthy" if rag_system and rag_system.is_ready else "not_ready",
        uptime_seconds=uptime,
        documents_processed=system_stats["documents_processed"],
        queries_processed=system_stats["queries_processed"],
        total_chunks=system_stats["total_chunks"],
        last_activity=system_stats["last_activity"],
        system_ready=rag_system and rag_system.is_ready
    )

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    if not rag_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    stats = rag_system.get_system_stats()
    
    return StatsResponse(
        system_ready=rag_system.is_ready,
        total_documents=system_stats["documents_processed"],
        total_chunks=system_stats["total_chunks"],
        vector_count=stats.get("total_vectors", 0),
        index_name=config.PINECONE_INDEX_NAME,
        embedding_model=config.EMBEDDING_MODEL_NAME
    )

@app.post("/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process documents."""
    global system_stats
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    start_time = time.time()
    temp_dir = None
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        uploaded_files = []
        
        # Save uploaded files
        for file in files:
            if file.filename:
                file_path = Path(temp_dir) / file.filename
                
                # Check file extension
                if not any(file.filename.lower().endswith(ext) for ext in config.SUPPORTED_EXTENSIONS):
                    continue
                
                # Save file
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                uploaded_files.append(file.filename)
        
        if not uploaded_files:
            raise HTTPException(status_code=400, detail="No valid files uploaded")
        
        # Process documents
        if rag_system.is_ready:
            success = rag_system.add_documents_from_directory(temp_dir)
        else:
            success = rag_system.setup_system(data_directory=temp_dir)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process documents")
        
        # Copy files to data directory
        data_dir = Path("./data")
        data_dir.mkdir(exist_ok=True)
        
        for filename in uploaded_files:
            src = Path(temp_dir) / filename
            dst = data_dir / filename
            if src.exists():
                shutil.copy2(src, dst)
        
        # Update stats
        processing_time = time.time() - start_time
        system_stats["documents_processed"] += len(uploaded_files)
        system_stats["last_activity"] = datetime.now().isoformat()
        
        stats = rag_system.get_system_stats()
        chunks_created = stats.get("total_vectors", 0) - system_stats["total_chunks"]
        system_stats["total_chunks"] = stats.get("total_vectors", 0)
        
        return UploadResponse(
            success=True,
            message=f"Successfully processed {len(uploaded_files)} files",
            files_processed=len(uploaded_files),
            chunks_created=chunks_created,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        system_stats["errors"] += 1
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    finally:
        # Cleanup
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

def download_file_from_url(url: str, timeout: int = 30) -> tuple[str, bytes]:
    """Download a file from URL and return filename and content."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, timeout=timeout, headers=headers, stream=True)
        response.raise_for_status()
        
        # Try to get filename from URL or Content-Disposition header
        filename = None
        if 'content-disposition' in response.headers:
            content_disposition = response.headers['content-disposition']
            if 'filename=' in content_disposition:
                filename = content_disposition.split('filename=')[1].strip('"')
        
        if not filename:
            parsed_url = urlparse(url)
            filename = Path(parsed_url.path).name
            if not filename or '.' not in filename:
                # Try to determine extension from content type
                content_type = response.headers.get('content-type', '')
                ext = mimetypes.guess_extension(content_type.split(';')[0]) or '.txt'
                filename = f"document_{int(time.time())}{ext}"
        
        # Download content
        content = b''
        for chunk in response.iter_content(chunk_size=8192):
            content += chunk
        
        return filename, content
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download from {url}: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing URL {url}: {str(e)}")

@app.post("/upload-urls", response_model=UploadResponse)
async def upload_from_urls(request: URLUploadRequest):
    """Download and process documents from URLs."""
    global system_stats
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not request.urls:
        raise HTTPException(status_code=400, detail="No URLs provided")
    
    start_time = time.time()
    temp_dir = None
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        downloaded_files = []
        
        # Download files from URLs
        for url in request.urls:
            try:
                logger.info(f"Downloading from URL: {url}")
                filename, content = download_file_from_url(url, request.timeout)
                
                # Check file extension
                if not any(filename.lower().endswith(ext) for ext in config.SUPPORTED_EXTENSIONS):
                    logger.warning(f"Skipping unsupported file type: {filename}")
                    continue
                
                # Save file
                file_path = Path(temp_dir) / filename
                with open(file_path, "wb") as f:
                    f.write(content)
                downloaded_files.append(filename)
                logger.info(f"Successfully downloaded: {filename} ({len(content)} bytes)")
                
            except HTTPException as e:
                logger.error(f"Failed to download {url}: {e.detail}")
                # Continue with other URLs instead of failing completely
                continue
            except Exception as e:
                logger.error(f"Unexpected error downloading {url}: {str(e)}")
                continue
        
        if not downloaded_files:
            raise HTTPException(status_code=400, detail="No valid files could be downloaded from the provided URLs")
        
        # Process documents using the same logic as file upload
        if rag_system.is_ready:
            success = rag_system.add_documents_from_directory(temp_dir)
        else:
            success = rag_system.setup_system(data_directory=temp_dir)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process downloaded documents")
        
        # Copy files to data directory
        data_dir = Path("./data")
        data_dir.mkdir(exist_ok=True)
        
        for filename in downloaded_files:
            src = Path(temp_dir) / filename
            dst = data_dir / filename
            if src.exists():
                shutil.copy2(src, dst)
        
        # Update stats
        processing_time = time.time() - start_time
        system_stats["documents_processed"] += len(downloaded_files)
        system_stats["last_activity"] = datetime.now().isoformat()
        
        stats = rag_system.get_system_stats()
        chunks_created = stats.get("total_vectors", 0) - system_stats["total_chunks"]
        system_stats["total_chunks"] = stats.get("total_vectors", 0)
        
        return UploadResponse(
            success=True,
            message=f"Successfully processed {len(downloaded_files)} files from URLs",
            files_processed=len(downloaded_files),
            chunks_created=chunks_created,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        system_stats["errors"] += 1
        logger.error(f"URL upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"URL upload failed: {str(e)}")
    
    finally:
        # Cleanup
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query."""
    global system_stats
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not rag_system.is_ready:
        raise HTTPException(status_code=400, detail="No documents available. Please upload documents first.")
    
    start_time = time.time()
    
    try:
        result = rag_system.query(
            query=request.query,
            max_results=request.max_results,
            score_threshold=request.score_threshold
        )
        
        if not result:
            raise HTTPException(status_code=500, detail="Query processing failed")
        
        processing_time = time.time() - start_time
        
        # Update stats
        system_stats["queries_processed"] += 1
        system_stats["last_activity"] = datetime.now().isoformat()
        
        return QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            query=request.query,
            processing_time=processing_time,
            metadata=result.get("metadata", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        system_stats["errors"] += 1
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

if __name__ == "__main__":
    print("Starting RAG System FastAPI Server...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
