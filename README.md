# Enhanced RAG System - FastAPI Backend

A comprehensive Retrieval-Augmented Generation (RAG) system with enhanced metadata handling, document processing, and semantic search capabilities.

## 🚀 Features

### Core Capabilities
- **Advanced Document Processing**: Multi-format support (PDF, DOCX, TXT, MD, EML) with LlamaParse and community loaders
- **Enhanced Metadata Handling**: Comprehensive metadata preservation and safe extraction with fallbacks
- **Semantic Search**: SentenceTransformer embeddings with Pinecone vector database
- **FastAPI Backend**: RESTful API with automatic documentation and enhanced error handling
- **Real-time Monitoring**: Health checks, performance metrics, and system statistics
- **Robust Error Handling**: Graceful degradation and comprehensive logging

### Recent Enhancements
- ✅ **Fixed Metadata Bugs**: Resolved `coverage_type` and `document_section` handling issues
- ✅ **Enhanced FastAPI Routes**: Added comprehensive upload, query, health, and stats endpoints
- ✅ **Improved Error Handling**: Safe metadata extraction with fallbacks
- ✅ **Performance Monitoring**: Added uptime tracking, query statistics, and error rates
- ✅ **File Management**: Enhanced upload validation, processing statistics, and cleanup utilities

## 📋 API Endpoints

### Document Management
- `POST /upload` - Upload and process documents with validation
- `GET /documents` - List processed documents with filtering
- `DELETE /documents/{doc_name}` - Delete specific document
- `DELETE /reset` - Reset entire vector store

### Search & Query
- `POST /query` - Process search queries with enhanced metadata
- `GET /search/similar/{doc_id}` - Find similar documents

### System Monitoring
- `GET /health` - System health check with performance metrics
- `GET /stats` - Detailed system statistics
- `GET /` - Frontend web interface

## 🛠️ Installation & Setup

### Prerequisites
```bash
Python 3.8+
Node.js (for frontend development)
```

### Environment Variables
Create a `.env` file in the project root:
```env
# Required API Keys
PINECONE_API_KEY=your_pinecone_api_key_here
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Optional Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_FILE_SIZE_MB=50
TOP_K_DOCUMENTS=5
```

### Installation
```bash
# Clone and navigate to project
cd backend

# Install dependencies
pip install -r requirements.txt
# OR using uv (recommended)
uv sync

# Run the application
python app.py
```

### Docker Setup 
```bash
# Build Docker image
docker build -t rag-backend .

# Run container
docker run -p 8000:8000 --env-file .env rag-backend
```

## 🔧 Configuration

### Document Processing
- **Chunk Size**: Text chunk size for embedding (default: 1000)
- **Chunk Overlap**: Overlap between chunks (default: 200)
- **Supported Formats**: PDF, DOCX, TXT, MD, EML
- **Max File Size**: Configurable upload limit (default: 50MB)

### Vector Storage
- **Database**: Pinecone (serverless)
- **Embedding Model**: SentenceTransformer all-MiniLM-L6-v2 (384 dimensions)
- **Similarity Metric**: Cosine similarity
- **Index Configuration**: Auto-scaling serverless spec

### LLM Integration
- **Provider**: Google Gemini
- **Model**: gemini-pro (configurable)
- **Temperature**: 0.1 (for consistent responses)
- **Max Tokens**: 1024

## 📊 Usage Examples

### Upload Documents
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "files=@document1.pdf" \
  -F "files=@document2.docx"
```

### Query Documents
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key findings?",
    "max_sources": 5,
    "include_sources": true,
    "include_metadata": true
  }'
```

### Check System Health
```bash
curl "http://localhost:8000/health"
```

## 🔍 Enhanced Metadata Features

### Document Metadata
- `source`: Original filename
- `file_path`: Full file path
- `file_type`: File extension
- `parser`: Parser used (llama_parse or community_loader)
- `chunk_id`: Unique chunk identifier
- `coverage_type`: Document coverage classification
- `document_section`: Section within document
- `word_count`: Number of words in chunk
- `processing_time`: Time taken to process

### Safe Metadata Extraction
The system now includes robust metadata handling:
```python
def safe_get(key: str, default: Any = ""):
    """Safely get metadata value with fallback."""
    try:
        value = metadata.get(key, default)
        return value if value is not None else default
    except (AttributeError, TypeError):
        return default
```

## 📈 Performance Monitoring

### Health Check Response
```json
{
  "status": "healthy",
  "timestamp": "2025-01-31T10:30:00Z",
  "system_ready": true,
  "components": {
    "rag_system": true,
    "vector_store": true,
    "embeddings": true,
    "llm": true
  },
  "performance_metrics": {
    "uptime_hours": 24.5,
    "total_queries": 150,
    "total_documents": 25,
    "error_rate": 0.02
  }
}
```

### Statistics Response
```json
{
  "system_ready": true,
  "total_documents": 25,
  "total_chunks": 487,
  "total_vectors": 487,
  "index_name": "test-index",
  "embedding_model": "all-MiniLM-L6-v2",
  "last_updated": "2025-01-31T10:30:00Z"
}
```

## 🐛 Bug Fixes Applied

### Metadata Handling Issues
- **Problem**: `'dict' object has no attribute 'metadata'` errors
- **Solution**: Added safe metadata extraction with null checks and fallbacks
- **Impact**: Eliminated runtime errors during query processing

### Coverage Type and Document Section
- **Problem**: Inconsistent handling of `coverage_type` and `document_section` fields
- **Solution**: Implemented proper null checking with empty string defaults
- **Code Fix**:
```python
"coverage_type": document.metadata.get('coverage_type', "") if document.metadata.get('coverage_type') else "",
"document_section": document.metadata.get('document_section', "") if document.metadata.get('document_section') else "",
```

## 🧪 Testing

### Manual Testing
```bash
# Test upload
curl -X POST "http://localhost:8000/upload" -F "files=@test.pdf"

# Test query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "test question"}'

# Test health
curl "http://localhost:8000/health"
```

### Automated Testing
```bash
# Run pytest (if test files exist)
pytest tests/

# Load testing
locust -f load_test.py --host=http://localhost:8000
```

## 🧹 Maintenance

### Cleanup Unnecessary Files
```bash
python cleanup.py
```

### Log Management
- Logs are written to `rag_system.log`
- Configure log rotation in production
- Monitor error patterns and performance

### Database Maintenance
- Pinecone handles scaling automatically
- Monitor vector count and query performance
- Consider index optimization for large datasets

## 🚧 Troubleshooting

### Common Issues

1. **Metadata Errors**
   - Ensure all metadata fields have defaults
   - Check document processing pipeline
   - Verify chunk creation logic

2. **Upload Failures**
   - Check file size limits
   - Verify supported file types
   - Review temporary directory permissions

3. **Query Failures**
   - Verify vector store initialization
   - Check embedding model loading
   - Confirm LLM API keys

4. **Performance Issues**
   - Monitor memory usage
   - Check embedding batch sizes
   - Review chunk sizes and overlap

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python app.py
```

## 📝 Development

### Contributing
1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Update documentation
5. Submit pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add comprehensive docstrings
- Include error handling

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Support

For support, please:
1. Check the troubleshooting section
2. Review the API documentation at `/docs`
3. Check system health at `/health`
4. Review logs in `rag_system.log`

---

**Last Updated**: January 31, 2025
**Version**: 2.0.0 (Enhanced)
**Author**: RAG System Development Team
