# 🤖 SmartQuery-RAG – LLM-Powered Intelligent Retrieval System

A modern, full-stack Retrieval-Augmented Generation (RAG) system that enables intelligent document search and question-answering capabilities. Built with cutting-edge AI technologies for processing, vectorizing, and querying documents with semantic understanding.

## 🌟 Overview

This system combines advanced document processing, semantic search, and large language model capabilities to create an intelligent knowledge retrieval platform. Users can upload documents, process them automatically, and ask natural language questions to get contextual answers backed by relevant source citations.

### 🎯 Key Features

- **🔍 Intelligent Document Processing**: Support for multiple formats (PDF, DOCX, TXT, MD, EML) with advanced parsing
- **🧠 Semantic Search**: Vector-based similarity search using state-of-the-art embeddings
- **💬 Natural Language Q&A**: LLM-powered answer generation with source citations
- **🌐 Modern Web Interface**: Clean, responsive React frontend with real-time updates
- **⚡ High-Performance Backend**: FastAPI-based API with comprehensive monitoring
- **📊 Real-time Analytics**: System health monitoring and performance metrics
- **🔧 Scalable Architecture**: Modular design supporting easy extensions and scaling

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend       │    │   External      │
│   (React)       │───▶│   (FastAPI)      │───▶│   Services      │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • File Upload   │    │ • Document Proc. │    │ • Pinecone DB   │
│ • Search UI     │    │ • Vector Store   │    │ • Google Gemini │
│ • Results View  │    │ • Query Process  │    │ • LlamaParse    │
│ • System Stats  │    │ • Health Monitor │    │ • HuggingFace   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 🧩 Core Components

#### Backend Components
- **RAG System Orchestrator** - Coordinates all system components
- **Document Processor** - Handles parsing and chunking of various document formats
- **Vector Store Manager** - Manages embeddings and Pinecone integration
- **Query Processor** - Processes search queries and generates LLM responses
- **FastAPI Server** - RESTful API with comprehensive endpoints

#### Frontend Components
- **React Application** - Modern single-page application
- **Document Upload Interface** - Drag-and-drop file handling
- **Search Interface** - Natural language query input
- **Results Display** - Formatted answers with source citations
- **System Dashboard** - Real-time health and statistics monitoring

## 🛠️ Technology Stack

### Backend Technologies
- **Python 3.8+** - Core runtime environment
- **FastAPI** - Modern async web framework
- **LangChain** - LLM integration and document processing
- **Pinecone** - Vector database for semantic search
- **SentenceTransformers** - Text embedding generation
- **Google Gemini** - Large language model for answer generation
- **LlamaParse** - Advanced document parsing service

### Frontend Technologies
- **React 19** - Modern UI library
- **Vite** - Fast build tool and development server
- **Tailwind CSS** - Utility-first styling framework
- **Modern JavaScript** - ES2020+ features

### Infrastructure & Services
- **Pinecone Vector Database** - Serverless vector storage
- **Google AI Services** - Gemini LLM API
- **LlamaCloud** - Document parsing service
- **Docker** - Containerization support

## 📦 Project Structure

```
HackRx6.0/
├── 📁 backend/                 # Python FastAPI backend
│   ├── 📄 app.py              # Main FastAPI application
│   ├── 📄 rag_system.py       # Core RAG orchestrator
│   ├── 📄 document_processor.py # Document parsing & chunking
│   ├── 📄 vector_store_manager.py # Vector operations
│   ├── 📄 query_processor.py  # Query processing & LLM
│   ├── 📄 config.py           # Configuration management
│   ├── 📄 requirements.txt    # Python dependencies
│   ├── 📄 Dockerfile          # Container configuration
│   ├── 📁 data/               # Document storage
│   └── 📁 hf_models_cache/    # Cached ML models
├── 📁 frontend/               # React frontend application
│   ├── 📁 src/
│   │   ├── 📄 App.jsx         # Main app component
│   │   ├── 📄 main.jsx        # App entry point
│   │   └── 📁 components/
│   │       ├── 📄 RAGSystemUI.jsx # Main interface
│   │       └── 📁 ui/         # Reusable UI components
│   ├── 📄 package.json        # Node.js dependencies
│   ├── 📄 vite.config.js      # Vite configuration
│   └── 📄 index.html          # HTML template
└── 📄 README.md               # This file
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** installed
- **Node.js 16+** installed
- **API Keys** for required services:
  - Pinecone API key
  - Google AI API key
  - LlamaCloud API key (optional)

### 1. Environment Setup

Create a `.env` file in the backend directory:
```env
# Required API Keys
PINECONE_API_KEY=your_pinecone_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here

# Configuration (optional)
PINECONE_REGION=us-east-1
CHUNK_SIZE=512
CHUNK_OVERLAP=20
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt
# Or using uv (recommended)
uv sync

# Start the FastAPI server
python app.py
```

The backend will start at `http://localhost:8000`

### 3. Frontend Setup

```bash
# Navigate to frontend directory (in a new terminal)
cd frontend

# Install Node.js dependencies
npm install

# Start the development server
npm run dev
```

The frontend will start at `http://localhost:5173`

### 4. Using the System

1. **Upload Documents**: Drop files or click to upload (PDF, DOCX, TXT, MD, EML)
2. **Process Documents**: System automatically processes and vectorizes content
3. **Ask Questions**: Use natural language queries to search your documents
4. **Review Answers**: Get comprehensive answers with source citations

## 📊 API Documentation

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web interface |
| `GET` | `/health` | System health check |
| `GET` | `/stats` | System statistics |
| `POST` | `/upload` | Upload documents |
| `POST` | `/upload-urls` | Process documents from URLs |
| `POST` | `/query` | Search and query documents |

### Example API Usage

**Upload Documents:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "files=@document.pdf"
```

**Query Documents:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main points?",
    "max_results": 5,
    "score_threshold": 0.1
  }'
```

**Check System Health:**
```bash
curl "http://localhost:8000/health"
```

## ⚙️ Configuration Options

### Document Processing
- **Chunk Size**: Text chunk size for embeddings (default: 512)
- **Chunk Overlap**: Overlap between chunks (default: 20)
- **Supported Formats**: PDF, DOCX, TXT, MD, EML
- **Max File Size**: Upload limit (configurable)

### Search & Retrieval
- **Top K Documents**: Number of results to retrieve (default: 5)
- **Similarity Threshold**: Minimum similarity score (default: 0.1)
- **Embedding Model**: SentenceTransformer model for embeddings
- **LLM Temperature**: Response randomness control (default: 0.1)

### System Performance
- **Vector Database**: Pinecone serverless with auto-scaling
- **Embedding Dimensions**: 384 (all-MiniLM-L6-v2 model)
- **Batch Processing**: Optimized for document chunking and vectorization

## 🐳 Docker Deployment

### Build and Run
```bash
# Build Docker image
docker build -t rag-system ./backend

# Run with environment file
docker run -p 8000:8000 --env-file ./backend/.env rag-system
```

### Docker Compose (Optional)
Create a `docker-compose.yml` for full stack deployment:
```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    env_file:
      - ./backend/.env
  
  frontend:
    build: ./frontend
    ports:
      - "5173:5173"
    depends_on:
      - backend
```

## 📈 Performance & Monitoring

### System Metrics
- **Document Processing**: Upload and parsing success rates
- **Query Performance**: Response times and accuracy
- **Resource Usage**: Memory and CPU utilization
- **Health Monitoring**: Component status and error tracking

### Health Check Response
```json
{
  "status": "healthy",
  "uptime_seconds": 86400,
  "documents_processed": 25,
  "queries_processed": 150,
  "total_chunks": 487,
  "system_ready": true
}
```

## 🔧 Development & Contributing

### Setting up Development Environment
1. Fork the repository
2. Create a feature branch
3. Set up local environment with API keys
4. Make changes and test thoroughly
5. Submit a pull request

### Code Style Guidelines
- **Python**: Follow PEP 8, use type hints
- **JavaScript**: Use modern ES2020+ syntax
- **Documentation**: Comprehensive docstrings and comments
- **Testing**: Include unit tests for new features

## 🚨 Troubleshooting

### Common Issues

**API Key Errors:**
- Verify all API keys are correctly set in `.env`
- Check API key permissions and quotas

**Document Processing Failures:**
- Ensure supported file formats
- Check file size limits
- Verify LlamaParse connectivity

**Search Not Working:**
- Confirm vector store initialization
- Check embedding model download
- Verify Pinecone index creation

**Frontend Not Connecting:**
- Ensure backend is running on port 8000
- Check CORS configuration
- Verify API endpoints are accessible

### Debug Mode
```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
python app.py
```

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🤝 Support & Community

- **Documentation**: Check `/docs` endpoint for detailed API documentation
- **Health Status**: Monitor system at `/health` endpoint
- **Logs**: Review `rag_system.log` for detailed system information
- **Issues**: Report bugs and feature requests through GitHub issues

---

**Built with ❤️ for intelligent document processing and knowledge retrieval**

*Last updated: August 2025*
