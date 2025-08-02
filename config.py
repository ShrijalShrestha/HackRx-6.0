"""
Configuration module for the RAG system.
Handles all environment variables and configuration settings.
"""
import os
import logging
from typing import Optional
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    
    # Get the directory where this config.py file is located
    current_dir = Path(__file__).parent
    env_path = current_dir / '.env'
    
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment variables from {env_path}")
    else:
        print(f"⚠️ No .env file found at {env_path}")
        
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
    print("⚠️ Using system environment variables only")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class to manage all settings and API keys."""
    
    def __init__(self):
        self._setup_cache_directory()
        self._validate_environment()

    def _setup_cache_directory(self):
        """Setup Hugging Face cache directory."""
        # Choose your preferred cache directory
        cwd = os.getcwd()
        # Define the name of your cache subdirectory
        cache_subdir = "hf_models_cache"
        cache_dir = os.path.join(cwd, cache_subdir)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Set the environment variables
        os.environ['HF_HOME'] = cache_dir

    # API Keys
    LLAMA_CLOUD_API_KEY: str = os.getenv("LLAMA_CLOUD_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")
    
    # Model Configuration
    LLM_MODEL_NAME: str = "gemini-2.5-flash"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384  # all-MiniLM-L6-v2 produces 384-dimensional embeddings
    
    # Pinecone Configuration
    PINECONE_INDEX_NAME: str = "llm-rag-langchain-index"
    PINECONE_METRIC: str = "cosine"
    PINECONE_CLOUD: str = "aws"
    
    # Document Processing Configuration
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 20
    MAX_TOKENS_PER_CHUNK: int = 384  # For better embedding quality
    
    # Retrieval Configuration
    TOP_K_DOCUMENTS: int = 3
    SIMILARITY_THRESHOLD: float = 0.7
    
    # File Processing Configuration
    SUPPORTED_EXTENSIONS: tuple = (".pdf", ".docx", ".eml", ".txt", ".md")
    DATA_DIRECTORY: str = "./data"
    MAX_FILE_SIZE_MB: int = 100
    
    def _validate_environment(self) -> None:
        """Validate that all required environment variables are set."""
        required_vars = {
            "LLAMA_CLOUD_API_KEY": self.LLAMA_CLOUD_API_KEY,
            "GOOGLE_API_KEY": self.GOOGLE_API_KEY,
            "PINECONE_API_KEY": self.PINECONE_API_KEY,
            "PINECONE_REGION": self.PINECONE_REGION,
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}. "
                "Please check your .env file."
            )
        
        logger.info("Environment configuration validated successfully")
    
    @classmethod
    def get_config(cls) -> 'Config':
        """Get a singleton instance of the configuration."""
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance

# Global config instance
config = Config.get_config()
