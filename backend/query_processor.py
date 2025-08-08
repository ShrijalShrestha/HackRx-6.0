"""
Simple query processing and retrieval module for the RAG system.
Handles question answering and source citation without multi-user functionality.
"""
import logging
from typing import List, Dict, Any, Optional
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from config import config
from vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Handles query processing and retrieval-augmented generation."""
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager
        self.llm = None
        self.qa_chain = None
        self._initialize_llm()
        self._setup_qa_chain()
    
    def _initialize_llm(self) -> None:
        """Initialize the language model."""
        try:
            logger.info(f"Initializing LLM: {config.LLM_MODEL_NAME}")
            
            self.llm = ChatGoogleGenerativeAI(
                model=config.LLM_MODEL_NAME,
                google_api_key=config.GOOGLE_API_KEY,
                temperature=0.1,  # Low temperature for more consistent responses
                max_tokens=1024,  # Reasonable response length
                top_p=0.9,       # Nucleus sampling for better quality
            )
            
            logger.info("LLM initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _setup_qa_chain(self) -> None:
        """Setup the question-answering chain with custom prompt."""
        try:
            # Custom prompt template for better responses
            prompt_template = """
            You are an expert assistant trained to summarize and answer insurance policy questions accurately based only on the provided context.

            Context:
            {context}

            Question:
            {question}

            Instructions:
            1. Use only the information from the context.
            2. Provide a clear, concise, and professional summary of the relevant policy clause.
            3. Do not quote the document verbatim unless necessary — **paraphrase into natural, formal language**.
            4. If multiple conditions apply, summarize them logically and cohesively.
            5. Do not add information beyond the context.
            6. If the answer cannot be derived from the context, reply: "I don't have enough information to answer this question."

            Answer:
            """

            
            self.prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            logger.info("QA chain setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup QA chain: {e}")
            raise
    
    def process_query(self, query: str, max_results: int = 5, score_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Process a query and return the answer with sources.
        
        Args:
            query: The query string
            max_results: Maximum number of source documents to retrieve
            score_threshold: Minimum similarity score threshold
            
        Returns:
            Dictionary containing answer and sources
        """
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            # Retrieve relevant documents with scores
            search_results = self.vector_store_manager.similarity_search(
                query=query,
                k=max_results,
                score_threshold=score_threshold
            )
            
            if not search_results:
                logger.warning("No relevant documents found")
                return {
                    "answer": "I don't have enough information to answer this question based on the available documents.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Extract documents and scores from the tuples
            relevant_docs = []
            scores = []
            
            for doc, score in search_results:
                relevant_docs.append(doc)
                scores.append(float(score))
            
            logger.info(f"Retrieved {len(relevant_docs)} documents with scores: {scores}")
            
            # Prepare context from retrieved documents
            context = self._prepare_context(relevant_docs)
            
            # Generate answer using LLM
            answer = self._generate_answer(query, context)
            
            # Format sources
            sources = self._format_sources(relevant_docs, scores)
            
            # Calculate confidence based on actual similarity scores
            avg_score = sum(scores) / len(scores) if scores else 0.0
            confidence = min(0.95, avg_score * 0.8 + (len(relevant_docs) * 0.05))
            
            result = {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "num_sources": len(relevant_docs)
            }
            
            logger.info(f"Query processed successfully with {len(relevant_docs)} sources")
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "answer": f"An error occurred while processing your query: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    
    def _prepare_context(self, documents: List[Document]) -> str:
        """Prepare context string from retrieved documents."""
        try:
            context_parts = []
            
            for i, doc in enumerate(documents, 1):
                # Get source information
                source = doc.metadata.get('source', f'Document {i}')
                chunk_id = doc.metadata.get('chunk_id', f'chunk_{i}')
                
                # Format context part
                context_part = f"[Source {i}: {source}, {chunk_id}]\n{doc.page_content}\n"
                context_parts.append(context_part)
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Failed to prepare context: {e}")
            return ""
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using the LLM."""
        try:
            if not self.llm:
                return "Language model not available"
            
            # Create prompt
            formatted_prompt = self.prompt.format(
                context=context,
                question=query
            )
            
            # Generate response
            response = self.llm.invoke(formatted_prompt)
            
            # Extract text content
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def _format_sources(self, documents: List[Document], scores: List[float] = None) -> List[Dict[str, Any]]:
        """Format source documents for response with flat structure for frontend."""
        try:
            sources = []
            
            logger.info(f"Formatting {len(documents)} documents with scores: {scores}")
            
            for i, doc in enumerate(documents):
                # Get the actual similarity score
                score = scores[i] if scores and i < len(scores) else 0.0
                
                # Ensure score is a valid float between 0 and 1
                try:
                    normalized_score = float(score) if score is not None else 0.0
                    # Clamp score between 0 and 1
                    normalized_score = max(0.0, min(1.0, normalized_score))
                except (ValueError, TypeError):
                    normalized_score = 0.0
                
                # Extract source name with fallbacks
                source_name = (
                    doc.metadata.get("source") or 
                    doc.metadata.get("filename") or 
                    doc.metadata.get("file_name") or 
                    f"Document_{i+1}"
                )
                
                # Create content and preview
                content = doc.page_content or ""
                content_preview = content[:200].strip()
                if len(content) > 200:
                    content_preview += "..."
                
                # Create flat structure that matches frontend expectations
                source_info = {
                    "source": source_name,           # Frontend expects this field
                    "filename": source_name,         # Fallback field
                    "score": normalized_score,       # Frontend expects this field (0.0 to 1.0)
                    "content": content,              # Full content
                    "content_preview": content_preview or "No preview available",  # Frontend expects this
                    "chunk_id": doc.metadata.get('chunk_id', f'chunk_{i+1}'),
                    "chunk_index": doc.metadata.get('chunk_index', i),
                    # Keep original metadata as well for compatibility
                    "metadata": doc.metadata or {}
                }
                
                logger.debug(f"Formatted source {i+1}: {source_name}, score: {normalized_score:.3f}")
                sources.append(source_info)
            
            return sources
            
        except Exception as e:
            logger.error(f"Failed to format sources: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get query processor statistics."""
        try:
            return {
                "llm_model": config.LLM_MODEL_NAME,
                "llm_initialized": self.llm is not None,
                "qa_chain_ready": self.qa_chain is not None,
                "prompt_template_ready": hasattr(self, 'prompt')
            }
            
        except Exception as e:
            logger.error(f"Failed to get query processor stats: {e}")
            return {"error": str(e)}
