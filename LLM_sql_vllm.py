
"""
RAG-based Language Model Service for Video Content Analysis
Provides semantic search and conversational AI capabilities using ChromaDB and OpenAI-compatible LLM
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI


@dataclass
class RAGConfig:
    """Configuration class for RAG system"""
    chroma_db_path: str = "chroma_db"
    llm_base_url: str = "http://localhost:8001/v1"
    llm_api_key: str = "dummy"
    model_name: str = "Meta-Llama/Meta-Llama-3.1-8B-Instruct"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    max_tokens: int = 8192
    temperature: float = 0.2
    top_p: float = 0.9
    max_results: int = 20
    default_k: int = 3


class BaseRAGService(ABC):
    """Abstract base class for RAG services"""
    
    @abstractmethod
    def retrieve_context(self, query: str, collection_name: str, session_id: str, k: int) -> Dict[str, List[List[str]]]:
        """Retrieve relevant context from vector database"""
        pass
    
    @abstractmethod
    def generate_response(self, contexts: Dict, query: str, chat_history: Optional[List] = None) -> str:
        """Generate response using language model"""
        pass


class EmbeddingService:
    """Service for handling text embeddings"""
    
    def __init__(self, model_name: str):
        self._logger = logging.getLogger(__name__)
        try:
            self._embedder = SentenceTransformer(model_name, trust_remote_code=True)
            self._logger.info(f"Initialized embedding model: {model_name}")
        except Exception as e:
            self._logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def encode(self, text: str) -> List[float]:
        """Encode text to embedding vector"""
        try:
            return self._embedder.encode([text])[0]
        except Exception as e:
            self._logger.error(f"Failed to encode text: {e}")
            raise


class VectorDatabaseService:
    """Service for vector database operations"""
    
    def __init__(self, db_path: str):
        self._logger = logging.getLogger(__name__)
        try:
            self._client = PersistentClient(path=db_path)
            self._logger.info(f"Initialized ChromaDB at: {db_path}")
        except Exception as e:
            self._logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def get_collection(self, collection_name: str):
        """Get or create collection"""
        try:
            return self._client.get_or_create_collection(name=collection_name)
        except Exception as e:
            self._logger.error(f"Failed to get collection {collection_name}: {e}")
            raise
    
    def query(self, collection, embedding: List[float], n_results: int) -> Dict:
        """Query collection with embedding"""
        try:
            return collection.query(query_embeddings=[embedding], n_results=n_results)
        except Exception as e:
            self._logger.error(f"Failed to query collection: {e}")
            raise


class LLMService:
    """Service for language model operations"""
    
    def __init__(self, config: RAGConfig):
        self._config = config
        self._logger = logging.getLogger(__name__)
        try:
            self._client = OpenAI(
                base_url=config.llm_base_url,
                api_key=config.llm_api_key
            )
            self._logger.info(f"Initialized LLM client for model: {config.model_name}")
        except Exception as e:
            self._logger.error(f"Failed to initialize LLM client: {e}")
            raise
    
    def generate_completion(self, prompt: str) -> str:
        """Generate completion from prompt"""
        try:
            response = self._client.chat.completions.create(
                model=self._config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                top_p=self._config.top_p
            )
            return response.choices[0].message.content
        except Exception as e:
            self._logger.error(f"Failed to generate completion: {e}")
            raise


class VideoContentRAGService(BaseRAGService):
    """RAG service specialized for video content analysis"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self._config = config or RAGConfig()
        self._logger = logging.getLogger(__name__)
        
        # Initialize services
        self._embedding_service = EmbeddingService(self._config.embedding_model)
        self._vector_db_service = VectorDatabaseService(self._config.chroma_db_path)
        self._llm_service = LLMService(self._config)
        
        self._logger.info("VideoContentRAGService initialized successfully")
    
    def retrieve_context(
        self, 
        query: str, 
        collection_name: str, 
        session_id: str, 
        k: Optional[int] = None
    ) -> Dict[str, List[List[str]]]:
        """Retrieve relevant context from vector database with session filtering"""
        k = k or self._config.default_k
        
        try:
            # Get collection and encode query
            collection = self._vector_db_service.get_collection(collection_name)
            embedding = self._embedding_service.encode(query)
            
            # Query for more results to allow filtering
            results = self._vector_db_service.query(
                collection, 
                embedding, 
                self._config.max_results
            )
            
            # Filter by session_id from metadata
            filtered_results = self._filter_results_by_session(
                results, 
                session_id, 
                k
            )
            
            return {
                "documents": [[doc for doc, _ in filtered_results]],
                "metadatas": [[meta for _, meta in filtered_results]]
            }
            
        except Exception as e:
            self._logger.error(f"Failed to retrieve context: {e}")
            raise
    
    def _filter_results_by_session(
        self, 
        results: Dict, 
        session_id: str, 
        k: int
    ) -> List[Tuple[str, Dict]]:
        """Filter search results by session ID"""
        filtered_results = []
        
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            if meta.get("session_id") == session_id:
                filtered_results.append((doc, meta))
                if len(filtered_results) >= k:
                    break
        
        # Fallback to unfiltered results if no session matches
        if not filtered_results:
            self._logger.warning(f"No results found for session {session_id}, using unfiltered results")
            filtered_results = list(zip(
                results['documents'][0][:k], 
                results['metadatas'][0][:k]
            ))
        
        return filtered_results
    
    def generate_response(
        self, 
        contexts: Dict, 
        query: str, 
        chat_history: Optional[List] = None
    ) -> str:
        """Generate response using retrieved contexts and chat history"""
        try:
            # Format context
            context_text = self._format_context(contexts)
            
            # Build prompt
            prompt = self._build_prompt(context_text, query, chat_history or [])
            
            # Generate response
            return self._llm_service.generate_completion(prompt)
            
        except Exception as e:
            self._logger.error(f"Failed to generate response: {e}")
            raise
    
    def _format_context(self, contexts: Dict) -> str:
        """Format retrieved contexts into readable text"""
        try:
            return "\n---\n".join([
                f"[{meta['video_name']}] ({meta['start_time']} - {meta['end_time']}): {doc}"
                for meta, doc in zip(contexts['metadatas'][0], contexts['documents'][0])
            ])
        except KeyError as e:
            self._logger.error(f"Missing required metadata field: {e}")
            raise
    
    def _build_prompt(self, context_text: str, query: str, chat_history: List) -> str:
        """Build prompt for language model"""
        return f"""You are an expert video content analyzer.

Analyze the following context segments retrieved from videos. Your task is to answer the question strictly based on the content, without generating random or speculative answers.

- Provide a concise summary in paragraph form.
- If multiple segments refer to the same video name, combine their information into a single response.
- Mention the relevant timestamps, if available.
- Rephrase the final answer according to the language used in the original query.

Ensure that your response is factual, clearly structured, and reflects only what is observed or stated in the video content.

Context:
{context_text}

Chat History: {chat_history}

Question: {query}
Answer:"""


# Legacy function wrappers for backward compatibility
_rag_service = None

def _get_rag_service() -> VideoContentRAGService:
    """Get singleton RAG service instance"""
    global _rag_service
    if _rag_service is None:
        _rag_service = VideoContentRAGService()
    return _rag_service

def retrieve_context(query: str, collection_name: str, session_id: str, k: int = 3) -> Dict[str, List[List[str]]]:
    """Legacy function wrapper for backward compatibility"""
    return _get_rag_service().retrieve_context(query, collection_name, session_id, k)

def ask_model(contexts: Dict, query: str, chat_history: Optional[List] = None) -> str:
    """Legacy function wrapper for backward compatibility"""
    return _get_rag_service().generate_response(contexts, query, chat_history)
