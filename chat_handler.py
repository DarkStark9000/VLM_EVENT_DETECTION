"""
Chat Handler Module for Multi-Turn Conversations
Manages conversational AI responses using RAG and chat history with industrial OOP patterns
"""

import json
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Protocol
from dataclasses import dataclass
from enum import Enum
from openai import OpenAI
from datetime import datetime


@dataclass
class ChatConfig:
    """Configuration for chat handler operations"""
    llm_base_url: str = "http://localhost:8001/v1"
    llm_api_key: str = "dummy"
    llm_model: str = "Meta-Llama/Meta-Llama-3.1-8B-Instruct"
    vlm_base_url: str = "http://localhost:8000/v1"
    vlm_api_key: str = "dummy"
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    max_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 0.9
    max_history_messages: int = 4
    max_content_length: int = 80
    max_summary_length: int = 100
    max_events: int = 5
    max_violations: int = 3


class QueryIntent(Enum):
    """Enumeration of possible query intents"""
    EVENT_INQUIRY = "event_inquiry"
    TEMPORAL_INQUIRY = "temporal_inquiry"
    SAFETY_INQUIRY = "safety_inquiry"
    SUMMARY_REQUEST = "summary_request"
    DETAIL_REQUEST = "detail_request"
    GENERAL_INQUIRY = "general_inquiry"


class ContextFormatter(Protocol):
    """Protocol for formatting analysis context"""
    
    def format_analysis_context(self, analysis_result: Dict) -> str:
        """Format video analysis results into context"""
        ...
    
    def format_chat_history(self, chat_history: List[Dict]) -> str:
        """Format chat history for context"""
        ...


class IntentClassifier(Protocol):
    """Protocol for classifying user query intent"""
    
    def classify_intent(self, query: str, analysis_context: Dict) -> QueryIntent:
        """Classify the user's query intent"""
        ...


class BaseChatHandler(ABC):
    """Abstract base class for chat handlers"""
    
    @abstractmethod
    async def generate_response(
        self, 
        query: str, 
        analysis_context: Dict, 
        chat_history: List[Dict], 
        session_id: str
    ) -> str:
        """Generate conversational response"""
        pass
    
    @abstractmethod
    def generate_follow_up_questions(self, analysis_context: Dict) -> List[str]:
        """Generate suggested follow-up questions"""
        pass


class VideoAnalysisContextFormatter:
    """Service for formatting video analysis context"""
    
    def __init__(self, config: ChatConfig):
        self._config = config
        self._logger = logging.getLogger(__name__)
    
    def format_analysis_context(self, analysis_result: Dict) -> str:
        """Format video analysis results into context for LLM (optimized for token limit)"""
        context_parts = []
        
        # Add duration
        if analysis_result.get("duration"):
            context_parts.append(f"Duration: {analysis_result['duration']}")
        
        # Add condensed summary
        if analysis_result.get("summary"):
            summary = analysis_result['summary']
            if len(summary) > self._config.max_summary_length:
                summary = summary[:self._config.max_summary_length] + "..."
            context_parts.append(f"Summary: {summary}")
        
        # Add key events only
        events = analysis_result.get("events", [])
        if events:
            context_parts.append("Key Events:")
            for i, event in enumerate(events[:self._config.max_events], 1):
                event_text = f"{i}. {event.get('name', 'Event')} ({event.get('start', '00:00')}-{event.get('end', '00:00')})"
                desc = event.get('description', '')
                if desc and len(desc) > 50:
                    desc = desc[:50] + "..."
                if desc:
                    event_text += f": {desc}"
                context_parts.append(event_text)
        
        # Add violations
        violations = analysis_result.get("guidelines_violations", [])
        if violations:
            context_parts.append("Issues:")
            for i, violation in enumerate(violations[:self._config.max_violations], 1):
                viol_text = violation[:60] + "..." if len(violation) > 60 else violation
                context_parts.append(f"{i}. {viol_text}")
        
        return "\n".join(context_parts)
    
    def format_chat_history(self, chat_history: List[Dict]) -> str:
        """Format chat history for context (optimized for token limit)"""
        if not chat_history:
            return "No previous conversation."
        
        formatted_history = []
        for message in chat_history[-self._config.max_history_messages:]:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            # Truncate long messages
            if len(content) > self._config.max_content_length:
                content = content[:self._config.max_content_length] + "..."
            formatted_history.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(formatted_history)


class QueryIntentClassifier:
    """Service for classifying user query intent"""
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        
        # Intent keywords mapping
        self._intent_keywords = {
            QueryIntent.EVENT_INQUIRY: ["event", "happen", "occurred", "what", "when", "where"],
            QueryIntent.TEMPORAL_INQUIRY: ["time", "timestamp", "minute", "second", "start", "end"],
            QueryIntent.SAFETY_INQUIRY: ["violation", "safety", "illegal", "wrong", "problem", "issue"],
            QueryIntent.SUMMARY_REQUEST: ["summary", "overview", "describe", "explain", "tell me about"],
            QueryIntent.DETAIL_REQUEST: ["detail", "more", "elaborate", "explain further"]
        }
    
    def classify_intent(self, query: str, analysis_context: Dict) -> QueryIntent:
        """Classify the user's query intent"""
        query_lower = query.lower()
        
        # Check each intent type
        for intent, keywords in self._intent_keywords.items():
            if any(word in query_lower for word in keywords):
                self._logger.debug(f"Classified query intent as: {intent.value}")
                return intent
        
        self._logger.debug("Classified query intent as: general_inquiry")
        return QueryIntent.GENERAL_INQUIRY


class LLMService:
    """Service for LLM operations in chat context"""
    
    def __init__(self, config: ChatConfig):
        self._config = config
        self._logger = logging.getLogger(__name__)
        
        try:
            self._client = OpenAI(
                base_url=config.llm_base_url,
                api_key=config.llm_api_key
            )
            self._logger.info(f"Initialized LLM service for model: {config.llm_model}")
        except Exception as e:
            self._logger.error(f"Failed to initialize LLM service: {e}")
            raise
    
    async def generate_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Generate completion from prompts"""
        try:
            response = await asyncio.to_thread(
                self._client.chat.completions.create,
                model=self._config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                top_p=self._config.top_p
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self._logger.error(f"Failed to generate completion: {e}")
            raise


class SystemPromptBuilder:
    """Service for building system prompts based on intent"""
    
    def __init__(self):
        self._base_prompt = """You are an expert video analysis assistant. You help users understand video content by answering questions about detected events, timelines, and safety issues.

Key Guidelines:
- Always base your responses on the provided video analysis context
- Reference specific timestamps when discussing events
- Be precise and factual, avoid speculation
- If asked about something not in the analysis, clearly state that
- Maintain conversation continuity using chat history
- Be helpful and engaging while staying professional"""
        
        self._intent_specific_prompts = {
            QueryIntent.EVENT_INQUIRY: "\n\nFocus on: Describing specific events, their locations in time, and what exactly happened.",
            QueryIntent.TEMPORAL_INQUIRY: "\n\nFocus on: Providing precise timestamps, duration information, and temporal relationships between events.",
            QueryIntent.SAFETY_INQUIRY: "\n\nFocus on: Safety violations, guideline adherence issues, potential risks, and regulatory compliance.",
            QueryIntent.SUMMARY_REQUEST: "\n\nFocus on: Providing comprehensive overviews while highlighting the most important aspects.",
            QueryIntent.DETAIL_REQUEST: "\n\nFocus on: Elaborating on specific points with additional context and explanation.",
            QueryIntent.GENERAL_INQUIRY: "\n\nFocus on: Understanding the user's needs and providing comprehensive, helpful information."
        }
    
    def build_prompt(self, intent: QueryIntent) -> str:
        """Build system prompt based on query intent"""
        specific_prompt = self._intent_specific_prompts.get(
            intent, 
            self._intent_specific_prompts[QueryIntent.GENERAL_INQUIRY]
        )
        return self._base_prompt + specific_prompt


class ChatHandler(BaseChatHandler):
    """Main chat handler implementing industrial OOP patterns"""
    
    def __init__(self, config: Optional[ChatConfig] = None):
        self._config = config or ChatConfig()
        self._logger = logging.getLogger(__name__)
        
        # Initialize services using dependency injection
        self._context_formatter = VideoAnalysisContextFormatter(self._config)
        self._intent_classifier = QueryIntentClassifier()
        self._llm_service = LLMService(self._config)
        self._prompt_builder = SystemPromptBuilder()
        
        # Initialize VLM client for follow-up analysis (legacy support)
        try:
            self._vlm_client = OpenAI(
                base_url=self._config.vlm_base_url,
                api_key=self._config.vlm_api_key
            )
        except Exception as e:
            self._logger.error(f"Failed to initialize VLM client: {e}")
            raise
        
        self._logger.info("ChatHandler initialized with industrial OOP structure")
    
    # Legacy methods for backward compatibility
    def format_analysis_context(self, analysis_result: Dict) -> str:
        """Legacy method - delegates to service"""
        return self._context_formatter.format_analysis_context(analysis_result)
    
    def format_chat_history(self, chat_history: List[Dict]) -> str:
        """Legacy method - delegates to service"""
        return self._context_formatter.format_chat_history(chat_history)
    
    def classify_query_intent(self, query: str, analysis_context: Dict) -> str:
        """Legacy method - delegates to service and returns string"""
        intent = self._intent_classifier.classify_intent(query, analysis_context)
        return intent.value
    
    def build_system_prompt(self, intent: str) -> str:
        """Legacy method - converts string intent to enum and delegates"""
        try:
            intent_enum = QueryIntent(intent)
            return self._prompt_builder.build_prompt(intent_enum)
        except ValueError:
            return self._prompt_builder.build_prompt(QueryIntent.GENERAL_INQUIRY)
    
    async def generate_response(
        self, 
        query: str, 
        analysis_context: Dict, 
        chat_history: List[Dict], 
        session_id: str
    ) -> str:
        """Generate conversational response based on query and context - maintains exact interface"""
        try:
            # Format contexts using services
            video_context = self._context_formatter.format_analysis_context(analysis_context)
            history_context = self._context_formatter.format_chat_history(chat_history)
            
            # Classify query intent
            intent = self._intent_classifier.classify_intent(query, analysis_context)
            
            # Build system prompt based on intent
            system_prompt = self._prompt_builder.build_prompt(intent)
            
            # Build user prompt (condensed)
            user_prompt = f"""Video: {video_context}
History: {history_context}
Q: {query}
A:"""

            # Generate response using LLM service
            return await self._llm_service.generate_completion(system_prompt, user_prompt)
            
        except Exception as e:
            self._logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    def generate_follow_up_questions(self, analysis_context: Dict) -> List[str]:
        """Generate suggested follow-up questions based on analysis"""
        questions = []
        
        events = analysis_context.get("events", [])
        violations = analysis_context.get("guidelines_violations", [])
        
        if events:
            questions.append("Can you tell me more about the events that occurred?")
            questions.append("What was the timeline of events in the video?")
            
            # Event-specific questions
            for event in events[:2]:  # First 2 events
                event_name = event.get("name", "event")
                questions.append(f"What exactly happened during the {event_name}?")
        
        if violations:
            questions.append("What safety violations were detected?")
            questions.append("How serious are the violations you found?")
        
        # General questions
        questions.extend([
            "Can you provide a summary of the entire video?",
            "Were there any unusual or concerning behaviors?",
            "What should I focus on in this video?"
        ])
        
        return questions[:5]  # Return top 5 questions
    
    async def handle_clarification_request(
        self, 
        original_query: str, 
        clarification: str, 
        analysis_context: Dict
    ) -> str:
        """Handle follow-up clarification requests"""
        try:
            video_context = self._context_formatter.format_analysis_context(analysis_context)
            
            system_prompt = "You are a helpful video analysis assistant providing clarifications."
            user_prompt = f"""
The user originally asked: "{original_query}"
Now they're asking for clarification: "{clarification}"

Based on the video analysis context below, provide a clear, specific answer to their clarification request.

Video Analysis Context:
{video_context}"""

            return await self._llm_service.generate_completion(system_prompt, user_prompt)
            
        except Exception as e:
            self._logger.error(f"Error handling clarification: {e}")
            return f"I apologize, but I couldn't process your clarification request: {str(e)}"
    
    def extract_time_references(self, query: str) -> List[str]:
        """Extract time references from user query"""
        import re
        
        # Pattern for MM:SS format
        time_pattern = r'\b(\d{1,2}):(\d{2})\b'
        matches = re.findall(time_pattern, query)
        
        time_refs = []
        for match in matches:
            minutes, seconds = match
            time_refs.append(f"{minutes.zfill(2)}:{seconds}")
        
        return time_refs
    
    def validate_time_reference(self, time_ref: str, video_duration: str) -> bool:
        """Validate if time reference is within video duration"""
        try:
            def time_to_seconds(time_str: str) -> int:
                parts = time_str.split(':')
                return int(parts[0]) * 60 + int(parts[1])
            
            ref_seconds = time_to_seconds(time_ref)
            duration_seconds = time_to_seconds(video_duration)
            
            return ref_seconds <= duration_seconds
            
        except Exception:
            return False

