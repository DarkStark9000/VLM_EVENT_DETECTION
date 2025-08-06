"""
Chat Handler Module for Multi-Turn Conversations
Manages conversational AI responses using RAG and chat history
"""

import json
import asyncio
from typing import List, Dict, Optional, Any
from openai import OpenAI
from datetime import datetime

class ChatHandler:
    def __init__(self):
        # LLM client for conversations
        self.llm_client = OpenAI(
            base_url="http://localhost:8001/v1",
            api_key="dummy"
        )
        self.llm_model = "Meta-Llama/Meta-Llama-3.1-8B-Instruct"
        
        # VLM client for follow-up video analysis
        self.vlm_client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="dummy"
        )
        self.vlm_model = "Qwen/Qwen2.5-VL-7B-Instruct"

    def format_analysis_context(self, analysis_result: Dict) -> str:
        """Format video analysis results into context for LLM (optimized for token limit)"""
        context_parts = []
        
        # Add duration
        if analysis_result.get("duration"):
            context_parts.append(f"Duration: {analysis_result['duration']}")
        
        # Add condensed summary (limit to 100 chars)
        if analysis_result.get("summary"):
            summary = analysis_result['summary'][:100] + "..." if len(analysis_result['summary']) > 100 else analysis_result['summary']
            context_parts.append(f"Summary: {summary}")
        
        # Add key events only (limit to 5 events, 50 chars each)
        events = analysis_result.get("events", [])
        if events:
            context_parts.append("Key Events:")
            for i, event in enumerate(events[:5], 1):  # Limit to 5 events
                event_text = f"{i}. {event.get('name', 'Event')} ({event.get('start', '00:00')}-{event.get('end', '00:00')})"
                desc = event.get('description', '')
                if desc and len(desc) > 50:
                    desc = desc[:50] + "..."
                if desc:
                    event_text += f": {desc}"
                context_parts.append(event_text)
        
        # Add violations (limit to 3)
        violations = analysis_result.get("guidelines_violations", [])
        if violations:
            context_parts.append("Issues:")
            for i, violation in enumerate(violations[:3], 1):  # Limit to 3
                viol_text = violation[:60] + "..." if len(violation) > 60 else violation
                context_parts.append(f"{i}. {viol_text}")
        
        return "\n".join(context_parts)

    def format_chat_history(self, chat_history: List[Dict]) -> str:
        """Format chat history for context (optimized for token limit)"""
        if not chat_history:
            return "No previous conversation."
        
        formatted_history = []
        for message in chat_history[-4:]:  # Last 4 messages (2 turns) - reduced
            role = message.get("role", "unknown")
            content = message.get("content", "")
            # Truncate long messages
            if len(content) > 80:
                content = content[:80] + "..."
            formatted_history.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(formatted_history)

    def classify_query_intent(self, query: str, analysis_context: Dict) -> str:
        """Classify the user's query intent"""
        query_lower = query.lower()
        
        # Check for specific event questions
        if any(word in query_lower for word in ["event", "happen", "occurred", "what", "when", "where"]):
            return "event_inquiry"
        
        # Check for time-specific questions
        if any(word in query_lower for word in ["time", "timestamp", "minute", "second", "start", "end"]):
            return "temporal_inquiry"
        
        # Check for safety/violation questions
        if any(word in query_lower for word in ["violation", "safety", "illegal", "wrong", "problem", "issue"]):
            return "safety_inquiry"
        
        # Check for summary/overview questions
        if any(word in query_lower for word in ["summary", "overview", "describe", "explain", "tell me about"]):
            return "summary_request"
        
        # Check for specific detail requests
        if any(word in query_lower for word in ["detail", "more", "elaborate", "explain further"]):
            return "detail_request"
        
        return "general_inquiry"

    async def generate_response(
        self, 
        query: str, 
        analysis_context: Dict, 
        chat_history: List[Dict], 
        session_id: str
    ) -> str:
        """Generate conversational response based on query and context"""
        
        try:
            # Format contexts
            video_context = self.format_analysis_context(analysis_context)
            history_context = self.format_chat_history(chat_history)
            
            # Classify query intent
            intent = self.classify_query_intent(query, analysis_context)
            
            # Build system prompt based on intent
            system_prompt = self.build_system_prompt(intent)
            
            # Build user prompt (condensed)
            user_prompt = f"""Video: {video_context}
History: {history_context}
Q: {query}
A:"""

            # Generate response using LLM
            response = await asyncio.to_thread(
                self.llm_client.chat.completions.create,
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.7,
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"

    def build_system_prompt(self, intent: str) -> str:
        """Build system prompt based on query intent"""
        
        base_prompt = """You are an expert video analysis assistant. You help users understand video content by answering questions about detected events, timelines, and safety issues.

Key Guidelines:
- Always base your responses on the provided video analysis context
- Reference specific timestamps when discussing events
- Be precise and factual, avoid speculation
- If asked about something not in the analysis, clearly state that
- Maintain conversation continuity using chat history
- Be helpful and engaging while staying professional"""

        intent_specific = {
            "event_inquiry": "\n\nFocus on: Describing specific events, their locations in time, and what exactly happened.",
            
            "temporal_inquiry": "\n\nFocus on: Providing precise timestamps, duration information, and temporal relationships between events.",
            
            "safety_inquiry": "\n\nFocus on: Safety violations, guideline adherence issues, potential risks, and regulatory compliance.",
            
            "summary_request": "\n\nFocus on: Providing comprehensive overviews while highlighting the most important aspects.",
            
            "detail_request": "\n\nFocus on: Elaborating on specific points with additional context and explanation.",
            
            "general_inquiry": "\n\nFocus on: Understanding the user's needs and providing comprehensive, helpful information."
        }
        
        return base_prompt + intent_specific.get(intent, intent_specific["general_inquiry"])

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
        
        prompt = f"""
The user originally asked: "{original_query}"
Now they're asking for clarification: "{clarification}"

Based on the video analysis context below, provide a clear, specific answer to their clarification request.

Video Analysis Context:
{self.format_analysis_context(analysis_context)}
"""

        try:
            response = await asyncio.to_thread(
                self.llm_client.chat.completions.create,
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful video analysis assistant providing clarifications."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.5
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
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
            
        except:
            return False
