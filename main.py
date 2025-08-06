#!/usr/bin/env python3
"""
Visual Understanding Chat Assistant - Round 1 Implementation
Hackathon Submission: Agentic chat assistant for video processing and multi-turn conversations

This is the main entry point that integrates:
- Video event recognition and summarization using Tarsier2-7B (open-source SOTA VLM)
- RAG-based retrieval and conversation using open-source LLM 
- Multi-turn conversation support with chat history
- ChromaDB for storing and retrieving video segments
"""

import os
import sys
import argparse
import uuid
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our custom modules
from vlm import process_video_with_llava, Event
from LLM_sql import retrieve_context, ask_model, chroma_client, embedder
from chat_history_db import (
    save_chat_to_db, 
    load_chat_history, 
    create_new_session, 
    get_user_sessions,
    rename_session
)

class VideoUnderstandingAssistant:
    """
    Main class for the Visual Understanding Chat Assistant
    """
    
    def __init__(self, user_name: str = "hackathon_user"):
        self.user_name = user_name
        self.session_id = None
        self.collection_name = "video_segments"
        
        logger.info("Visual Understanding Chat Assistant - Round 1")
        logger.info("=" * 60)
        logger.info("Features:")
        logger.info("- Video Event Recognition & Summarization")
        logger.info("- Multi-turn Conversations with Context")
        logger.info("- Open-source Models (LLaVA-1.5-7B + Phi-3.5)")
        logger.info("- RAG-based Video Content Retrieval")
        logger.info("=" * 60)
        
    def process_video(self, video_path: str) -> bool:
        """
        Process a video file and store segments in ChromaDB
        """
        logger.info(f"Processing video: {video_path}")
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found at {video_path}")
            return False
        
        try:
            # Step 1: Extract events using LLaVA
            logger.info("Analyzing video with LLaVA-1.5-7B")
            events = process_video_with_llava(video_path)
            
            if not events:
                logger.warning("No events detected in the video")
                return False
            
            # Step 2: Store in ChromaDB for retrieval
            logger.info("Storing video segments in ChromaDB")
            collection = chroma_client.get_or_create_collection(name=self.collection_name)
            
            documents = []
            metadatas = []
            ids = []
            
            for i, event in enumerate(events):
                # Create document text for embedding
                doc_text = f"Event: {event['event']} from {event['start']} to {event['end']}. Description: {event['description']}"
                documents.append(doc_text)
                
                # Create metadata
                metadata = {
                    "video_path": video_path,
                    "video_name": Path(video_path).stem,
                    "event_name": event['event'],
                    "start_time": event['start'],
                    "end_time": event['end'],
                    "description": event['description'],
                    "session_id": self.session_id or "default",
                    "timestamp": datetime.now().isoformat()
                }
                metadatas.append(metadata)
                
                # Create unique ID
                ids.append(f"{Path(video_path).stem}_{i}_{uuid.uuid4().hex[:8]}")
            
            # Add to ChromaDB
            embeddings = embedder.encode(documents).tolist()
            collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully stored {len(events)} video segments")
            return True
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return False
    
    def start_new_session(self, title: str = None) -> str:
        """
        Start a new chat session
        """
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        session_title = title or f"Video Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        try:
            create_new_session(self.user_name, session_id, session_title)
            self.session_id = session_id
            logger.info(f"Started new session: {session_title} (ID: {session_id})")
            return session_id
        except Exception as e:
            logger.warning(f"Could not create session in DB: {e}")
            self.session_id = session_id
            return session_id
    
    def chat(self, query: str) -> str:
        """
        Process a chat query with context retrieval
        """
        if not self.session_id:
            self.start_new_session()
        
        try:
            # Step 1: Retrieve relevant context from ChromaDB
            logger.info("Searching for relevant video segments")
            contexts = retrieve_context(
                query=query, 
                collection_name=self.collection_name,
                session_id=self.session_id,
                k=3
            )
            
            if not contexts['documents'][0]:
                response = "I don't have any video content to analyze for your question. Please process a video first using the 'process_video' command."
            else:
                # Step 2: Load chat history for context
                chat_history = load_chat_history(self.user_name, self.session_id, limit=6)
                
                # Step 3: Generate response using LLM + RAG
                logger.info("Generating response with open-source LLM")
                response = ask_model(contexts, query, chat_history)
            
            # Step 4: Save to chat history
            try:
                save_chat_to_db(self.user_name, query, response, self.session_id)
            except Exception as e:
                logger.warning(f"Could not save to chat history: {e}")
            
            return response
            
        except Exception as e:
            error_msg = f"I encountered an error while processing your question: {e}"
            logger.error(error_msg)
            return error_msg
    
    def list_sessions(self):
        """
        List all chat sessions for the user
        """
        try:
            sessions = get_user_sessions(self.user_name)
            if sessions:
                logger.info(f"Chat Sessions for {self.user_name}:")
                for session in sessions:
                    logger.info(f"  - {session['title']} (ID: {session['session_id']})")
                    if session['last_used']:
                        logger.info(f"    Last used: {session['last_used']}")
            else:
                logger.info("No chat sessions found")
        except Exception as e:
            logger.error(f"Error retrieving sessions: {e}")

def main():
    """
    Main function with CLI interface
    """
    parser = argparse.ArgumentParser(
        description="Visual Understanding Chat Assistant for Video Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --video /path/to/video.mp4 --interactive
  python main.py --query "What traffic violations did you see?"
  python main.py --demo
        """
    )
    
    parser.add_argument("--video", "-v", help="Path to video file to process")
    parser.add_argument("--query", "-q", help="Ask a question about processed videos")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive chat mode")
    parser.add_argument("--demo", action="store_true", help="Run demo with sample interactions")
    parser.add_argument("--user", default="hackathon_user", help="User name for chat sessions")
    parser.add_argument("--session", help="Session ID to continue previous chat")
    
    args = parser.parse_args()
    
    # Initialize assistant
    assistant = VideoUnderstandingAssistant(user_name=args.user)
    
    # Set session if provided
    if args.session:
        assistant.session_id = args.session
    
    # Process video if provided
    if args.video:
        success = assistant.process_video(args.video)
        if not success:
            return 1
    
    # Handle different modes
    if args.demo:
        run_demo(assistant)
    elif args.query:
        response = assistant.chat(args.query)
        print(f"\nAssistant: {response}")
    elif args.interactive:
        run_interactive_mode(assistant)
    else:
        logger.info("Use --help to see available options")
        return 1
    
    return 0

def run_demo(assistant):
    """
    Run a demonstration of the system capabilities
    """
    logger.info("DEMO MODE - Visual Understanding Chat Assistant")
    logger.info("=" * 60)
    
    # Check if we have any video to demonstrate with
    demo_queries = [
        "What events did you detect in the video?",
        "Were there any traffic violations?",
        "Can you describe what happened around the 30-second mark?",
        "What safety issues did you observe?",
        "Summarize the main events in chronological order"
    ]
    
    logger.info("Demo queries that you can try:")
    for i, query in enumerate(demo_queries, 1):
        logger.info(f"{i}. {query}")
    
    logger.info("Note: Process a video first using --video option, then try these queries!")

def run_interactive_mode(assistant):
    """
    Run interactive chat mode
    """
    logger.info("INTERACTIVE MODE")
    logger.info("Commands:")
    logger.info("  'process <video_path>' - Process a new video")
    logger.info("  'sessions' - List chat sessions")
    logger.info("  'quit' or 'exit' - Exit the program")
    logger.info("  Any other text - Ask a question about the videos")
    logger.info("-" * 40)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif user_input.startswith('process '):
                video_path = user_input[8:].strip()
                assistant.process_video(video_path)
            elif user_input.lower() == 'sessions':
                assistant.list_sessions()
            elif user_input:
                response = assistant.chat(user_input)
                print(f"\nAssistant: {response}")
            else:
                print("Please enter a question or command.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")

if __name__ == "__main__":
    sys.exit(main())
