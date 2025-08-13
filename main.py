"""
Visual Understanding Chat Assistant - Main Application
A comprehensive system for video analysis, event detection, and multi-turn conversations.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import asyncio
import os
import tempfile
import uuid
from typing import List, Dict, Optional
import json
from datetime import datetime

from video_processor import VideoProcessor
from chat_handler import ChatHandler
from chat_history_db import (
    save_chat_to_db, 
    load_chat_history, 
    create_new_session, 
    get_user_sessions
)

# Initialize FastAPI app
app = FastAPI(
    title="Visual Understanding Chat Assistant",
    description="AI-powered video analysis and conversational assistant",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize components
video_processor = VideoProcessor()
chat_handler = ChatHandler()

# In-memory storage for video analysis results (in production, use Redis/DB)
analysis_cache = {}

@app.post("/analyze-video")
async def analyze_video(video: UploadFile = File(...)):
    """Analyze uploaded video for events and content"""
    
    if not video.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        content = await video.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Process video
        analysis_result = await video_processor.analyze_video(tmp_path)
        
        # Cache results
        analysis_cache[session_id] = {
            'analysis': analysis_result,
            'video_path': tmp_path,
            'timestamp': datetime.now().isoformat()
        }
        
        # Create chat session
        create_new_session("demo_user", session_id, f"Video Analysis - {video.filename}")
        
        return JSONResponse({
            "session_id": session_id,
            "analysis": analysis_result,
            "message": "Video analysis completed successfully"
        })
        
    except Exception as e:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

class ChatRequest(BaseModel):
    message: str
    session_id: str
    user_id: str = "demo_user"

@app.post("/chat")
async def chat_with_assistant(request: ChatRequest):
    """Handle chat conversations about the analyzed video"""
    
    if request.session_id not in analysis_cache:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a video first.")
    
    try:
        # Get cached analysis
        cached_data = analysis_cache[request.session_id]
        analysis_result = cached_data['analysis']
        
        # Load chat history
        chat_history = load_chat_history(request.user_id, request.session_id, limit=10)
        
        # Generate response using chat handler
        response = await chat_handler.generate_response(
            query=request.message,
            analysis_context=analysis_result,
            chat_history=chat_history,
            session_id=request.session_id
        )
        
        # Save to database
        save_chat_to_db(request.user_id, request.message, response, request.session_id)
        
        return JSONResponse({
            "response": response,
            "session_id": request.session_id
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/sessions/{user_id}")
async def get_user_chat_sessions(user_id: str):
    """Get all chat sessions for a user"""
    sessions = get_user_sessions(user_id)
    return JSONResponse({"sessions": sessions})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Visual Understanding Chat Assistant is running"}

if __name__ == "__main__":
    
    print("🚀 Starting Visual Understanding Chat Assistant...")
    print("📹 VLM Server: http://localhost:8000")
    print("🤖 LLM Server: http://localhost:8001") 
    
    uvicorn.run(app, host="0.0.0.0", port=8002)
