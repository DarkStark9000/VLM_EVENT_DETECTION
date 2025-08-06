"""
Visual Understanding Chat Assistant - Main Application
A comprehensive system for video analysis, event detection, and multi-turn conversations.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
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

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize components
video_processor = VideoProcessor()
chat_handler = ChatHandler()

# In-memory storage for video analysis results (in production, use Redis/DB)
analysis_cache = {}

@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    """Serve the main web interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Visual Understanding Chat Assistant</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 30px; border-bottom: 2px solid #e0e0e0; padding-bottom: 20px; }
            .upload-section { margin-bottom: 30px; padding: 20px; border: 2px dashed #ccc; border-radius: 10px; text-align: center; }
            .chat-section { display: none; }
            .chat-container { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 5px; background: #fafafa; }
            .message { margin-bottom: 15px; padding: 10px; border-radius: 8px; }
            .user-message { background: #007bff; color: white; margin-left: 20%; }
            .assistant-message { background: #e9ecef; margin-right: 20%; }
            .input-group { display: flex; gap: 10px; }
            input[type="text"] { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .analysis-results { margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #007bff; }
            .event-item { background: white; margin: 10px 0; padding: 10px; border-radius: 5px; border: 1px solid #ddd; }
            .loading { text-align: center; color: #666; font-style: italic; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎥 Visual Understanding Chat Assistant</h1>
                <p>Upload a video to analyze events and engage in intelligent conversations about the content</p>
            </div>
            
            <div class="upload-section" id="uploadSection">
                <h3>📁 Upload Video for Analysis</h3>
                <input type="file" id="videoFile" accept="video/*" style="margin: 10px;">
                <br>
                <button onclick="uploadVideo()">🚀 Analyze Video</button>
                <div id="uploadStatus"></div>
            </div>
            
            <div class="chat-section" id="chatSection">
                <h3>💬 Chat About Your Video</h3>
                <div class="analysis-results" id="analysisResults"></div>
                <div class="chat-container" id="chatContainer"></div>
                <div class="input-group">
                    <input type="text" id="messageInput" placeholder="Ask questions about your video..." onkeypress="handleKeyPress(event)">
                    <button onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>

        <script>
            let currentSessionId = null;
            let analysisData = null;

            async function uploadVideo() {
                const fileInput = document.getElementById('videoFile');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select a video file');
                    return;
                }
                
                const statusDiv = document.getElementById('uploadStatus');
                statusDiv.innerHTML = '<div class="loading">🔄 Analyzing video... This may take a few minutes.</div>';
                
                const formData = new FormData();
                formData.append('video', file);
                
                try {
                    const response = await fetch('/analyze-video', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        currentSessionId = result.session_id;
                        analysisData = result.analysis;
                        displayAnalysisResults(result.analysis);
                        document.getElementById('chatSection').style.display = 'block';
                        statusDiv.innerHTML = '<div style="color: green;">✅ Analysis complete! You can now chat about your video.</div>';
                    } else {
                        statusDiv.innerHTML = '<div style="color: red;">❌ Error: ' + result.detail + '</div>';
                    }
                } catch (error) {
                    statusDiv.innerHTML = '<div style="color: red;">❌ Upload failed: ' + error.message + '</div>';
                }
            }
            
            function displayAnalysisResults(analysis) {
                const resultsDiv = document.getElementById('analysisResults');
                let html = '<h4>📊 Video Analysis Results</h4>';
                
                if (analysis.events && analysis.events.length > 0) {
                    html += '<h5>🎯 Detected Events:</h5>';
                    analysis.events.forEach(event => {
                        html += `
                            <div class="event-item">
                                <strong>${event.name}</strong> (${event.start} - ${event.end})
                                ${event.description ? '<br><em>' + event.description + '</em>' : ''}
                            </div>
                        `;
                    });
                } else {
                    html += '<p>No specific events detected in this video.</p>';
                }
                
                if (analysis.summary) {
                    html += '<h5>📋 Summary:</h5><p>' + analysis.summary + '</p>';
                }
                
                resultsDiv.innerHTML = html;
            }
            
            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                
                if (!message) return;
                
                // Add user message to chat
                addMessageToChat(message, 'user');
                input.value = '';
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: message,
                            session_id: currentSessionId,
                            user_id: 'demo_user'
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        addMessageToChat(result.response, 'assistant');
                    } else {
                        addMessageToChat('Error: ' + result.detail, 'assistant');
                    }
                } catch (error) {
                    addMessageToChat('Error: ' + error.message, 'assistant');
                }
            }
            
            function addMessageToChat(message, sender) {
                const chatContainer = document.getElementById('chatContainer');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}-message`;
                messageDiv.textContent = message;
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }
        </script>
    </body>
    </html>
    """

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
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    
    print("🚀 Starting Visual Understanding Chat Assistant...")
    print("📹 VLM Server: http://localhost:8000")
    print("🤖 LLM Server: http://localhost:8001") 
    print("🌐 Web Interface: http://localhost:8002")
    
    uvicorn.run(app, host="0.0.0.0", port=8002)
