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
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #1f2937;
            }
            
            .sidebar {
                position: fixed;
                left: 0;
                top: 0;
                width: 250px;
                height: 100vh;
                background: rgba(31, 41, 55, 0.95);
                backdrop-filter: blur(10px);
                padding: 20px;
                z-index: 100;
            }
            
            .sidebar h2 {
                color: white;
                font-size: 18px;
                margin-bottom: 30px;
                font-weight: 600;
            }
            
            .sidebar-menu {
                list-style: none;
            }
            
            .sidebar-menu li {
                margin-bottom: 8px;
            }
            
            .sidebar-menu a {
                color: #9ca3af;
                text-decoration: none;
                padding: 12px 16px;
                border-radius: 8px;
                display: block;
                transition: all 0.2s;
                font-weight: 400;
            }
            
            .sidebar-menu a:hover, .sidebar-menu a.active {
                background: rgba(99, 102, 241, 0.2);
                color: white;
            }
            
            .main-content {
                margin-left: 250px;
                padding: 40px;
                min-height: 100vh;
            }
            
            .header {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 16px;
                padding: 24px 32px;
                margin-bottom: 32px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .header h1 {
                font-size: 28px;
                font-weight: 700;
                color: #1f2937;
                margin-bottom: 8px;
            }
            
            .header p {
                color: #6b7280;
                font-size: 16px;
                font-weight: 400;
            }
            
            .content-grid {
                display: grid;
                grid-template-columns: 400px 1fr;
                gap: 32px;
                align-items: start;
                height: calc(100vh - 200px);
            }
            
            .upload-panel {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 16px;
                padding: 32px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .upload-panel h3 {
                font-size: 20px;
                font-weight: 600;
                margin-bottom: 8px;
                color: #1f2937;
            }
            
            .upload-panel p {
                color: #6b7280;
                margin-bottom: 24px;
                font-size: 14px;
            }
            
            .file-upload-area {
                border: 2px dashed #d1d5db;
                border-radius: 12px;
                padding: 40px 20px;
                text-align: center;
                background: #f9fafb;
                transition: all 0.2s;
                cursor: pointer;
                margin-bottom: 20px;
            }
            
            .file-upload-area:hover {
                border-color: #6366f1;
                background: #f0f9ff;
            }
            
            .file-upload-area.dragover {
                border-color: #6366f1;
                background: #eff6ff;
            }
            
            .upload-icon {
                width: 48px;
                height: 48px;
                margin: 0 auto 16px;
                background: #e5e7eb;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 20px;
            }
            
            .upload-text {
                font-size: 16px;
                color: #374151;
                margin-bottom: 8px;
                font-weight: 500;
            }
            
            .upload-subtext {
                font-size: 14px;
                color: #9ca3af;
            }
            
            .btn-primary {
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s;
                font-size: 14px;
                width: 100%;
            }
            
            .btn-primary:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
            }
            
            .btn-primary:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            .results-panel {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 16px;
                padding: 24px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                display: none;
                height: fit-content;
                max-height: calc(100vh - 200px);
                overflow-y: auto;
            }
            
            .results-panel h3 {
                font-size: 20px;
                font-weight: 600;
                margin-bottom: 24px;
                color: #1f2937;
            }
            
            .event-card {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 12px;
                transition: all 0.2s;
            }
            
            .event-card:hover {
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                transform: translateY(-1px);
            }
            
            .event-title {
                font-weight: 600;
                color: #1f2937;
                margin-bottom: 6px;
                font-size: 15px;
            }
            
            .event-time {
                color: #6366f1;
                font-size: 13px;
                font-weight: 500;
                margin-bottom: 8px;
                font-family: monospace;
            }
            
            .event-description {
                color: #6b7280;
                font-size: 13px;
                line-height: 1.4;
            }
            
            .chat-panel {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 16px;
                padding: 32px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                display: none;
                grid-column: 1 / -1;
                margin-top: 32px;
            }
            
            .chat-container {
                height: 400px;
                overflow-y: auto;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                background: #f9fafb;
            }
            
            .message {
                margin-bottom: 16px;
                display: flex;
                align-items: flex-start;
                gap: 12px;
            }
            
            .message-content {
                max-width: 70%;
                padding: 12px 16px;
                border-radius: 12px;
                font-size: 14px;
                line-height: 1.5;
            }
            
            .user-message .message-content {
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                color: white;
                margin-left: auto;
            }
            
            .assistant-message .message-content {
                background: white;
                border: 1px solid #e5e7eb;
                color: #374151;
            }
            
            .chat-input {
                display: flex;
                gap: 12px;
            }
            
            .chat-input input {
                flex: 3;
                padding: 12px 16px;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                font-size: 14px;
                outline: none;
                transition: all 0.2s;
            }
            
            .chat-input .btn-primary {
                flex: 1;
                min-width: 100px;
                width: auto;
            }
            
            .chat-input input:focus {
                border-color: #6366f1;
                box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
            }
            
            .status-message {
                padding: 12px;
                border-radius: 8px;
                margin-top: 16px;
                font-size: 14px;
                font-weight: 500;
            }
            
            .status-success {
                background: #d1fae5;
                color: #065f46;
                border: 1px solid #a7f3d0;
            }
            
            .status-error {
                background: #fee2e2;
                color: #991b1b;
                border: 1px solid #fca5a5;
            }
            
            .status-loading {
                background: #dbeafe;
                color: #1d4ed8;
                border: 1px solid #93c5fd;
            }
            
            .hidden { display: none !important; }
            
            .summary-card {
                background: #eff6ff;
                border: 1px solid #bfdbfe;
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 20px;
            }
            
            .summary-title {
                font-weight: 600;
                color: #1e40af;
                margin-bottom: 10px;
                font-size: 15px;
            }
            
            .summary-text {
                color: #374151;
                font-size: 13px;
                line-height: 1.5;
                max-height: 120px;
                overflow-y: auto;
            }
            
            .events-container {
                max-height: calc(100vh - 400px);
                overflow-y: auto;
                padding-right: 8px;
            }
            
            .events-container::-webkit-scrollbar {
                width: 6px;
            }
            
            .events-container::-webkit-scrollbar-track {
                background: #f1f5f9;
                border-radius: 3px;
            }
            
            .events-container::-webkit-scrollbar-thumb {
                background: #cbd5e1;
                border-radius: 3px;
            }
            
            .events-container::-webkit-scrollbar-thumb:hover {
                background: #94a3b8;
            }
            
            .events-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 16px;
                padding-bottom: 8px;
                border-bottom: 1px solid #e5e7eb;
            }
            
            .events-count {
                background: #6366f1;
                color: white;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 500;
            }
        </style>
    </head>
    <body>
        <div class="sidebar">
            <h2>Visual Assistant</h2>
            <ul class="sidebar-menu">
                <li><a href="#" class="active">Web UI</a></li>
                <li><a href="#">Logs</a></li>
                <li><a href="#">About</a></li>
                <li><a href="#">Settings</a></li>
            </ul>
        </div>
        
        <div class="main-content">
            <div class="header">
                <h1>Visual Understanding Chat Assistant</h1>
                <p>Upload a video to analyze events and engage in intelligent conversations about the content</p>
            </div>
            
            <div class="content-grid">
                <div class="upload-panel">
                    <h3>Upload your video!</h3>
                    <p>Choose your video file</p>
                    
                    <div class="file-upload-area" id="fileUploadArea">
                        <div class="upload-icon">📹</div>
                        <div class="upload-text">Drag and drop file here</div>
                        <div class="upload-subtext">Limit 200MB per file • MP4, WebM, AVI</div>
                        <input type="file" id="videoFile" accept="video/*" style="display: none;">
                    </div>
                    
                    <button class="btn-primary" onclick="document.getElementById('videoFile').click()">Browse files</button>
                    <button class="btn-primary" id="analyzeBtn" onclick="uploadVideo()" style="margin-top: 12px; display: none;">Analyze Video</button>
                    
                    <div id="uploadStatus"></div>
                </div>
                
                <div class="results-panel" id="resultsPanel">
                    <h3>Video Analysis Results</h3>
                    <div id="analysisResults"></div>
                </div>
            </div>
            
            <div class="chat-panel" id="chatPanel">
                <h3>Chat About Your Video</h3>
                <div class="chat-container" id="chatContainer"></div>
                <div class="chat-input">
                    <input type="text" id="messageInput" placeholder="Ask questions about your video..." onkeypress="handleKeyPress(event)">
                    <button class="btn-primary" onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>

        <script>
            let currentSessionId = null;
            let analysisData = null;

            // File upload handling
            document.getElementById('videoFile').addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    document.getElementById('analyzeBtn').style.display = 'block';
                    document.getElementById('uploadStatus').innerHTML = 
                        `<div class="status-message status-success">Selected: ${file.name}</div>`;
                }
            });

            // Drag and drop handling
            const uploadArea = document.getElementById('fileUploadArea');
            
            uploadArea.addEventListener('dragover', function(e) {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', function(e) {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', function(e) {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    const fileInput = document.getElementById('videoFile');
                    const dt = new DataTransfer();
                    dt.items.add(files[0]);
                    fileInput.files = dt.files;
                    
                    // Trigger the change event manually
                    const changeEvent = new Event('change', { bubbles: true });
                    fileInput.dispatchEvent(changeEvent);
                }
            });

            async function uploadVideo() {
                const fileInput = document.getElementById('videoFile');
                const file = fileInput.files[0];
                
                if (!file) {
                    document.getElementById('uploadStatus').innerHTML = 
                        '<div class="status-message status-error">Please select a video file</div>';
                    return;
                }
                
                const statusDiv = document.getElementById('uploadStatus');
                statusDiv.innerHTML = '<div class="status-message status-loading">Analyzing video... This may take a few minutes.</div>';
                
                // Disable button during upload
                document.getElementById('analyzeBtn').disabled = true;
                
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
                        document.getElementById('resultsPanel').style.display = 'block';
                        document.getElementById('chatPanel').style.display = 'block';
                        statusDiv.innerHTML = '<div class="status-message status-success">Analysis complete! You can now chat about your video.</div>';
                    } else {
                        statusDiv.innerHTML = '<div class="status-message status-error">Error: ' + result.detail + '</div>';
                    }
                } catch (error) {
                    statusDiv.innerHTML = '<div class="status-message status-error">Upload failed: ' + error.message + '</div>';
                } finally {
                    document.getElementById('analyzeBtn').disabled = false;
                }
            }
            
            function displayAnalysisResults(analysis) {
                const resultsDiv = document.getElementById('analysisResults');
                let html = '';
                
                // Summary card
                if (analysis.summary) {
                    html += `
                        <div class="summary-card">
                            <div class="summary-title">Summary</div>
                            <div class="summary-text">${analysis.summary}</div>
                        </div>
                    `;
                }
                
                // Events header with count
                const eventCount = analysis.events && analysis.events.length > 0 ? analysis.events.length : 0;
                html += `
                    <div class="events-header">
                        <h4 style="margin: 0; color: #374151; font-size: 16px;">Detected Events</h4>
                        <span class="events-count">${eventCount}</span>
                    </div>
                `;
                
                // Events container
                html += '<div class="events-container">';
                
                if (analysis.events && analysis.events.length > 0) {
                    analysis.events.forEach((event, index) => {
                        html += `
                            <div class="event-card">
                                <div class="event-title">${event.name}</div>
                                <div class="event-time">${event.start} - ${event.end}</div>
                                <div class="event-description">${event.description || 'No description available'}</div>
                            </div>
                        `;
                    });
                } else {
                    html += '<div class="event-card"><div class="event-description">No specific events detected in this video.</div></div>';
                }
                
                html += '</div>'; // Close events-container
                
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
                
                const messageContent = document.createElement('div');
                messageContent.className = 'message-content';
                messageContent.textContent = message;
                
                messageDiv.appendChild(messageContent);
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
