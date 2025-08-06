"""
Video Processing Module for Event Detection and Analysis
Handles video upload, frame extraction, and VLM-based event recognition
"""

import os
import base64
import json
import asyncio
from typing import List, Dict, Optional
import tempfile
import io
import av
from openai import OpenAI
from pydantic import BaseModel, Field

class Event(BaseModel):
    name: str = Field(..., description="Event name or type")
    start: str = Field(..., description="Start time in MM:SS")
    end: str = Field(..., description="End time in MM:SS")
    description: Optional[str] = Field(None, description="Detailed description of the event")

class VideoAnalysisResult(BaseModel):
    events: List[Event] = Field(default=[], description="Detected events with timestamps")
    summary: str = Field(..., description="Overall video summary")
    duration: str = Field(..., description="Video duration")
    guidelines_violations: List[str] = Field(default=[], description="Detected guideline violations")

class VideoProcessor:
    def __init__(self):
        # VLM client for video analysis
        self.vlm_client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="dummy"
        )
        self.vlm_model = "Qwen/Qwen2.5-VL-7B-Instruct"
        
        # Tools for structured output
        self.event_extraction_tools = [
            {
                "type": "function",
                "function": {
                    "name": "extract_video_events",
                    "description": "Detects events in a video and provides their timestamps and descriptions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string", "description": "Name or type of the event"},
                                        "start": {"type": "string", "description": "Start time in MM:SS format"},
                                        "end": {"type": "string", "description": "End time in MM:SS format"},
                                        "description": {"type": "string", "description": "Detailed description of what happens"}
                                    },
                                    "required": ["name", "start", "end", "description"]
                                }
                            },
                            "summary": {"type": "string", "description": "Overall summary of the video content"},
                            "guidelines_violations": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of detected guideline violations or safety issues"
                            }
                        },
                        "required": ["events", "summary", "guidelines_violations"]
                    }
                }
            }
        ]

    def extract_video_info(self, video_path: str) -> Dict:
        """Extract basic video information"""
        try:
            container = av.open(video_path)
            video_stream = container.streams.video[0]
            duration_seconds = float(container.duration / av.time_base)
            
            # Format duration as MM:SS
            minutes = int(duration_seconds // 60)
            seconds = int(duration_seconds % 60)
            duration_formatted = f"{minutes:02d}:{seconds:02d}"
            
            return {
                "duration": duration_formatted,
                "duration_seconds": duration_seconds,
                "fps": float(video_stream.average_rate),
                "width": video_stream.width,
                "height": video_stream.height
            }
        except Exception as e:
            print(f"Error extracting video info: {e}")
            return {"duration": "00:00", "duration_seconds": 0}

    def extract_video_frames(self, video_path: str, max_frames: int = 3) -> List[str]:
        """Extract frames from video for analysis"""
        try:
            container = av.open(video_path)
            video_stream = container.streams.video[0]
            
            total_frames = video_stream.frames
            if total_frames == 0:
                # Estimate frames if not available
                duration = float(container.duration / av.time_base)
                fps = float(video_stream.average_rate)
                total_frames = int(duration * fps)
            
            # Calculate frame sampling interval
            interval = max(1, total_frames // max_frames)
            
            frames_b64 = []
            frame_count = 0
            
            for frame in container.decode(video_stream):
                if frame_count % interval == 0:
                    # Convert frame to image
                    img = frame.to_image()
                    
                    # Resize and convert to base64
                    img = img.resize((320, 240))  # Reduce image size
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=60)  # Lower quality
                    b64_string = base64.b64encode(buf.getvalue()).decode('utf-8')
                    frames_b64.append(b64_string)
                    
                    if len(frames_b64) >= max_frames:
                        break
                
                frame_count += 1
            
            container.close()
            return frames_b64
            
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return []

    async def analyze_video_content(self, video_path: str) -> Dict:
        """Analyze video content using VLM"""
        try:
            # Extract frames
            frames = self.extract_video_frames(video_path, max_frames=2)
            if not frames:
                raise Exception("Could not extract frames from video")
            
            # Prepare content for VLM
            content = []
            
            # Add frames as images
            for i, frame_b64 in enumerate(frames):
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_b64}"
                    }
                })
            
            # Add analysis prompt
            content.append({
                "type": "text", 
                "text": "Describe this video in detail. What do you see? Who are the people/characters? What are they doing? What objects, locations, or activities are present? What happens in the sequence? Be specific and descriptive."
            })
            
            # Call VLM with simple text response
            response = await asyncio.to_thread(
                self.vlm_client.chat.completions.create,
                model=self.vlm_model,
                messages=[{"role": "user", "content": content}],
                max_tokens=400,
                temperature=0.3
            )
            
            # Parse text response
            if response.choices and response.choices[0].message.content:
                content_text = response.choices[0].message.content
                # Get video duration for context
                video_info = self.extract_video_info(video_path)
                return self._parse_text_response(content_text, video_info.get("duration", "00:30"))
            
            # Fallback
            return {
                "events": [],
                "summary": "Could not analyze video content",
                "guidelines_violations": []
            }
            
        except Exception as e:
            print(f"Error in video analysis: {e}")
            return {
                "events": [],
                "summary": f"Analysis failed: {str(e)}",
                "guidelines_violations": []
            }

    async def analyze_video(self, video_path: str) -> Dict:
        """Main video analysis function"""
        try:
            print(f"Starting analysis of video: {video_path}")
            
            # Extract video information
            video_info = self.extract_video_info(video_path)
            print(f"Video duration: {video_info['duration']}")
            
            # Analyze content
            analysis_result = await self.analyze_video_content(video_path)
            
            # Combine results
            result = {
                "duration": video_info["duration"],
                "events": analysis_result.get("events", []),
                "summary": analysis_result.get("summary", "No summary available"),
                "guidelines_violations": analysis_result.get("guidelines_violations", []),
                "video_info": video_info
            }
            
            print(f"Analysis complete. Found {len(result['events'])} events")
            return result
            
        except Exception as e:
            print(f"Video analysis failed: {e}")
            return {
                "duration": "00:00",
                "events": [],
                "summary": f"Analysis failed: {str(e)}",
                "guidelines_violations": [],
                "error": str(e)
            }

    def analyze_video_segment(self, video_path: str, start_time: str, end_time: str) -> Optional[str]:
        """Analyze a specific segment of the video"""
        try:
            def time_to_seconds(time_str: str) -> int:
                parts = time_str.split(':')
                return int(parts[0]) * 60 + int(parts[1])
            
            start_sec = time_to_seconds(start_time)
            end_sec = time_to_seconds(end_time)
            
            # Extract frames from specific segment
            container = av.open(video_path)
            video_stream = container.streams.video[0]
            time_base = video_stream.time_base
            
            start_pts = int(start_sec / time_base)
            end_pts = int(end_sec / time_base)
            
            frames = []
            for frame in container.decode(video_stream):
                if frame.pts is None:
                    continue
                if frame.pts < start_pts:
                    continue
                if frame.pts > end_pts:
                    break
                
                img = frame.to_image()
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                b64_string = base64.b64encode(buf.getvalue()).decode('utf-8')
                frames.append(b64_string)
                
                if len(frames) >= 10:  # Limit frames for segment analysis
                    break
            
            container.close()
            
            if not frames:
                return "No frames found in specified time range"
            
            # Analyze segment with VLM
            content = []
            for frame_b64 in frames:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
                })
            
            content.append({
                "type": "text",
                "text": f"Describe what happens in this video segment from {start_time} to {end_time}. Focus on specific actions, movements, and any notable events."
            })
            
            response = self.vlm_client.chat.completions.create(
                model=self.vlm_model,
                messages=[{"role": "user", "content": content}],
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Segment analysis failed: {str(e)}"

    def _extract_event_name_from_content(self, line: str) -> str:
        """Extract appropriate event name from content description"""
        line_lower = line.lower()
        words = line_lower.split()
        
        # Key objects/subjects that could be event names
        subjects = []
        
        # Find main subjects (nouns)
        important_nouns = ['person', 'people', 'man', 'woman', 'child', 'group', 'character', 'actor',
                          'car', 'vehicle', 'bus', 'train', 'bike', 'motorcycle', 'truck',
                          'building', 'house', 'room', 'scene', 'camera', 'view',
                          'conversation', 'dialogue', 'meeting', 'interview', 'performance',
                          'action', 'movement', 'activity', 'event', 'incident',
                          'music', 'song', 'dance', 'game', 'sport', 'match']
        
        # Find actions (verbs)
        key_actions = ['walking', 'running', 'sitting', 'standing', 'talking', 'speaking',
                      'driving', 'moving', 'dancing', 'singing', 'playing', 'working',
                      'eating', 'drinking', 'reading', 'writing', 'watching', 'looking',
                      'entering', 'leaving', 'arriving', 'departing', 'meeting', 'greeting']
        
        # Extract subjects
        for noun in important_nouns:
            if noun in words:
                subjects.append(noun)
        
        # Extract actions
        actions = []
        for action in key_actions:
            if action in words:
                actions.append(action)
        
        # Build event name
        if subjects and actions:
            subject = subjects[0].title()
            action = actions[0].title()
            return f"{subject} {action}"
        elif subjects:
            return f"{subjects[0].title()} Activity"
        elif actions:
            return f"{actions[0].title()} Scene"
        else:
            # Extract first meaningful word
            meaningful_words = [w for w in words if len(w) > 3 and 
                              w not in ['this', 'that', 'with', 'from', 'they', 'there', 'appear', 'shows', 'video']]
            if meaningful_words:
                return f"{meaningful_words[0].title()} Scene"
            else:
                return "Video Scene"

    def _parse_text_response(self, text_response: str, video_duration: str = "00:30") -> Dict:
        """Parse free-form text response to extract structured information"""
        import re
        
        # Extract events using simple pattern matching
        events = []
        violations = []
        
        # Look for time patterns and event descriptions
        time_pattern = r'(\d{1,2}):(\d{2})'
        lines = text_response.split('\n')
        
        current_event = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for time references
            time_matches = re.findall(time_pattern, line)
            
            # Generic content detection - look for any descriptive content
            action_words = ['shows', 'appears', 'visible', 'seen', 'moving', 'standing', 'sitting', 'walking', 
                           'talking', 'holding', 'wearing', 'looking', 'coming', 'going', 'entering', 'leaving',
                           'happens', 'occurring', 'taking', 'doing', 'being', 'having', 'getting', 'making']
            
            # Check for any meaningful content description
            has_content = (len(line.split()) > 5 and 
                          any(word in line.lower() for word in action_words))
            
            if has_content:
                # Extract or estimate time
                if time_matches:
                    start_time = f"{time_matches[0][0].zfill(2)}:{time_matches[0][1]}"
                    end_time = start_time if len(time_matches) == 1 else f"{time_matches[1][0].zfill(2)}:{time_matches[1][1]}"
                else:
                    # Use actual video duration instead of default
                    start_time = "00:00"
                    end_time = video_duration
                    
                # Extract event name from content intelligently
                event_name = self._extract_event_name_from_content(line)
                
                events.append({
                    "name": event_name,
                    "start": start_time,
                    "end": end_time,
                    "description": line[:120]  # Slightly longer description
                })
            
            # Check for concerning content (generic)
            concern_keywords = ['violation', 'illegal', 'wrong', 'unsafe', 'dangerous', 'concerning', 
                              'inappropriate', 'problem', 'issue', 'risk', 'hazard', 'warning',
                              'accident', 'injury', 'damage', 'conflict', 'fight', 'aggressive']
            if any(keyword in line.lower() for keyword in concern_keywords):
                violations.append(line[:100])
        
        # If no specific events found, create a general summary event
        if not events:
            # Look for any time references in the text
            all_times = re.findall(time_pattern, text_response)
            if all_times:
                start_time = f"{all_times[0][0].zfill(2)}:{all_times[0][1]}"
                end_time = f"{all_times[-1][0].zfill(2)}:{all_times[-1][1]}" if len(all_times) > 1 else video_duration
                events.append({
                    "name": "Video Activity",
                    "start": start_time,
                    "end": end_time,
                    "description": "General activity observed in video"
                })
        
        # Always create at least one event if we have any content
        if not events and text_response:
            # Extract any time references or create general event
            events.append({
                "name": "Video Content",
                "start": "00:00", 
                "end": video_duration,
                "description": text_response[:150]
            })
        
        return {
            "events": events,
            "summary": text_response,  # Return full summary
            "guidelines_violations": violations
        }
