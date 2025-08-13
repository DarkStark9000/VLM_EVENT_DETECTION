"""
Video Processing Module for Event Detection and Analysis
Handles video upload, frame extraction, and VLM-based event recognition using industrial OOP patterns
"""

import base64
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Protocol
from dataclasses import dataclass
import io
import av
from openai import OpenAI
from pydantic import BaseModel, Field


@dataclass
class VideoProcessingConfig:
    """Configuration for video processing operations"""
    vlm_base_url: str = "http://localhost:8000/v1"
    vlm_api_key: str = "dummy"
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    max_frames: int = 2
    image_width: int = 320
    image_height: int = 240
    image_quality: int = 60
    max_tokens: int = 400
    temperature: float = 0.3


class Event(BaseModel):
    """Data model for video events"""
    name: str = Field(..., description="Event name or type")
    start: str = Field(..., description="Start time in MM:SS")
    end: str = Field(..., description="End time in MM:SS")
    description: Optional[str] = Field(None, description="Detailed description of the event")


class VideoInfo(BaseModel):
    """Data model for video information"""
    duration: str = Field(..., description="Video duration in MM:SS format")
    duration_seconds: float = Field(..., description="Duration in seconds")
    fps: float = Field(..., description="Frames per second")
    width: int = Field(..., description="Video width")
    height: int = Field(..., description="Video height")


class VideoAnalysisResult(BaseModel):
    """Data model for complete video analysis results"""
    events: List[Event] = Field(default=[], description="Detected events with timestamps")
    summary: str = Field(..., description="Overall video summary")
    duration: str = Field(..., description="Video duration")
    guidelines_violations: List[str] = Field(default=[], description="Detected guideline violations")
    video_info: Optional[VideoInfo] = Field(None, description="Technical video information")


class FrameExtractor(Protocol):
    """Protocol for frame extraction operations"""
    
    def extract_frames(self, video_path: str, max_frames: int) -> List[str]:
        """Extract frames from video as base64 strings"""
        ...


class VideoAnalyzer(Protocol):
    """Protocol for video analysis operations"""
    
    async def analyze_content(self, video_path: str) -> Dict:
        """Analyze video content and return structured results"""
        ...


class BaseVideoProcessor(ABC):
    """Abstract base class for video processors"""
    
    @abstractmethod
    async def analyze_video(self, video_path: str) -> Dict:
        """Analyze video and return results"""
        pass
    
    @abstractmethod
    def extract_video_info(self, video_path: str) -> VideoInfo:
        """Extract basic video information"""
        pass


class VideoMetadataExtractor:
    """Service for extracting video metadata and information"""
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
    
    def extract_info(self, video_path: str) -> VideoInfo:
        """Extract comprehensive video information"""
        try:
            container = av.open(video_path)
            video_stream = container.streams.video[0]
            duration_seconds = float(container.duration / av.time_base)
            
            # Format duration as MM:SS
            minutes = int(duration_seconds // 60)
            seconds = int(duration_seconds % 60)
            duration_formatted = f"{minutes:02d}:{seconds:02d}"
            
            video_info = VideoInfo(
                duration=duration_formatted,
                duration_seconds=duration_seconds,
                fps=float(video_stream.average_rate),
                width=video_stream.width,
                height=video_stream.height
            )
            
            container.close()
            return video_info
            
        except Exception as e:
            self._logger.error(f"Error extracting video info: {e}")
            return VideoInfo(
                duration="00:00",
                duration_seconds=0,
                fps=0,
                width=0,
                height=0
            )


class VideoFrameExtractor:
    """Service for extracting frames from video files"""
    
    def __init__(self, config: VideoProcessingConfig):
        self._config = config
        self._logger = logging.getLogger(__name__)
    
    def extract_frames(self, video_path: str, max_frames: Optional[int] = None) -> List[str]:
        """Extract frames from video as base64 encoded strings"""
        max_frames = max_frames or self._config.max_frames
        
        try:
            container = av.open(video_path)
            video_stream = container.streams.video[0]
            
            total_frames = self._estimate_total_frames(container, video_stream)
            interval = max(1, total_frames // max_frames)
            
            frames_b64 = []
            frame_count = 0
            
            for frame in container.decode(video_stream):
                if frame_count % interval == 0:
                    frame_b64 = self._process_frame(frame)
                    if frame_b64:
                        frames_b64.append(frame_b64)
                    
                    if len(frames_b64) >= max_frames:
                        break
                
                frame_count += 1
            
            container.close()
            return frames_b64
            
        except Exception as e:
            self._logger.error(f"Error extracting frames: {e}")
            return []
    
    def _estimate_total_frames(self, container, video_stream) -> int:
        """Estimate total number of frames in video"""
        total_frames = video_stream.frames
        if total_frames == 0:
            duration = float(container.duration / av.time_base)
            fps = float(video_stream.average_rate)
            total_frames = int(duration * fps)
        return total_frames
    
    def _process_frame(self, frame) -> Optional[str]:
        """Process individual frame and convert to base64"""
        try:
            img = frame.to_image()
            img = img.resize((self._config.image_width, self._config.image_height))
            
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self._config.image_quality)
            return base64.b64encode(buf.getvalue()).decode('utf-8')
            
        except Exception as e:
            self._logger.error(f"Error processing frame: {e}")
            return None


class VLMAnalysisService:
    """Service for VLM-based video content analysis"""
    
    def __init__(self, config: VideoProcessingConfig):
        self._config = config
        self._logger = logging.getLogger(__name__)
        
        try:
            self._client = OpenAI(
                base_url=config.vlm_base_url,
                api_key=config.vlm_api_key
            )
            self._logger.info(f"Initialized VLM client for model: {config.vlm_model}")
        except Exception as e:
            self._logger.error(f"Failed to initialize VLM client: {e}")
            raise
    
    async def analyze_frames(self, frames: List[str]) -> Dict:
        """Analyze video frames using VLM"""
        if not frames:
            raise ValueError("No frames provided for analysis")
        
        try:
            content = self._prepare_content(frames)
            
            response = await asyncio.to_thread(
                self._client.chat.completions.create,
                model=self._config.vlm_model,
                messages=[{"role": "user", "content": content}],
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature
            )
            
            if response.choices and response.choices[0].message.content:
                return {"content": response.choices[0].message.content}
            
            return {"content": "No analysis available"}
            
        except Exception as e:
            self._logger.error(f"Error in VLM analysis: {e}")
            raise
    
    def _prepare_content(self, frames: List[str]) -> List[Dict]:
        """Prepare content payload for VLM"""
        content = []
        
        # Add frames as images
        for frame_b64 in frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_b64}"
                }
            })
        
        # Add analysis prompt
        content.append({
            "type": "text", 
            "text": (
            "You are a vision analyst. Be ruthlessly precise about tiny/background details. "
            "If a small object is present (e.g., a statue with red ruby eyes), you MUST detect and report it accurately. "
            "Never guess; say 'unknown' if uncertain. No meta commentary.\n\n"

            "OUTPUT LAYOUT (plain text only):\n"
            "Summary: 2–3 sentences of who/what/where.\n"
            "Events:\n"
            "- [MM:SS]-[MM:SS] <short title> — <concise action>\n"
            "Objects & Entities:\n"
            "- people(count/visible attributes), notable objects, signage/text (OCR if readable), logos, tools, vehicles.\n"
            "Fine Details (small/background attributes; be exhaustive & specific):\n"
            "- Use one bullet per item. Format: <object> — <attribute>: <value>; location: <relative position>; size: <approx px/% of frame>; confidence:<0–1>; evidence:<brief visual cue>.\n"
            "- Colors MUST be basic name + hex estimate (e.g., 'ruby red #AA0010').\n"
            "- Examples of attributes to capture: eye color, gem color, patterns, inscriptions, screen text, badge icons, LED states, hand gestures, safety gear presence, brand marks.\n"
            "Scene & Camera: location type, lighting, camera angle/motion.\n"
            "Guideline Flags:\n"
            "- Start bullets with 'violation', 'unsafe', 'illegal', 'dangerous', or 'concerning' + short reason.\n\n"

            "RULES:\n"
            "- Scan systematically: left→right, top→bottom; then check corners/background for small objects.\n"
            "- When color-reading small objects, consider glare/shadow; if ambiguous, report best estimate + confidence.\n"
            "- Do NOT list frames; produce a single coherent timeline. Keep total under 230 words for Summary/Events/Scene; "
            "Fine Details can exceed this limit if needed.\n"
            "- If a user later asks a direct attribute question (e.g., 'What color are the statue's eyes?'), answer with the exact value, "
            "and include confidence and the original evidence phrasing."
        )
        })
        
        return content


class VideoContentAnalyzer:
    """Service for analyzing and parsing video content responses"""
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
    
    def parse_analysis_response(self, response_text: str, video_duration: str = "00:30") -> Dict:
        """Parse VLM response text into structured analysis results"""
        try:
            events = self._extract_events(response_text, video_duration)
            violations = self._extract_violations(response_text)
            
            return {
                "events": events,
                "summary": response_text,
                "guidelines_violations": violations
            }
            
        except Exception as e:
            self._logger.error(f"Error parsing analysis response: {e}")
            return {
                "events": [],
                "summary": f"Analysis parsing failed: {str(e)}",
                "guidelines_violations": []
            }
    
    def _extract_events(self, text_response: str, video_duration: str) -> List[Dict]:
        """Extract events from response text using pattern matching"""
        import re
        
        events = []
        time_pattern = r'(\d{1,2}):(\d{2})'
        lines = text_response.split('\n')
        
        action_words = [
            'shows', 'appears', 'visible', 'seen', 'moving', 'standing', 'sitting', 'walking',
            'talking', 'holding', 'wearing', 'looking', 'coming', 'going', 'entering', 'leaving',
            'happens', 'occurring', 'taking', 'doing', 'being', 'having', 'getting', 'making'
        ]
        
        for line in lines:
            line = line.strip()
            if not line or len(line.split()) <= 5:
                continue
            
            has_content = any(word in line.lower() for word in action_words)
            if has_content:
                time_matches = re.findall(time_pattern, line)
                
                if time_matches:
                    start_time = f"{time_matches[0][0].zfill(2)}:{time_matches[0][1]}"
                    end_time = start_time if len(time_matches) == 1 else f"{time_matches[1][0].zfill(2)}:{time_matches[1][1]}"
                else:
                    start_time = "00:00"
                    end_time = video_duration
                
                event_name = self._extract_event_name(line)
                
                events.append({
                    "name": event_name,
                    "start": start_time,
                    "end": end_time,
                    "description": line[:120]
                })
        
        # Ensure at least one event exists
        if not events and text_response:
            events.append({
                "name": "Video Content",
                "start": "00:00",
                "end": video_duration,
                "description": text_response[:150]
            })
        
        return events
    
    def _extract_violations(self, text_response: str) -> List[str]:
        """Extract guideline violations from response text"""
        violations = []
        concern_keywords = [
            'violation', 'illegal', 'wrong', 'unsafe', 'dangerous', 'concerning',
            'inappropriate', 'problem', 'issue', 'risk', 'hazard', 'warning',
            'accident', 'injury', 'damage', 'conflict', 'fight', 'aggressive'
        ]
        
        lines = text_response.split('\n')
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in concern_keywords):
                violations.append(line[:100])
        
        return violations
    
    def _extract_event_name(self, line: str) -> str:
        """Extract appropriate event name from content description"""
        line_lower = line.lower()
        words = line_lower.split()
        
        # Key subjects and actions for event naming
        important_nouns = [
            'person', 'people', 'man', 'woman', 'child', 'group', 'character', 'actor',
            'car', 'vehicle', 'bus', 'train', 'bike', 'motorcycle', 'truck',
            'building', 'house', 'room', 'scene', 'camera', 'view',
            'conversation', 'dialogue', 'meeting', 'interview', 'performance',
            'action', 'movement', 'activity', 'event', 'incident',
            'music', 'song', 'dance', 'game', 'sport', 'match'
        ]
        
        key_actions = [
            'walking', 'running', 'sitting', 'standing', 'talking', 'speaking',
            'driving', 'moving', 'dancing', 'singing', 'playing', 'working',
            'eating', 'drinking', 'reading', 'writing', 'watching', 'looking',
            'entering', 'leaving', 'arriving', 'departing', 'meeting', 'greeting'
        ]
        
        subjects = [noun for noun in important_nouns if noun in words]
        actions = [action for action in key_actions if action in words]
        
        if subjects and actions:
            return f"{subjects[0].title()} {actions[0].title()}"
        elif subjects:
            return f"{subjects[0].title()} Activity"
        elif actions:
            return f"{actions[0].title()} Scene"
        else:
            meaningful_words = [
                w for w in words 
                if len(w) > 3 and w not in ['this', 'that', 'with', 'from', 'they', 'there', 'appear', 'shows', 'video']
            ]
            if meaningful_words:
                return f"{meaningful_words[0].title()} Scene"
            else:
                return "Video Scene"


class VideoProcessor(BaseVideoProcessor):
    """Main video processor implementing industrial OOP patterns"""
    
    def __init__(self, config: Optional[VideoProcessingConfig] = None):
        self._config = config or VideoProcessingConfig()
        self._logger = logging.getLogger(__name__)
        
        # Initialize services using dependency injection
        self._metadata_extractor = VideoMetadataExtractor()
        self._frame_extractor = VideoFrameExtractor(self._config)
        self._vlm_service = VLMAnalysisService(self._config)
        self._content_analyzer = VideoContentAnalyzer()
        
        self._logger.info("VideoProcessor initialized with industrial OOP structure")
        


    def extract_video_info(self, video_path: str) -> Dict:
        """Extract basic video information - legacy method for backward compatibility"""
        try:
            video_info = self._metadata_extractor.extract_info(video_path)
            return video_info.dict()
        except Exception as e:
            self._logger.error(f"Error in extract_video_info: {e}")
            return {"duration": "00:00", "duration_seconds": 0}
    
    def extract_video_frames(self, video_path: str, max_frames: int = 3) -> List[str]:
        """Extract frames from video - legacy method for backward compatibility"""
        return self._frame_extractor.extract_frames(video_path, max_frames)
    
    async def analyze_video_content(self, video_path: str) -> Dict:
        """Analyze video content using VLM - legacy method for backward compatibility"""
        try:
            # Extract frames
            frames = self._frame_extractor.extract_frames(video_path)
            if not frames:
                raise ValueError("Could not extract frames from video")
            
            # Analyze with VLM
            vlm_response = await self._vlm_service.analyze_frames(frames)
            
            # Parse response
            video_info = self._metadata_extractor.extract_info(video_path)
            analysis_result = self._content_analyzer.parse_analysis_response(
                vlm_response.get("content", ""),
                video_info.duration
            )
            
            return analysis_result
            
        except Exception as e:
            self._logger.error(f"Error in video analysis: {e}")
            return {
                "events": [],
                "summary": f"Analysis failed: {str(e)}",
                "guidelines_violations": []
            }
    
    async def analyze_video(self, video_path: str) -> Dict:
        """Main video analysis function - maintains exact same interface"""
        try:
            self._logger.info(f"Starting analysis of video: {video_path}")
            
            # Extract video information
            video_info = self._metadata_extractor.extract_info(video_path)
            self._logger.info(f"Video duration: {video_info.duration}")
            
            # Analyze content
            analysis_result = await self.analyze_video_content(video_path)
            
            # Combine results - maintaining exact same output format
            result = {
                "duration": video_info.duration,
                "events": analysis_result.get("events", []),
                "summary": analysis_result.get("summary", "No summary available"),
                "guidelines_violations": analysis_result.get("guidelines_violations", []),
                "video_info": video_info.dict()
            }
            
            self._logger.info(f"Analysis complete. Found {len(result['events'])} events")
            return result
            
        except Exception as e:
            self._logger.error(f"Video analysis failed: {e}")
            return {
                "duration": "00:00",
                "events": [],
                "summary": f"Analysis failed: {str(e)}",
                "guidelines_violations": [],
                "error": str(e)
            }
    
    def analyze_video_segment(self, video_path: str, start_time: str, end_time: str) -> Optional[str]:
        """Analyze a specific segment of the video - legacy method maintained"""
        try:
            def time_to_seconds(time_str: str) -> int:
                parts = time_str.split(':')
                return int(parts[0]) * 60 + int(parts[1])
            
            start_sec = time_to_seconds(start_time)
            end_sec = time_to_seconds(end_time)
            
            # Extract frames from specific segment using existing logic
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
                
                if len(frames) >= 10:
                    break
            
            container.close()
            
            if not frames:
                return "No frames found in specified time range"
            
            # Analyze segment with VLM using the same client
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
            
            response = self._vlm_service._client.chat.completions.create(
                model=self._config.vlm_model,
                messages=[{"role": "user", "content": content}],
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self._logger.error(f"Segment analysis failed: {e}")
            return f"Segment analysis failed: {str(e)}"

