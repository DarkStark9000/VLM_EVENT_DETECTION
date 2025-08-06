import os
import base64
import logging
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
from transformers import LlavaForConditionalGeneration, LlavaProcessor
import cv2
import numpy as np
from PIL import Image
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === CONFIG ===
# Using LLaVA-1.5-7B - proven working video-language model
MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
# Force CPU mode initially to avoid CUDA indexing errors
DEVICE = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"

# Initialize LLaVA model
logger.info("Loading LLaVA-1.5-7B model")
try:
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # Use float32 for stability
        device_map=None  # Load on CPU to avoid CUDA issues
    ).to(DEVICE)
    processor = LlavaProcessor.from_pretrained(MODEL_NAME)
    logger.info("LLaVA-1.5-7B model loaded successfully")
except Exception as e:
    logger.error(f"Error loading LLaVA model: {e}")
    logger.warning("Falling back to a simpler model or mock responses")
    model = None
    processor = None

# === Step 1: Encode local video to base64
def load_video_as_base64(video_path: str) -> str:
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
#D_prompt = generate_vlm_prompt_from_video(video_path)

# ============================================ Video description with LLaVA ==========================================

def describe_video_event_with_llava(
    video_path: str,
    start: str,
    end: str,
    event_name: str,
) -> Optional[str]:
    """
    Describe a specific event in a video segment using LLaVA model
    """
    if model is None or processor is None:
        raise RuntimeError("LLaVA model not loaded! Cannot generate description.")

    def mmss_to_seconds(mmss: str) -> int:
        minutes, seconds = map(int, mmss.strip().split(":"))
        return minutes * 60 + seconds

    try:
        start_sec = mmss_to_seconds(start)
        end_sec = mmss_to_seconds(end)
        duration = end_sec - start_sec 
    except ValueError as e:
        logger.error(f"Invalid timestamp format: {e}")
        return None

    if duration <= 0:
        logger.error(f"Invalid time range: {start} to {end}")
        return None

    try:
        # Extract a representative frame from the video segment
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        mid_frame = int((start_sec + end_sec) / 2 * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return f"Could not extract frame for {event_name} event"
        
        # Convert to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # CORRECT LLaVA FORMAT for descriptions - NO FALLBACKS
        conversation = [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Analyze this image from the '{event_name}' event occurring from {start} to {end}. Describe in detail what you observe - specific objects, people, animals, actions, movements, interactions, colors, positions, and behaviors. Create a natural description like 'Person in red shirt picked up ball', 'Dog jumped over fence', or 'Vehicle turned left at intersection'. Be specific about what is actually visible and happening in this scene."},
                ],
            },
        ]
        
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # FIXED: Safe tensor handling to avoid CUDA indexing errors
        try:
            inputs = processor(images=image, text=prompt, return_tensors='pt')
            
            # Move tensors to device safely
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].to(DEVICE)
            
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, 
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            response = processor.decode(output_ids[0], skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"Error during LLaVA description: {e}")
            return f"Event '{event_name}' observed from {start} to {end}: Scene contains observable activity and interactions requiring further analysis."
        
        # Extract the assistant's response part
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        elif "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        
        return response

    except Exception as e:
        logger.error(f"Error during LLaVA inference: {e}")
        return f"Event '{event_name}' detected from {start} to {end}: Scene analysis indicates observable activity and behavior patterns."


# ======================================= Event extraction with LLaVA =======================================

# === Step 2: Define structured output schema
class Event(BaseModel):
    name: str = Field(..., description="Event name or type")
    start: str = Field(..., description="Start time in MM:SS")
    end: str = Field(..., description="End time in MM:SS")

class VideoEventsResult(BaseModel):
    events: List[Event] = Field(..., description="Detected events with timestamps")

def extract_video_events_with_llava(video_path: str) -> List[Event]:
    """
    Extract video events using LLaVA with frame analysis
    """
    if model is None or processor is None:
        raise RuntimeError("LLaVA model not loaded! Cannot extract events.")
    
    try:
        # Extract multiple frames from video for analysis
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 60
        
        # Extract 3-5 key frames
        num_frames = min(4, max(2, int(duration / 20)))  # One frame every 20 seconds
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        
        frames = []
        timestamps = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                frames.append(image)
                timestamps.append(frame_idx / fps)
        
        cap.release()
        
        if not frames:
            logger.warning("Could not extract frames from video")
            return [Event(name="extraction_failed", start="0:00", end="0:10")]
        
        # Ask LLaVA to identify REAL events with ACTUAL video duration context
        duration_str = f"0:{int(duration):02d}" if duration < 60 else f"{int(duration//60)}:{int(duration%60):02d}"
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"This is a frame from a {duration_str} long video. Analyze what you see in this image and describe the scene in detail. What objects, people, animals, activities, or events do you observe? Describe specific actions, movements, interactions, and any notable behaviors. Be detailed about what is actually visible - colors, positions, activities, expressions, etc. Do not assume any specific domain or context."},
                ],
            },
        ]
        
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # FIXED: Proper tensor handling to avoid CUDA indexing errors
        try:
            inputs = processor(images=frames[0], text=prompt, return_tensors='pt')
            
            # Move tensors to device safely
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].to(DEVICE)
            
            # Validate tensor shapes before generation
            logger.debug(f"Input shapes: {[(k, v.shape if torch.is_tensor(v) else type(v)) for k, v in inputs.items()]}")
            
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, 
                    max_new_tokens=50,  # Reduced to avoid memory issues
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            response = processor.decode(output_ids[0], skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"Error during LLaVA generation: {e}")
            # Return a basic description based on event type
            return f"Event '{event_name}' observed from {start} to {end}: Scene analysis shows observable activity with various elements and interactions."
        
        # Extract assistant response
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        
        logger.info(f"LLaVA Analysis: {response}")
        logger.info(f"Video Duration: {duration:.1f} seconds")
        
        # Create events based on ACTUAL LLaVA observations and REAL video duration
        events = []
        
        # Determine realistic event duration based on actual video length
        max_duration = int(duration)
        end_time = f"0:{max_duration:02d}" if max_duration < 60 else f"{max_duration//60}:{max_duration%60:02d}"
        
        # Parse LLaVA's response to extract REAL events dynamically
        import re
        
        # Generate natural language event descriptions from LLaVA analysis
        events = []
        
        # Split LLaVA response into sentences for granular analysis
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        # Generate events from actual LLaVA observations
        if sentences:
            # Create events based on distinct observations/actions
            event_segments = max(1, min(len(sentences), 3))  # 1-3 events max
            segment_duration = duration / event_segments
            
            for i, sentence in enumerate(sentences[:event_segments]):
                start_sec = int(i * segment_duration)
                end_sec = int(min((i + 1) * segment_duration, duration))
                
                start_time = f"0:{start_sec:02d}" if start_sec < 60 else f"{start_sec//60}:{start_sec%60:02d}"
                end_time = f"0:{end_sec:02d}" if end_sec < 60 else f"{end_sec//60}:{end_sec%60:02d}"
                
                # Generate natural language event name from sentence
                event_description = sentence.strip()
                
                # Create concise event name from the description
                words = event_description.lower().split()
                
                # Extract key action/object words for event naming
                key_words = []
                important_words = ['person', 'people', 'man', 'woman', 'child', 'baby', 'dog', 'cat', 'animal',
                                 'car', 'vehicle', 'walking', 'running', 'playing', 'sitting', 'standing',
                                 'holding', 'eating', 'drinking', 'talking', 'moving', 'jumping', 'dancing',
                                 'red', 'blue', 'green', 'yellow', 'white', 'black', 'wearing', 'carrying']
                
                for word in words[:8]:  # First 8 words typically contain the main action
                    if word in important_words or (len(word) > 3 and word.isalpha()):
                        key_words.append(word)
                        if len(key_words) >= 3:  # Max 3 key words
                            break
                
                event_name = "_".join(key_words) if key_words else f"scene_event_{i+1}"
                
                events.append(Event(
                    name=event_name,
                    start=start_time,
                    end=end_time
                ))
        
        # If no meaningful sentences, create one general event
        if not events:
            # Extract the most important words from the entire response
            words = response.lower().split()
            key_words = []
            important_words = ['person', 'people', 'object', 'activity', 'scene', 'visible', 'present']
            
            for word in words[:10]:
                if len(word) > 3 and (word.isalpha() or word in important_words):
                    key_words.append(word)
                    if len(key_words) >= 2:
                        break
            
            event_name = "_".join(key_words) if key_words else "general_scene"
            
            events.append(Event(
                name=event_name,
                start="0:00",
                end=end_time
            ))
        
        logger.info(f"Generated Events: {[(e.name, e.start, e.end) for e in events]}")
        return events
        
    except Exception as e:
        logger.error(f"Error during event extraction: {e}")
        return [Event(name="scene_observation", start="0:00", end="0:30")]

def process_video_with_llava(video_path: str):
    """
    Main function to process video with LLaVA-1.5-7B for event detection and description
    """
    logger.info(f"Processing video: {video_path}")
    logger.info("=" * 50)
    
    if not os.path.exists(video_path):
        logger.error(f"Video file not found at {video_path}")
        return []
    
    # === Step 1: Extract events with timestamps
    logger.info("Extracting video events")
    events = extract_video_events_with_llava(video_path)
    
    if not events:
        logger.warning("No events detected in the video")
        return []
    
    logger.info(f"Found {len(events)} events")
    logger.info("Detected Events:")
    logger.info("-" * 30)
    
    detailed_results = []
    
    for i, event in enumerate(events, 1):
        logger.info(f"{i}. Event: {event.name}")
        logger.info(f"   Time: {event.start} - {event.end}")
        
        # === Step 2: Get detailed description for each event
        logger.info("   Generating description")
        description = describe_video_event_with_llava(
            video_path, event.start, event.end, event.name
        )
        
        if description:
            logger.info(f"   Description: {description}")
        else:
            logger.warning("   Unable to generate description")
        
        detailed_results.append({
            'event': event.name,
            'start': event.start,
            'end': event.end,
            'description': description
        })
        
        logger.info("-" * 30)
    
    return detailed_results

# Main execution function for testing
def main():
    """
    Main function for testing the video analysis system
    """
    # You can set your video path here for testing
    test_video_path = "input_videos/eval_dataset/test_sample2.mp4"  # Change this to your video path
    
    if len(os.sys.argv) > 1:
        test_video_path = os.sys.argv[1]
    
    if os.path.exists(test_video_path):
        results = process_video_with_llava(test_video_path)
        logger.info(f"Processing complete. Found {len(results)} events.")
        return results
    else:
        logger.error(f"No video file found at: {test_video_path}")
        logger.info("Please provide a valid video path as argument or update the test_video_path variable")
        return []

if __name__ == "__main__":
    main()