import os
import base64
from pydantic import BaseModel, Field
from typing import List, Optional
from openai import OpenAI
from D_prompt import generate_vlm_prompt_from_video

# === CONFIG ===
# vLLM OpenAI-compatible endpoint
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # vLLM doesn't require real API key
)
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
VIDEO_PATH = "/home/natarajan-senguttuvan/Downloads/cyberbad.mp4"

# Ensure the video file exists
if not os.path.exists(VIDEO_PATH):
    print(f"Error: Video file not found at {VIDEO_PATH}")
    exit()

# === Step 1: Encode local video to base64
def load_video_as_base64(video_path: str) -> str:
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
#D_prompt = generate_vlm_prompt_from_video(video_path)

#============================================Video description==========================================================================
import base64
import io
import av
from typing import Optional

def describe_video_event_with_gemini(
    video_path: str,
    start: str,
    end: str,
    event_name: str,
) -> Optional[str]:
    
    # === Configure vLLM ===
    tools = [
        {
            "type": "function",
            "function": {
                "name": "describe_events",
                "description": "Clearly explain and describe what is happening in the video in a brief and concise manner.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "events": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "event_detail": {
                                        "type": "string",
                                        "description": "Describe the event, e.g., a person on a red bike is not wearing a helmet, the bike collides with a woman who is crossing the road. , the bus is running red light"
                                    }
                                },
                                "required": ["event_detail"]
                            }
                        }
                    },
                    "required": ["events"]
                }
            }
        }
    ]

    def mmss_to_seconds(mmss: str) -> int:
        minutes, seconds = map(int, mmss.strip().split(":"))
        return minutes * 60 + seconds

    def extract_frames_with_pyav(video_path: str, start_sec: int, end_sec: int, max_frames: int = 150):
        try:
            container = av.open(video_path)
        except av.AVError as e:
            print(f"Failed to open video with PyAV: {e}")
            return []

        stream = container.streams.video[0]
        time_base = stream.time_base
        start_pts = int(start_sec / time_base)
        end_pts = int(end_sec / time_base)

        selected_frames = []
        count = 0

        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            if frame.pts < start_pts:
                continue
            if frame.pts > end_pts:
                break

            img = frame.to_image()
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            encoded = base64.b64encode(buf.getvalue()).decode('utf-8')

            selected_frames.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded}"
                }
            })

            # count += 1
            # if count >= max_frames:
            #     break

        container.close()
        return selected_frames

    try:
        start_sec = mmss_to_seconds(start)
        end_sec = mmss_to_seconds(end)
        duration = end_sec - start_sec 
    except ValueError as e:
        print(f"Invalid timestamp format: {e}")
        return None

    if duration <= 0:
        print(f"Invalid time range: {start} to {end}")
        return None

    # === Extract Frames ===
    image_parts = extract_frames_with_pyav(video_path, start_sec, end_sec)
    if not image_parts:
        print("Failed to extract frames.")
        return None

    # === Add instruction text ===
    content = image_parts + [{
        "type": "text",
        "text": f"Look at these video clips and describe what is happening. Focus on the event called '{event_name}' see like that pointview ,give the right short description and summary."
    }]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": content}],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "describe_events"}}
        )

        if not response.choices:
            print("No choices found.")
            return None

        # Check for function call response
        message = response.choices[0].message
        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "describe_events":
                    import json
                    args = json.loads(tool_call.function.arguments)
                    events = args.get("events", [])
                    if events:
                        return events[0]["event_detail"]

        print("vLLM did not return a valid function call.")
        return None

    except Exception as e:
        print(f"Error during vLLM request: {e}")
        return None


#=======================================Time stamp collection===============================================================================


# === Step 2: Define structured output schema
class Event(BaseModel):
    name: str = Field(..., description="Event name or type")
    start: str = Field(..., description="Start time in MM:SS")
    end: str = Field(..., description="End time in MM:SS")

class VideoEventsResult(BaseModel):
    events: List[Event] = Field(..., description="Detected events with timestamps")

# === Step 3: Initialize OpenAI function calling tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_video_events",
            "description": "Detects events in a video and gives their timestamps",
            "parameters": {
                "type": "object",
                "properties": {
                    "events": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Name or type of the event, e.g., 'accident', 'person crossing', 'traffic violation'."},
                                "start": {"type": "string", "description": "Start time of the event in MM:SS format."},
                                "end": {"type": "string", "description": "End time of the event in MM:SS format."}
                            },
                            "required": ["name", "start", "end"]
                        }
                    }
                },
                "required": ["events"]
            }
        }
    }
]

# === Step 4: Send prompt and video
video_base64 = load_video_as_base64(VIDEO_PATH)

print("Sending request to vLLM model...")
try:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:video/mp4;base64,{video_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            "Analyze this video and identify all significant events, "
                            "such as accidents,traffic,unsafe pedestrian crossing road, pedestrian accidents and motorcycle riding without helmet violations this are my main event "
                            "For each detected event, provide a Identified event name and strictly adhere to this event names only, capture multiple events if any in that timestamp. "
                            "provide it's precise start and end timestamps in MM:SS format of such events. "
                            "Strictly use the `extract_video_events` function to report the findings."
                        )
                    }
                ]
            }
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "extract_video_events"}}
    )
    print("Received response from vLLM.")

    if not response.choices:
        print("No choices found in the response.")
        exit()

    # === Step 5: Parse the structured function output
    message = response.choices[0].message
    if message.tool_calls:
        for tool_call in message.tool_calls:
            if tool_call.function.name == "extract_video_events":
                import json
                arguments = json.loads(tool_call.function.arguments)
                print(f"Raw arguments received: {arguments}")
                try:
                    result = VideoEventsResult.parse_obj(arguments)
                    # === Step 6: Display results
                    print("\nDetected Events:")
                    if result.events:
                        for event in result.events:
                            print(f"- {event.name} from {event.start} to {event.end}")

                            description = describe_video_event_with_gemini(VIDEO_PATH, event.start, event.end,event.name)
                            if description:
                                print(f"{description}")
                            else:
                                print("No description.") 

                    else:
                        print("No events detected by the model.")
                except Exception as e:
                    print(f"Error parsing Pydantic object: {e}")
                    print(f"Arguments that failed parsing: {arguments}")
    else:
        print("The model did not make the expected `extract_video_events` function call.")
        print("Raw response content:")
        print(f"Message content: {message.content}")

except Exception as e:
    print(f"An error occurred during content generation: {e}")
