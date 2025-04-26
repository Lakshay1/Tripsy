"""
Skeleton orchestrator agent that demonstrates Anthropic tool‑use (function calling).

▸ Requires the `anthropic` Python SDK ≥ 0.23.0
   pip3 install anthropic
   pip3 install google-api-python-client google-auth-oauthlib google-auth-httplib2

▸ Make sure the `ANTHROPIC_API_KEY` environment variable is set before running:
   export ANTHROPIC_API_KEY="sk‑..."

The orchestrator keeps the chat loop alive until Claude no longer wants to call a
client‑side tool. It extracts tool‑use blocks, executes the mapped Python
function, feeds the result back to Claude with a `tool_result` block, and
finally prints the model’s answer.

Replace the dummy `get_weather` implementation with a real API call (or register
additional tools) to build richer agents.
"""
from __future__ import annotations

import cv2
import os
import requests
from typing import Any, Callable, Dict, List
import base64
import subprocess
import json
import numpy as np

import anthropic # type: ignore

from tools.email_tool.tool import fetch_emails

BASE_PROMPT = '/Users/lakshayk/Developer/Tripsy/Tripsy/agent/base_prompt.txt'

# ---------------------------------------------------------------------------
# Core orchestrator class
# ---------------------------------------------------------------------------

class AnthropicOrchestrator:
    """Minimal orchestrator that handles Anthropic tool‑use messages."""

    def __init__(
        self,
        *,
        model: str = "claude-3-7-sonnet-20250219",  # ⇠ pick your favourite C3 tier
        tools: List[dict] | None = None,
        api_key: str | None = None,
        max_tokens: int = 64000,
    ) -> None:
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.tools: List[dict] = tools or []
        self._registry: Dict[str, Callable[..., Any]] = {}

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def register_tool(self, tool_def: dict, python_fn: Callable[..., Any]) -> None:
        """Attach a JSON tool definition **and** the Python function that executes it."""
        name = tool_def["name"]
        assert callable(python_fn), "python_fn must be a callable"
        self.tools.append(tool_def)
        self._registry[name] = python_fn

    def chat(self, user_prompt: str) -> anthropic.types.Message:
        """High‑level helper: send `user_prompt`, run tools if requested, return final msg."""
        messages: List[dict] = [{"role": "user", "content": user_prompt}]

        while True:
            response = self._send(messages)
            # pdb.set_trace()
            if response.stop_reason == "tool_use":
                # Claude wants one or more tools. Iterate over tool_use blocks.
                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    tool_use_id = block.id
                    tool_name = block.name
                    tool_args = block.input

                    result, error_flag = self._execute_local_tool(tool_name, tool_args)

                    # Tell Claude the result so it can compose its answer.
                    messages.extend(
                        [
                            {"role": "assistant", "content": response.content},
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_use_id,
                                        "content": str(result),
                                        **({"is_error": True} if error_flag else {}),
                                    }
                                ],
                            },
                        ]
                    )
                # Loop continues – Claude may want another tool call or produce final answer.
                continue

            # Any other stop_reason means we’re done ("stop_sequence", "max_tokens", etc.)
            return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send(self, messages: List[dict]) -> anthropic.types.Message:
        """Wrapper around anthropic.messages.create()."""
        return self.client.messages.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            max_tokens=self.max_tokens,
            # Uncomment for token‑efficient tools beta (if you have access):
            # headers={"anthropic-beta": "token-efficient-tools-2025-02-19"},
        )

    def _execute_local_tool(self, name: str, kwargs: dict) -> tuple[Any, bool]:
        """Look up and run the registered python function. Returns (result, is_error)."""
        fn = self._registry.get(name)
        if fn is None:
            return f"No local implementation found for tool '{name}'", True
        try:
            return fn(**kwargs), False
        except Exception as exc:  # ⬅️ never let an exception kill the orchestrator
            return f"Tool execution failed: {exc}", True


def get_image_location(image_path: str) -> str:
    client = anthropic.Anthropic(api_key="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",)
    
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_media_type = "image/jpeg"
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    
    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": encoded_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": "What is the location is the image?"
                    }
                ]
            }
        ]
    )
    
    return response.content[0].text



def get_images_location(image_paths: list) -> str:
    """
    Sends multiple images to Anthropic for location analysis.
    
    Args:
        image_paths (list): List of image file paths (up to 4 images).
        
    Returns:
        str: Claude's text response analyzing the images.
    """
    client = anthropic.Anthropic(api_key="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")

    # Prepare the list of content (images + instruction)
    contents = []

    for image_path in image_paths:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_media_type = "image/jpeg"

        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        contents.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_media_type,
                "data": encoded_image,
            }
        })

    # Finally, add the instruction text
    contents.append({
        "type": "text",
        "text": "Based on these screenshots, where is the location?"
    })

    # Send to Anthropic
    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": contents
            }
        ]
    )

    return response.content[0].text


def extract_n_keyframes(video_path: str, output_dir: str, n_frames: int = 4) -> list:
    """
    Extracts exactly `n_frames` evenly spaced frames from a video, rotating if needed.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        raise ValueError("Video has no frames!")

    frame_indices = np.linspace(0, frame_count - 1, n_frames, dtype=int)

    saved_frames = []
    idx = 0

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        height, width = frame.shape[:2]        
        if height < width:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        frame_filename = os.path.join(output_dir, f"keyframe_{idx}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_frames.append(frame_filename)
        idx += 1

    cap.release()
    return saved_frames



def get_video_rotation(video_path: str) -> int:
    """
    Reads the rotation metadata from a video file.
    
    Returns:
        Rotation in degrees (0, 90, 180, 270).
    """
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream_tags=rotate',
        '-of', 'json', video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        rotation_info = json.loads(result.stdout)
        rotation = int(rotation_info['streams'][0]['tags']['rotate'])
        return rotation
    except (KeyError, IndexError, ValueError):
        return 0  # Default: no rotation
    

UNDERSTAND_IMAGE_TOOL_DEF = {
    "name": "get_image_location",
    "description": (
        "Analyzes the provided screenshot and returns a location."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Local file path to the screenshot image (JPEG/PNG).",
            }
        },
        "required": ["image_path"],
    },
}

UNDERSTAND_IMAGES_TOOL_DEF = {
    "name": "get_images_location",
    "description": (
        "Analyzes the provided screenshots and returns the location."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "image_paths": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "Local file path to a screenshot image (JPEG/PNG).",
                },
                "description": "List of local file paths to screenshots to be analyzed together.",
                "minItems": 1,
                "maxItems": 4
            }
        },
        "required": ["image_paths"],
    },
}

EMAIL_TOOL_DEF = {
    "name": "fetch_emails",
    "description": (
        "Gets emails from the user's gmail inbox using filters to find relevant bookings "
        "for flights, hotels, and other travel-related items."
        "Please use following parameters to filter the emails: " \
        "1. title_keywords=['itinerary', 'booking', 'reservation', 'airline', 'emirates', 'united', 'air india'], " \
        "2. content_keywords=['trip', 'fly', 'booking', 'hotel', 'itinerary', 'flight', 'reservation', 'airbnb', 'booking']" \
        "3. start_date: Always set it to 2025/03/05" \
        "4. end_date: Always set it to 2025/03/15" \
        "Example invocation: - fetch_emails(labels=['INBOX', 'UPDATES'], start_date='2025/03/09', end_date='2025/03/20', title_keywords=['itinerary', 'booking'], " \
                         "content_keywords=['emirates', 'trip', 'fly', 'booking', 'hotel', 'itinerary', 'flight', 'reservation', 'airbnb', 'booking'])"
        ),
    "input_schema": {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": { "type": "string" },
                "description": "List of labels to filter emails by (e.g. ['travel', 'bookings', 'flights', 'location']).",
            },
            "start_date": {
                "type": "string",
                "description": "Filter to specify start date from where emails should be read from.",
            },
            "end_date": {
                "type": "string",
                "description": "Filter to specify end date until where emails should be read from.",
            },
            "title_keywords": {
                "type": "array",
                "items": { "type": "string" },
                "description": "List of keywords to filter emails by (e.g. ['flight', 'hotel', 'booking']).",
            },
            "content_keywords": {
                "type": "array",
                "items": { "type": "string" },
                "description": "List of keywords to filter emails by (e.g. ['flight', 'hotel', 'booking']). Defaults to []",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of emails to return. Defaults to 50.",
            },
        },
        "required": [],
    },
}



# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    orchestrator = AnthropicOrchestrator(api_key="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")
    orchestrator.register_tool(EMAIL_TOOL_DEF, fetch_emails)

    image_path = "location.jpg"
    video_path = "video.mp4"
    output_dir = "keyframes"
    download_folder = "./downloads"

    keyframes = extract_n_keyframes(video_path, output_dir)
    print(f"Extracted {len(keyframes)} keyframes:")
    for path in keyframes:
        print(path)
    image_paths = keyframes

    orchestrator.register_tool(UNDERSTAND_IMAGE_TOOL_DEF, get_image_location)
    orchestrator.register_tool(UNDERSTAND_IMAGES_TOOL_DEF, get_images_location)

    screenshots_prompt = f"I have uploaded 4 images at {image_paths}. Can you figure out where is this location?"
    # final_msg = orchestrator.chat(screenshots_prompt)


    final_msg = orchestrator.chat("Give me the itinerary of my upcoming trip in June?")
    # Claude’s reply is a list of content blocks – normally just one text block here.
    print(final_msg.content[0].text)
