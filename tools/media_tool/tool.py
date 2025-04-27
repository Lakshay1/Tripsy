import anthropic # type: ignore
import cv2
import os
import base64
import subprocess
import json
import numpy as np

from typing import List

API_KEY = ''
with open('/Users/lakshayk/Developer/Tripsy/Tripsy/anthropic_key.txt', 'r') as file:
    API_KEY = file.read().strip()

def get_image_location(image_path: str) -> str:
    client = anthropic.Anthropic(api_key=API_KEY,)
    
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



def get_video_location(video_path: str) -> str:
    """
    Sends multiple images to Anthropic for location analysis.
    
    Args:
        video_path (str): List of image file paths (up to 4 images).
        
    Returns:
        str: Claude's text response analyzing the images.
    """
    client = anthropic.Anthropic(api_key=API_KEY,)

    # Prepare the list of content (images + instruction)
    contents = []
    output_dir = "keyframes"
    image_paths = extract_n_keyframes(video_path, output_dir)
    # print(f"Extracted {len(keyframes)} keyframes:")
    # for path in keyframes:
        # print(path)
    # image_paths = keyframes

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