import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import av
import requests
import torch
from PIL import Image
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


def download_video(url: str, output_path: str) -> str:
    """Download video from URL"""
    print(f"Downloading video from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Video downloaded to {output_path}")
    return output_path


def extract_frames_ffmpeg(
    video_path: str, fps: float = 1.0, max_resolution: int = 720
) -> Tuple[List[Image.Image], List[float], float]:
    """Extract frames from video using ffmpeg at specified FPS and resolution"""
    frames = []
    temp_dir = tempfile.mkdtemp()

    try:
        print(f"Extracting frames at {fps} FPS, max resolution {max_resolution}p...")

        # Get video duration
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        duration = float(subprocess.check_output(probe_cmd).decode().strip())
        print(f"Video duration: {duration:.2f} seconds")

        # Extract frames with ffmpeg
        output_pattern = os.path.join(temp_dir, "frame_%04d.jpg")
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            f"fps={fps},scale=-1:min(ih\\,{max_resolution})",
            "-q:v",
            "2",
            output_pattern,
        ]

        subprocess.run(cmd, check=True)

        # Load extracted frames
        frame_files = sorted(Path(temp_dir).glob("frame_*.jpg"))
        for frame_file in frame_files:
            img = Image.open(frame_file).convert("RGB")
            frames.append(img)

        print(f"Extracted {len(frames)} frames")

        # Calculate timestamps
        timestamps = [i / fps for i in range(len(frames))]

        return frames, timestamps, duration

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def extract_frames_av(
    video_path: str, fps: float = 1.0, max_resolution: int = 720
) -> Tuple[List[Image.Image], List[float], float]:
    """Extract frames using av library (pure Python)"""
    frames = []
    timestamps = []

    container = av.open(video_path)
    stream = container.streams.video[0]

    # Calculate frame interval
    fps_original = float(stream.average_rate)
    frame_interval = max(1, int(round(fps_original / fps)))
    if stream.duration is not None:
        duration = float(stream.duration * stream.time_base)
    elif container.duration is not None:
        duration = float(container.duration / av.time_base)
    else:
        duration = 0.0

    print(f"Video duration: {duration:.2f} seconds")
    print(f"Original FPS: {fps_original:.2f}, extracting every {frame_interval} frames for {fps} FPS output")

    frame_count = 0
    for frame in container.decode(stream):
        if frame_count % frame_interval == 0:
            # Convert to PIL Image
            img = frame.to_image()

            # Resize if needed
            if img.height > max_resolution:
                aspect_ratio = img.width / img.height
                new_height = max_resolution
                new_width = int(new_height * aspect_ratio)
                img = img.resize((new_width, new_height), Image.LANCZOS)

            frames.append(img)
            timestamps.append(float(frame.time))

        frame_count += 1

    print(f"Extracted {len(frames)} frames")
    return frames, timestamps, duration


def process_video_with_llava(
    video_path: str, fps: float = 1.0, use_ffmpeg: bool = True
) -> str:
    """Process video with LLaVA-NeXT-Video model"""

    print(f"Loading LLaVA-NeXT-Video model: {MODEL_ID}")
    print(f"Device: {device}, dtype: {dtype}")

    processor = LlavaNextVideoProcessor.from_pretrained(MODEL_ID)
    if device == "cuda":
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=dtype, device_map="auto"
        )
    else:
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=dtype
        ).to(device)

    print("Model loaded successfully!")

    # Extract frames
    if use_ffmpeg:
        frames, timestamps, duration = extract_frames_ffmpeg(video_path, fps=fps)
    else:
        frames, timestamps, duration = extract_frames_av(video_path, fps=fps)

    # Create timestamped narrative prompt
    prompt = """<video>
Describe this video in vivid, precise detail as a timestamped narrative. 
For each second, describe:
- What the camera sees and how it moves
- Objects, people, architecture, and environments visible
- Lighting, colors, textures, and materials
- Actions and movements happening
- Spatial relationships and depth
- Any text, signs, or notable details

Format as a story with timestamps, describing the journey through space as the camera moves.
Provide rich sensory details and precise observations for each timestamp."""

    print("Processing video frames with model...")

    # Process video frames with the model
    inputs = processor(
        text=prompt, videos=frames, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    print("Generating temporal description...")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            num_beams=3,
            temperature=0.7,
            do_sample=True,
            min_length=500,
        )

    description = processor.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ].strip()

    # Format with actual timestamps
    timestamped_description = f"Video Duration: {duration:.2f} seconds\n"
    timestamped_description += f"Frames analyzed: {len(frames)} at {fps} FPS\n\n"
    timestamped_description += "=== Temporal Narrative ===\n\n"

    # Add timestamp markers if not present in description
    if "[0:" not in description and "0s:" not in description:
        # Add our own timestamp structure
        lines = description.split(". ")
        for i, (timestamp, line) in enumerate(zip(timestamps, lines)):
            if i < len(lines):
                timestamped_description += f"[{timestamp:.1f}s] {line.strip()}.\n\n"
    else:
        timestamped_description += description

    return timestamped_description


def main():
    # Video URL or local path
    video_url = (
        "https://mcmoodoo-playground.s3.us-east-1.amazonaws.com/super_hallway.mp4"
    )

    # Download or use local video
    if video_url.startswith("http"):
        video_path = "temp_video.mp4"
        download_video(video_url, video_path)
    else:
        video_path = video_url

    try:
        # Process video
        description = process_video_with_llava(
            video_path,
            fps=1.0,  # 1 frame per second
            use_ffmpeg=True,  # Use ffmpeg if available, otherwise av
        )

        print("\n" + "=" * 60)
        print("VIDEO DESCRIPTION")
        print("=" * 60)
        print(description)
        print("=" * 60)

        # Save to file
        with open("video_description.txt", "w", encoding="utf-8") as f:
            f.write(description)
        print("\nDescription saved to video_description.txt")

    finally:
        # Clean up downloaded video
        if video_url.startswith("http") and os.path.exists(video_path):
            os.remove(video_path)
            print("Cleaned up temporary video file")


if __name__ == "__main__":
    main()
