import os
from typing import List, Tuple
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import requests
import av

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = (
    torch.float16
    if device == "cuda"
    else torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float32
)


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


def process_video_with_qwen(video_path: str, fps: float = 1.0) -> str:
    """Process video with Qwen2-VL model which has native video support"""

    print(f"Loading Qwen2-VL model: {MODEL_ID}")
    print(f"Device: {device}, dtype: {dtype}")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map="auto" if device == "cuda" else None
    ).to(device)

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print("Model loaded successfully!")

    # Create conversation with video
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": fps,  # Sample at this FPS
                    "max_pixels": 360 * 420,  # Reduce resolution for efficiency
                },
                {
                    "type": "text",
                    "text": """Describe this video as a vivid, timestamped narrative story. 

For every second of the video, provide:
- [Xs] timestamp marker
- Camera movement and perspective
- Detailed description of visible objects, architecture, people
- Colors, lighting, textures, materials
- Actions and movements
- Spatial relationships and depth
- Any text, signs, or notable details

Write it as a journey narrative, describing what the camera sees as it moves through the space.
Be extremely precise and detailed for each timestamp.""",
                },
            ],
        }
    ]

    print("Processing video with model...")

    # Process with model
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    print("Generating temporal description...")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=3000,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            min_length=800,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    description = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return description


def extract_frames_for_backup(
    video_path: str, fps: float = 1.0, max_frames: int = 30
) -> Tuple[List[Image.Image], List[float]]:
    """Extract frames using av library as backup for frame-by-frame processing"""
    frames = []
    timestamps = []

    container = av.open(video_path)
    stream = container.streams.video[0]

    fps_original = float(stream.average_rate)
    frame_interval = max(1, int(fps_original / fps))
    duration = float(stream.duration * stream.time_base) if stream.duration else 0

    print(f"Video info: duration={duration:.2f}s, original_fps={fps_original:.2f}")

    frame_count = 0
    for frame in container.decode(stream):
        if frame_count % frame_interval == 0 and len(frames) < max_frames:
            img = frame.to_image()

            # Resize for efficiency
            if img.height > 720:
                aspect_ratio = img.width / img.height
                new_height = 720
                new_width = int(new_height * aspect_ratio)
                img = img.resize((new_width, new_height), Image.LANCZOS)

            frames.append(img)
            timestamps.append(float(frame.time))

        frame_count += 1
        if len(frames) >= max_frames:
            break

    container.close()
    print(f"Extracted {len(frames)} frames")
    return frames, timestamps


def process_video_framewise(video_path: str, model, processor, fps: float = 1.0) -> str:
    """Process video frame by frame for detailed timestamped descriptions"""

    frames, timestamps = extract_frames_for_backup(video_path, fps=fps)

    descriptions = []

    print(f"Processing {len(frames)} frames individually...")

    for i, (frame, ts) in enumerate(zip(frames, timestamps)):
        print(f"Processing frame {i + 1}/{len(frames)} at {ts:.1f}s...")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": """Describe this scene in vivid detail. Include:
- All visible objects and their positions
- Architecture and spatial layout
- Lighting, colors, textures, materials
- Any people and their actions
- Signs, text, or notable details
- The perspective and what appears to be the camera's position

Be extremely precise and detailed.""",
                    },
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=[text],
            images=[frame],
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.7,
                do_sample=True,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        frame_description = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        descriptions.append(f"[{ts:.1f}s] {frame_description}")

    # Combine into narrative
    narrative = "=== Timestamped Video Journey ===\n\n"
    narrative += "\n\n".join(descriptions)

    return narrative


def main():
    video_url = "https://mcmoodoo.s3.us-east-1.amazonaws.com/super_hallway.mp4"

    # Download or use local video
    if video_url.startswith("http"):
        video_path = "temp_video.mp4"
        download_video(video_url, video_path)
    else:
        video_path = video_url

    try:
        # Try native video processing first
        try:
            print("\nAttempting native video processing with Qwen2-VL...")
            description = process_video_with_qwen(video_path, fps=1.0)
        except Exception as e:
            print(f"Native video processing failed: {e}")
            print("\nFalling back to frame-by-frame processing...")

            # Load model for frame processing
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
            ).to(device)
            processor = AutoProcessor.from_pretrained(MODEL_ID)

            description = process_video_framewise(video_path, model, processor, fps=1.0)

        print("\n" + "=" * 60)
        print("VIDEO DESCRIPTION")
        print("=" * 60)
        print(description)
        print("=" * 60)

        # Save to file
        with open("video_description_qwen.txt", "w", encoding="utf-8") as f:
            f.write(description)
        print("\nDescription saved to video_description_qwen.txt")

    finally:
        # Clean up
        if video_url.startswith("http") and os.path.exists(video_path):
            os.remove(video_path)
            print("Cleaned up temporary video file")


if __name__ == "__main__":
    main()
