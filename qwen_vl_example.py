from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image

# Qwen2-VL: Alibaba's powerful vision model
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"  # Or try 72B for even more detail!

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading Qwen2-VL model: {MODEL_ID}")
print(f"Device: {device}, dtype: {dtype}")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

image = Image.open("assets/Vacation_checkout_page.png").convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {
                "type": "text",
                "text": """You are a meticulous visual analyst. Examine this image with extraordinary attention to detail and provide an exhaustive description.

Your analysis should include multiple detailed paragraphs covering:

**Text Analysis**: Transcribe EVERY piece of text visible - headers, labels, buttons, prices, descriptions, fine print, tooltips, placeholders, error messages, etc. Note fonts, sizes, and emphasis.

**Visual Layout**: Describe the complete page structure - columns, rows, sections, cards, containers. Note spacing, alignment, borders, shadows, and visual hierarchy.

**Color Analysis**: Identify specific colors for backgrounds, text, buttons, links, borders. Note gradients, hover states, active states.

**Interactive Elements**: Detail every button, link, form field, dropdown, checkbox, radio button. Describe their states (enabled/disabled, selected/unselected).

**Data and Content**: Extract all prices, quantities, dates, times, percentages, product details, user information, order details.

**Images and Icons**: Describe every image, icon, logo, avatar. Note their size, position, and purpose.

**Micro-details**: Notice tooltips, badges, notifications, breadcrumbs, progress indicators, loading states, validation messages.

**Context and Purpose**: Explain what this interface is for, the user journey stage, and the intended actions.

Write at least 5-6 detailed paragraphs. Miss nothing."""
            }
        ]
    }
]

# Prepare inputs
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

print("Generating detailed description...")
generated_ids = model.generate(
    **inputs,
    max_new_tokens=3000,  # Allow very long outputs
    temperature=0.7,
    do_sample=True,
    top_p=0.95,
)

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
description = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(f"\n=== Qwen2-VL Detailed Description ===\n{description}\n========================")

with open("qwen_vl_description.txt", "w", encoding="utf-8") as f:
    f.write(description + "\n")