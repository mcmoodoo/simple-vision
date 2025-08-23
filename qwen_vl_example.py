import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

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

image = Image.open("assets/screenshot.png").convert("RGB")

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
                "text": """You are an expert interior designer and property assessor conducting a comprehensive visual analysis. Provide an extraordinarily detailed description of this interior space.

**Opening Overview**: Set the scene - what type of room, overall style, immediate impression, approximate era and price point.

**Architectural Analysis**: 
- Ceiling: height, type (coffered, tray, flat), crown molding details, any beams or architectural interest
- Walls: paint finish, wallpaper, wainscoting, texture, any architectural details
- Flooring: material (hardwood species, tile type, carpet), pattern, condition, transitions
- Windows/Doors: style, trim work, hardware, treatments (curtains, blinds, shutters)

**Furniture Inventory**:
- Primary seating: sofa/sectional style, fabric/leather, color, condition, throw pillows
- Secondary seating: chairs, ottomans, benches - materials and styles
- Tables: coffee table, end tables, console tables - materials, finishes, style period
- Storage: bookcases, cabinets, media units - wood type, configuration

**Lighting Design**:
- Natural light: direction, intensity, time of day suggested
- Fixtures: overhead (chandelier, flush mount, pendant), table lamps, floor lamps
- Ambiance: warm/cool tones, brightness levels, shadows and highlights

**Decorative Elements**:
- Wall art: paintings, prints, photographs - subjects, frames, arrangement
- Textiles: rugs (pattern, size, material), curtains, pillows, throws
- Accessories: vases, sculptures, books, plants, decorative objects
- Color coordination: how accessories complement the overall palette

**Material & Finish Quality**:
- Wood: stain color, grain visibility, species if identifiable
- Metals: brass, chrome, iron, bronze - patina and finish
- Fabrics: velvet, linen, cotton, leather - texture and pattern
- Stone/Glass: marble veining, glass clarity, mirror placement

**Spatial Design**:
- Room flow: traffic patterns, furniture arrangement logic
- Focal points: fireplace, view, art wall, architectural feature
- Balance: symmetrical vs asymmetrical arrangements
- Scale: proportion of furniture to room size

**Style Assessment**:
- Design style: Traditional, Contemporary, Transitional, Modern, etc.
- Period influences: Victorian, Mid-Century Modern, Art Deco, etc.
- Regional influences: French Country, Scandinavian, Mediterranean
- Designer touches: custom elements, high-end details

Write 6-8 detailed paragraphs using rich, descriptive language. Paint such a vivid picture that someone could recreate this room from your description alone.""",
            },
        ],
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
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
description = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(
    f"\n=== Qwen2-VL Detailed Description ===\n{description}\n========================"
)

with open("qwen_vl_description.txt", "w", encoding="utf-8") as f:
    f.write(description + "\n")
