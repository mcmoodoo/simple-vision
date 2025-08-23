import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

# LLaVA-NeXT: State-of-the-art vision-language model for detailed descriptions
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"  # Or "llava-hf/llava-v1.6-vicuna-7b-hf"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading LLaVA model: {MODEL_ID}")
print(f"Device: {device}, dtype: {dtype}")

processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    torch_dtype=dtype,
    device_map="auto",
    low_cpu_mem_usage=True
)

image = Image.open("assets/screenshot.png").convert("RGB")

# LLaVA excels at following detailed instructions
prompt = """You are a luxury real estate agent providing a detailed property description. Analyze this interior space with meticulous attention to detail:

**Architectural Features**: Ceiling height, crown molding, baseboards, window treatments, flooring type and condition, wall textures, built-ins, archways, columns

**Furniture & Layout**: Every piece of furniture - sofas, chairs, tables, cabinets. Their style (modern, traditional, contemporary), materials (leather, fabric, wood type), colors, arrangement, spacing

**Lighting Analysis**: Natural light sources, window placement and size, artificial lighting - chandeliers, lamps, recessed lights, sconces. Describe the quality and mood of lighting

**Color Palette & Textures**: Wall colors, accent colors, textile patterns, surface finishes (matte, glossy, textured), metal finishes (brass, chrome, bronze)

**Decorative Elements**: Artwork, mirrors, plants, vases, books, sculptures, rugs, pillows, throws. Note their placement and how they contribute to the aesthetic

**Materials & Finishes**: Wood grains, stone types (marble, granite), fabric textures, glass elements, metal accents

**Style & Ambiance**: Design style (minimalist, traditional, transitional, industrial), the mood created, target demographic, price point indication

**Spatial Qualities**: Room dimensions feel, flow between spaces, sight lines, focal points, symmetry or asymmetry

**Condition & Quality**: Signs of luxury, craftsmanship details, any wear or aging, maintenance level

Write multiple detailed paragraphs painting a vivid picture that would help someone visualize this space without seeing it."""

inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

print("Generating detailed description...")
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=2048,  # Very long output
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
    )

description = processor.decode(output[0], skip_special_tokens=True)
print(f"\n=== LLaVA Detailed Description ===\n{description}\n========================")

with open("llava_description.txt", "w", encoding="utf-8") as f:
    f.write(description + "\n")