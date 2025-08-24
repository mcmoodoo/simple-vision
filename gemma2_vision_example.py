import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Gemma-2-based vision model (using LLaVA architecture with Gemma backbone)
# Note: There isn't a standalone "Gemma3 12B vision" model, but we can use
# models that incorporate Gemma architecture for vision tasks

MODEL_ID = "TIGER-Lab/Mantis-8B-siglip-llama3"  # Alternative powerful vision model
# Or try: "llava-hf/llava-gemma-2b" for a lighter Gemma-based option

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading vision model: {MODEL_ID}")
print(f"Device: {device}, dtype: {dtype}")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=dtype, device_map="auto", low_cpu_mem_usage=True
)

image = Image.open("assets/screenshot.png").convert("RGB")

# Comprehensive property analysis prompt
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": """You are an expert interior designer analyzing a property photograph. Provide an exhaustive analysis covering:

**OPENING IMPRESSION**
Describe your immediate reaction to the space - the overall ambiance, style period, and quality level. What lifestyle does this space suggest?

**ARCHITECTURAL ANALYSIS**
- Ceiling details: height estimation, any coffers, beams, crown molding, or architectural interest
- Wall treatments: paint finishes, wallpaper, paneling, texture applications
- Flooring: material identification (wood species if visible), pattern, condition, transitions
- Windows and doors: style, trim details, hardware finishes, natural light quality
- Built-in features: shelving, cabinetry, architectural niches

**FURNITURE AUDIT**
For each piece of furniture visible:
- Type and probable manufacturer/style reference
- Materials: wood types, upholstery fabrics, leather grades
- Color and finish details
- Condition and quality indicators
- Arrangement and spatial relationships

**LIGHTING DESIGN**
- Natural light: direction, intensity, time of day suggested by shadows
- Artificial lighting: fixture types, styles, placement strategy
- Ambient vs task vs accent lighting layers
- How lighting affects the mood and functionality

**COLOR & MATERIAL PALETTE**
- Primary color scheme: wall colors with specific shade descriptions
- Secondary colors: furniture, textiles, accents
- Material mix: woods, metals, glass, stone, fabrics
- Texture interplay: smooth, rough, glossy, matte surfaces
- Pattern usage: geometric, organic, traditional motifs

**DECORATIVE LAYER**
- Artwork: subjects, framing, placement, scale
- Soft furnishings: pillows, throws, window treatments
- Objects and accessories: books, vases, sculptures
- Plants and natural elements
- How these elements contribute to the overall design story

**SPATIAL DESIGN**
- Room flow and circulation patterns
- Furniture groupings and conversation areas
- Focal points and visual anchors
- Proportion and scale relationships
- Use of negative space

**STYLE CLASSIFICATION**
- Primary design style: Traditional, Contemporary, Transitional, etc.
- Secondary influences and style mixing
- Era references: specific decade or period details
- Regional or cultural influences
- Designer signature elements

**VALUE INDICATORS**
- Quality of materials and craftsmanship
- Custom vs mass-market elements
- Maintenance and condition
- Approximate market positioning

Write 5-6 detailed paragraphs that would allow someone to perfectly visualize and recreate this space.""",
            },
        ],
    }
]

# Process the conversation
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(text=text_prompt, images=[image], return_tensors="pt").to(device)

print("Generating comprehensive interior analysis...")
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=2500,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        repetition_penalty=1.1,
    )

# Decode only the generated portion
generated_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
description = processor.decode(generated_tokens, skip_special_tokens=True)

print(f"\n=== Detailed Interior Analysis ===\n{description}\n========================")

with open("gemma_vision_description.txt", "w", encoding="utf-8") as f:
    f.write(description + "\n")
