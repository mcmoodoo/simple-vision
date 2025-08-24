import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

# CogVLM2: Excellent at detailed visual understanding
MODEL_ID = "THUDM/cogvlm2-llama3-chat-19B"  # Very powerful, 19B params

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading CogVLM2 model: {MODEL_ID}")
print(f"Device: {device}, dtype: {dtype}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

image = Image.open("assets/screenshot.png").convert("RGB")

prompt = """Provide an exhaustive interior design analysis of this property photo. Your description should read like a high-end architectural magazine feature:

Begin with the overall impression and atmosphere. Then systematically describe:

- The room's architectural bones: ceiling details, wall treatments, flooring materials, window configurations
- Every furniture piece: manufacturer style, upholstery, wood finishes, proportions, placement
- Lighting layers: how natural light enters, every fixture visible, shadows and highlights created
- The complete color story: primary palette, accent colors, how colors interact and flow
- Textiles and soft furnishings: fabric types, patterns, textures that add warmth
- Art and accessories: what's on the walls, decorative objects, their arrangement and purpose
- Material palette: identify wood species, stone types, metal finishes, glass elements
- Spatial planning: traffic flow, conversation areas, focal points, use of negative space
- Design style and era: specific style references, period details, contemporary vs traditional elements
- Quality indicators: craftsmanship details, luxury materials, custom elements

Write at least 4-5 substantial paragraphs with rich, descriptive language that captures every nuance."""

# CogVLM2 specific formatting
query = f"Human: {prompt}\nAssistant:"
inputs = model.build_conversation_input_ids(tokenizer, query=query, images=[image])

# Move only tensor inputs to device
for k, v in inputs.items():
    if torch.is_tensor(v):
        inputs[k] = v.to(device)

print("Generating detailed description...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
    )

description = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(
    f"\n=== CogVLM2 Detailed Description ===\n{description}\n========================"
)

with open("cogvlm_description.txt", "w", encoding="utf-8") as f:
    f.write(description + "\n")
