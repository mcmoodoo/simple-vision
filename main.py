import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

# Good descriptive model; swap to base for CPU if needed
# MODEL_ID = "Salesforce/blip2-flan-t5-xl"  # heavy, highest quality, requires auth
# MODEL_ID = "Salesforce/blip2-flan-t5-base"  # lighter/faster, requires auth
# MODEL_ID = "Salesforce/instructblip-flan-t5-xl"  # better at following prompts, requires auth
MODEL_ID = "Salesforce/blip2-opt-2.7b"  # No auth required

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading model: {MODEL_ID}")
print(f"Device: {device}, dtype: {dtype}")
processor = Blip2Processor.from_pretrained(MODEL_ID)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=dtype, device_map="auto" if device == "cuda" else None
).to(device)
print("Model loaded successfully!")

# Path to your image:
# (In this chat upload it was: /mnt/data/7 (2).png — on your machine, use your local path)
image = Image.open("assets/screenshot.png").convert("RGB")

prompt = "Question: Describe this property interior in vivid detail. What furniture is visible? What are the architectural features? Describe the lighting, colors, textures, materials, decor, and spatial arrangement. What style is the room? What mood does it convey? Answer:"

print("Processing image with prompt...")
inputs = processor(images=image, text=prompt, return_tensors="pt").to(
    device, dtype=dtype
)

print("Generating caption...")
generated_ids = model.generate(
    **inputs,
    max_new_tokens=1024,  # Maximum tokens for longest possible description
    num_beams=5,  # Higher beam search for quality
    temperature=0.7,
    do_sample=False,
    min_length=100,  # Force longer outputs
)

print("Decoding output...")
caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(f"\n=== Generated Caption ===\n{caption}\n========================")

# (optional) save to file
with open("blip2_description.txt", "w", encoding="utf-8") as f:
    f.write(caption + "\n")
