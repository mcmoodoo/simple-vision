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

processor = Blip2Processor.from_pretrained(MODEL_ID)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=dtype, device_map="auto" if device == "cuda" else None
).to(device)

# Path to your image:
# (In this chat upload it was: /mnt/data/7 (2).png — on your machine, use your local path)
image = Image.open("path/to/your_image.png").convert("RGB")

prompt = (
    "You are a meticulous vision assistant. Describe the image exhaustively: "
    "objects, layout, text on screen, numbers, colors, relationships, scene context, "
    "UI elements, and actions. Use short paragraphs and bullet points."
)

inputs = processor(images=image, text=prompt, return_tensors="pt").to(
    device, dtype=dtype
)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=128,  # ↑ for longer descriptions
    num_beams=1,  # beam search for quality; use 1 for speed
    length_penalty=1.05,
    repetition_penalty=1.05,
    early_stopping=True,
)

caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(caption)

# (optional) save to file
with open("blip2_description.txt", "w", encoding="utf-8") as f:
    f.write(caption + "\n")
