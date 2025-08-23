import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

# PaliGemma: Google's vision-language model based on Gemma
MODEL_ID = "google/paligemma-3b-mix-448"  # Most capable PaliGemma model

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading PaliGemma model: {MODEL_ID}")
print(f"Device: {device}, dtype: {dtype}")

processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto",
    revision="bfloat16",
).eval()

image = Image.open("assets/screenshot.png").convert("RGB")

# PaliGemma works best with task-specific prompts
prompts = [
    "describe this room in detail",
    "what furniture is visible?",
    "describe the architectural features",
    "what is the color scheme?",
    "describe the lighting",
    "what decorative elements are present?",
    "what materials and textures are visible?",
    "what style is this interior?",
]

print("Generating detailed descriptions...")
full_description = []

for prompt in prompts:
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding="longest",
    ).to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )
    
    # Decode only the generated portion
    generated_text = processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]
    
    full_description.append(f"**{prompt.capitalize()}**\n{generated_text}\n")
    print(f"Completed: {prompt}")

# Combine all descriptions
combined_description = "\n".join(full_description)

print(f"\n=== PaliGemma Detailed Description ===\n{combined_description}\n========================")

with open("paligemma_description.txt", "w", encoding="utf-8") as f:
    f.write(combined_description + "\n")