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

image = Image.open("assets/Vacation_checkout_page.png").convert("RGB")

# LLaVA excels at following detailed instructions
prompt = """<image>
Please provide an extremely detailed description of this image. Include:

1. **Overall Layout**: Describe the page structure, sections, and visual hierarchy
2. **Text Content**: Quote ALL visible text, headings, labels, buttons, prices, etc.
3. **Visual Elements**: Icons, images, logos, graphics, and their positions
4. **Colors**: Specific color schemes, backgrounds, text colors, button colors
5. **Forms and Inputs**: All form fields, dropdowns, checkboxes, their labels and states
6. **Navigation**: Menu items, breadcrumbs, links
7. **Specific Details**: Product information, pricing, quantities, dates, any numbers
8. **UI Components**: Cards, containers, dividers, spacing, alignment
9. **Interactive Elements**: What appears clickable or editable
10. **Context**: What type of page/application this appears to be

Be exhaustive - describe every single element you can see, no matter how small."""

inputs = processor(prompt, image, return_tensors="pt").to(device)

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