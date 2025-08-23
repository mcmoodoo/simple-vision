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
    low_cpu_mem_usage=True
)

image = Image.open("assets/Vacation_checkout_page.png").convert("RGB")

prompt = """Analyze this image with extreme attention to detail. Provide a comprehensive description that includes:

- Every piece of text visible (quote it exactly)
- All UI elements and their arrangement
- Color schemes and visual design
- Form fields and their contents
- Navigation elements
- Prices, quantities, dates
- Icons and graphics
- The purpose and context of the page
- Any subtle details or patterns

Write multiple paragraphs covering different aspects of the image. Be as thorough as possible."""

# CogVLM2 specific formatting
query = f"Human: {prompt}\nAssistant:"
inputs = model.build_conversation_input_ids(
    tokenizer,
    query=query,
    images=[image],
    device=device
)

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
print(f"\n=== CogVLM2 Detailed Description ===\n{description}\n========================")

with open("cogvlm_description.txt", "w", encoding="utf-8") as f:
    f.write(description + "\n")