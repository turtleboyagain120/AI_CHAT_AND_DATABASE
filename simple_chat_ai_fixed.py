import requests
import json
import os
import sys

# Global
USE_LOCAL = False
local_model = None
tokenizer = None

API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"  # Free, no key needed

# Local model setup
LOCAL_MODEL_PATH = "./my_finetuned_model"
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    if os.path.exists(LOCAL_MODEL_PATH):
        base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        local_model = PeftModel.from_pretrained(base_model, LOCAL_MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        local_model.eval()
        USE_LOCAL = True
        print("✅ Local model loaded. Starting in local mode (offline). /mode remote for HF API.")
    else:
        USE_LOCAL = False
        print("No local model found. Using remote HF API.")
except ImportError:
    print("Local mode requires torch/transformers/peft. Install from requirements.txt. Using remote.")
except Exception as e:
    print(f"Local setup failed ({e}). Using remote.")

def query_remote(prompt):
    try:
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 150, "temperature": 0.7, "return_full_text": False}}
        response = requests.post(API_URL, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        generated = data[0]["generated_text"].strip()
        if "Assistant:" in generated:
            return generated.split("Assistant:")[-1].strip()
        return generated or "Hmm, let me think..."
    except Exception as e:
        return f"Remote API issue: {str(e)[:100]}"

def generate_local(prompt):
    if not local_model or not tokenizer:
        return "Local model not ready."
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = local_model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
        return response or "Local AI thinking..."
    except Exception as e:
        return f"Local error: {str(e)}"

def get_reply(prompt):
    if USE_LOCAL and local_model:
        return generate_local(prompt)
    return query_remote(prompt)

print("🤖 Super AI Chat - 'quit' to exit, '/clear' history, '/mode local|remote'")
if USE_LOCAL:
    print("Local mode active (offline).")
print("Start chatting!")

history = []
MAX_HISTORY = 6

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() == 'quit':
        break
    if user_input == '/clear':
        history = []
        print("🧹 History cleared.")
        continue
    if user_input.startswith('/mode'):
        mode = user_input.split()[-1].lower() if len(user_input.split()) > 1 else ''
        global USE_LOCAL
        if mode == 'local' and local_model:
            USE_LOCAL = True
            print("🔄 Switched to local.")
        elif mode == 'remote':
            USE_LOCAL = False
            print("🔄 Switched to remote.")
        else:
            print("Usage: /mode local or /mode remote")
        continue

    # Build context
    context = ''.join([f"Human: {h['h']}\\nAI: {h['a']}\\n" for h in history[-MAX_HISTORY+1:]])
    prompt = f"{context}Human: {user_input}\\nAI:"
    reply = get_reply(prompt)
    print(f"AI ({'local 🔒' if USE_LOCAL else 'remote ☁️'}): {reply}")

    history.append({'h': user_input, 'a': reply})

print("👋 Bye!")
