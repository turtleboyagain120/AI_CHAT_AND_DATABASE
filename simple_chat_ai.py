import requests
import json
import os
import sys

API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"  # Free, no key needed
# Or use "gpt2" for tiny/fast

# Try local mode
LOCAL_MODEL_PATH = "./my_finetuned_model"
local_model = None
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    if os.path.exists(LOCAL_MODEL_PATH):
        base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        local_model = PeftModel.from_pretrained(base_model, LOCAL_MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        tokenizer.pad_token = tokenizer.eos_token
        local_model.eval()
        print("✅ Local model loaded. Using local AI (offline). Use /remote to switch.")
        USE_LOCAL = True
    else:
        USE_LOCAL = False
except ImportError as e:
    print(f"Local mode unavailable (missing deps: {e}). Using remote.")
    USE_LOCAL = False
except Exception as e:
    print(f"Local model load failed: {e}. Using remote.")
    USE_LOCAL = False

def query_remote(prompt):
    try:
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 150, "temperature": 0.7, "return_full_text": False}}
        response = requests.post(API_URL, json=payload, timeout=15)
        if response.status_code == 200:
            generated = response.json()[0]["generated_text"].strip()
            if "Assistant:" in generated:
                return generated.split("Assistant:")[-1].strip()
            return generated
        elif response.status_code == 503:
            return "AI is loading or busy. Try again in 30s!"
        else:
            return f"API error {response.status_code}: {response.text[:100]}"
    except requests.exceptions.RequestException as e:
        return f"Network error: {str(e)}"
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        return "AI response format error. Try again."
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def generate_local(prompt):
    if not local_model:
        return "Local model not available."
    try:
        inputs = tokenizer(f"{prompt} ", return_tensors="pt")
        with torch.no_grad():
            outputs = local_model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        return response
    except Exception as e:
        return f"Local generation error: {str(e)}"

def get_ai_reply(prompt, use_local):
    if use_local:
        return generate_local(prompt)
    else:
        return query_remote(prompt)

print("🤖 Free AI Chat (no API key!) - Type 'quit' to stop")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    ai_reply = query_ai(f"Human: {user_input}\\nAssistant:")
    print(f"AI: {ai_reply}")

# Uses pre-trained on billions convos - like "other AIs" data!

