import argparse
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline, 
    TextIteratorStreamer
)
from peft import PeftModel
from threading import Thread
import sys
import os
import requests
from database import ChatDatabase

def load_simple_model():
    """Load DialoGPT-medium for instant no-train chat."""
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def load_trained_model(model_path="./my_finetuned_model"):
    """Load LoRA-finetuned DistilGPT2."""
    base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer

def hf_api_generate(prompt, session_id, max_new_tokens=150, temperature=0.7, db=None):
    """Free HF API call to DialoGPT-large with rate limit retry."""
    api_url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large"
    headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN', '')}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True,
            "return_full_text": False
        }
    }
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    db.reset_rate_limit(session_id) if db else None  # Success reset
                    return result[0].get('generated_text', '').strip()
            elif response.status_code == 503:  # Rate limit
                retries = db.increment_retry(session_id) if db else attempt + 1
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"🌐 API rate limited (retry {retries}/{max_retries}). Waiting {wait_time}s...")
                import time
                time.sleep(wait_time)
                continue
            else:
                return f"API error {response.status_code}: try --simple."
        except Exception as e:
            print(f"🌐 API attempt {attempt+1} error: {str(e)[:50]}")
            if attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt)
    return "Max retries exceeded (rate limited). Use /reset-rate or --simple."


def chat_loop(model, tokenizer, mode, db, session_id):
    print(f"\n🤖 {mode.upper()} MODE - Infinite persistent chats with 1024 token context!")
    print("Commands: /list, /projects, /new, /switch, /delete, /clear, /reset-rate, quit")

    
    history = db.get_session_history(session_id, 1024)
    print(f"Session {session_id} loaded with {len(history)} messages ({sum(m.get('tokens', 0) for m in history)} estimated tokens).")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Session saved. Goodbye!")
            break

        if user_input.startswith('/'):
            cmd = user_input[1:].split()
            if cmd[0] == 'list':
                sessions = db.list_sessions()
                print("\\n📋 Sessions (ID, Name, Msgs):")
                for sid, name, created, updated in sessions:
                    msg_cnt = db.get_message_count(sid)
                    print(f"  {sid}: '{name}' ({msg_cnt} msgs)")
            elif cmd[0] == 'projects':
                print("\\n📁 ALL PROJECTS/FILES:")
                print("my_ai_trainer/:")
                for item in sorted(os.listdir('my_ai_trainer')):
                    print(f"  📄 {item}")
                print("\\nDesktop:")
                for item in sorted(os.listdir('C:/Users/turtl/Desktop'))[:20]:  # Top 20
                    print(f"  📄 {item}")
                print("...")
            elif cmd[0] == 'new':
                new_name = ' '.join(cmd[1:]) or input("New session name: ").strip()
                if new_name:
                    session_id = db.get_or_create_session(new_name)
                    history = db.get_session_history(session_id)
                    print(f"✅ New session '{new_name}' ID: {session_id}")
            elif cmd[0] == 'switch':
                try:
                    session_id = int(cmd[1])
                    history = db.get_session_history(session_id)
                    print(f"✅ Switched to ID {session_id}, {len(history)} msgs")
                except:
                    print("❌ Invalid ID")
            elif cmd[0] == 'delete':
                try:
                    del_id = int(cmd[1])
                    if db.delete_session(del_id):
                        print(f"🗑️ Deleted session {del_id}")
                    else:
                        print("❌ Not found")
                except:
                    print("❌ Invalid ID")
            elif cmd[0] == 'clear':
                cursor = db.conn.cursor()
                cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                db.conn.commit()
                history = []
                db.reset_rate_limit(session_id)
                print("🧹 Session cleared + rate limit reset!")
            elif cmd[0] == 'reset-rate':
                db.reset_rate_limit(session_id)
                print(f"🔄 Rate limit reset for session {session_id}.")
                continue
            continue


        # Human message
        db.add_message(session_id, 'human', user_input)
        history.append({'role': 'human', 'content': user_input})

        # Build truncated prompt
        prompt_parts = []
        total_input_ids_len = 0
        MAX_CONTEXT = 1024
        for msg in reversed(history[-30:]):  # Recent priority
            msg_text = f"{msg['role'].capitalize()}: {msg['content']}\\n"
            test_ids = tokenizer.encode(msg_text, add_special_tokens=False)
            if total_input_ids_len + len(test_ids) > MAX_CONTEXT:
                break
            prompt_parts.append(msg_text)
            total_input_ids_len += len(test_ids)
        prompt = ''.join(reversed(prompt_parts)) + 'Assistant: '
        print(f"📏 Context: {total_input_ids_len} tokens")

        # Generate response
        response = ""
        if mode == 'api':
            response = hf_api_generate(prompt, session_id, db=db)

        elif mode == 'simple' and model:
            prompt_ids = tokenizer.encode(prompt, return_tensors='pt')
            with torch.no_grad():
                output_ids = model.generate(
                    prompt_ids, 
                    max_new_tokens=150, 
                    temperature=0.7, 
                    do_sample=True, 
                    pad_token_id=tokenizer.eos_token_id
                )
            response = tokenizer.decode(output_ids[0][prompt_ids.shape[1]:], skip_special_tokens=True).strip()
        elif mode == 'trained' and model:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        else:
            response = "Model not loaded."

        print(f"🤖 AI: {response}")

        # Save assistant msg
        db.add_message(session_id, 'assistant', response)
        history.append({'role': 'assistant', 'content': response})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infinite AI Chat with persistent DB, web API retries, rate limit reset per chat!")
    parser.add_argument('--trained', action='store_true', help="Local trained model")
    parser.add_argument('--simple', action='store_true', help="Local DialoGPT-medium")
    parser.add_argument('--api', action='store_true', help="HF API with auto-retry (default)")
    parser.add_argument('--web', action='store_true', help="Force web API mode with rate limits")

    args = parser.parse_args()

    db = ChatDatabase()
    session_name = input("Session name (default 'Main'): ").strip() or 'Main'
    session_id = db.get_or_create_session(session_name)
    print(f"Session '{session_name}' (ID {session_id}) ready!")

    try:
        if args.api:
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            chat_loop(None, tokenizer, 'api', db, session_id)
        elif args.trained:
            model, tokenizer = load_trained_model()
            chat_loop(model, tokenizer, 'trained', db, session_id)
        elif args.simple:
            model, tokenizer = load_simple_model()
            chat_loop(model, tokenizer, 'simple', db, session_id)
        else:
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            chat_loop(None, tokenizer, 'api', db, session_id)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

print("All changes complete! Run 'python my_ai_trainer/chat_ai.py --api' to test infinite chats with DB, 1024 context, /projects lists all files.")
