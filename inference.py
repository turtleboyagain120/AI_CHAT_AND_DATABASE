import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load fine-tuned model
model_path = "./my_finetuned_model"
try:
    base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
except Exception as e:
    print(f"Model load error: {e}. Run 'python train.py' first.")
    exit(1)

def generate_response(history, user_input):
    # Multi-turn: last 4 exchanges
    context = "\n".join(history[-8:]) + f"\nHuman: {user_input}\nAssistant:"
    inputs = tokenizer(context, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return full_response

if __name__ == "__main__":
    history = []
    print("🤖 Trained AI Chatbot (hh-rlhf data)! 'quit' to exit, 'clear' to reset.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            print("👋 Bye!")
            break
        if user_input.lower() == 'clear':
            history = []
            print("History cleared.")
            continue
        print("AI: ", end="", flush=True)
        response = generate_response(history, user_input)
        # Print char-by-char for streaming effect
        for char in response:
            print(char, end="", flush=True)
        print("\n")
        history.append(f"Human: {user_input}")
        history.append(f"Assistant: {response}")
