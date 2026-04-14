import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer
import os
from dotenv import load_dotenv

load_dotenv()

# FULL MERGED TRAINING: Best of train.py + train_full.py (dotenv, wandb opt, trust_remote_code, tokenizing desc, my_finetuned_model_merged)
dataset_name = "Anthropic/hh-rlhf"
dataset = load_dataset(dataset_name, split="train[:5000]")  # Full power

def format_prompt(example):
    chosen = example.get('chosen', '')
    if 'human' in example:
        return f"Human: {example['human']}\\nAssistant: {chosen}"
    return f"Human: {chosen[:200]}...\\nAssistant: {chosen}"

dataset = dataset.map(lambda x: {"text": format_prompt(x)})

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
)

model = get_peft_model(model, lora_config)

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

tokenized_dataset = dataset.map(
    tokenize, 
    batched=True, 
    remove_columns=dataset.column_names,
    desc="Tokenizing"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./my_finetuned_model_merged",
    num_train_epochs=3,
    per_device_train_batch_size=8 if torch.cuda.is_available() else 2,
    gradient_accumulation_steps=8,
    warmup_steps=100,
    learning_rate=1e-4,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    save_steps=500,
    evaluation_strategy="no",
    report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
    dataloader_num_workers=0,  # Windows safe
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    max_seq_length=512,
)

trainer.train()

model.save_pretrained("./my_finetuned_model_merged")
tokenizer.save_pretrained("./my_finetuned_model_merged")
print("🚀 FULL MERGED Training complete! Use python chat_ai.py --trained")

