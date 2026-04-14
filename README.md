# 🚀 Infinite AI Chat Trainer - Full DB, 1024 Token Context, Projects List, Free HF API!

Remade for **infinite persistent chats** with **SQLite DB** (chats.db), **1024 token context window**, **/projects shows ALL files**, **free better AI via HF API**.

## 🎯 New Features
- **Persistent Chats**: History saved forever across runs. Multiple sessions.
- **Context Window**: Exactly 1024 tokens max (truncated smartly).
- **ALL PROJECTS ON SCREEN**: `/projects` lists my_ai_trainer/ + Desktop files.
- **Free Better AI**: `python chat_ai.py --api` (DialoGPT-large online, no key).
- **Commands**: `/list`, `/projects`, `/new`, `/switch`, `/delete`, `/clear`, `/reset-rate` (web rate limit reset).


## 🏃‍♂️ ELI5 Quick Start - Like I'm 5! (Default = FREE WEB CHAT)

**Default when you click `run_ai.bat`? FREE ONLINE AI** (like ChatGPT, connects to web brain, auto-fixes "too many asks" with waits/resets).

**3 Brains Explained Like Candy:**
1. **Web (--api/default)** 🍭 Free online super-smart (DialoGPT-large), but sometimes "wait 4 quota". Auto-retries + /reset-rate = new asks!
2. **Simple (--simple)** 🧁 Instant local small brain (DialoGPT-medium), no internet/train.
3. **Trained (--trained)** 🧠 YOUR brain! `run_ai.bat train` first (teach on convos, ~5min GPU), then super personal.

**Steps:**
- Click `run_ai.bat` → Web chat! Say hi.
- `/new MyChat` new talk, `quit`.
- Want yours? `run_ai.bat train` → `python chat_ai.py --trained`.

Saved forever in chats.db. /projects = see files!




**First run**: Enter session name e.g. "MyChat". Chats saved!

## 📱 Example Chat
```
You: /projects
AI: Lists all your files/projects!

You: Hello
AI: Hi! (context-aware)
quit
```

Next run: History loads automatically!

## 🛠 Train Your Own (Enhanced)
```
run_ai.bat train   # Merged full train (~5min GPU, hh-rlhf 5k samples, LoRA r=32)
python chat_ai.py --trained
```


## 🗄 Database (100+ LoC)
- `chats.db`: sessions + messages (role, content, tokens).
- Infinite history, auto-truncate to 1024 tokens.

## Tech
- Models: HF DialoGPT-large (--api), local DistilGPT2/LoRA.
- DB: SQLite full CRUD.
- Token truncate: Real tokenizer len.

**Your mini-GPT with persistence + file browser!** 🎉
