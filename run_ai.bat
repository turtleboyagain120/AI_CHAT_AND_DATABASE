@echo off
cd /d "C:/Users/turtl/Desktop/my_ai_trainer"
echo Starting enhanced AI chatbot with web API ^(retries/rate reset per chat^), merged training.
echo Modes: --api/web (default), --trained, --simple, or python train.py first.
if "%1"=="train" (
    python train.py
    pause
    exit /b
)
python chat_ai.py --api %*
pause


