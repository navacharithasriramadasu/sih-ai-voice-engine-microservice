from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from pathlib import Path

import jwt
import whisper
from tts import generate_speech
from nlp_engine import nlp_engine
from typing import Optional

app = FastAPI(title="AgriConnect Custom Voice Assistant API")

# Load the extremely lightweight 'tiny' model into RAM on boot
print("Loading Whisper STT model...")
try:
    whisper_model = whisper.load_model("tiny")
    print("Whisper model loaded!")
except Exception as e:
    print(f"Error loading whisper: {e}")
    whisper_model = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = Path("./temp")
TEMP_DIR.mkdir(exist_ok=True)

NESTJS_JWT_SECRET = "fallback-secret-for-dev"

@app.get("/")
def health_check():
    return {"status": "Custom Voice AI Engine Online"}

@app.post("/api/v1/voice/chat")
async def process_voice(
    request: Request,
    audio: UploadFile = File(...)
):
    """
    1. Extracts JWT to identify Farmer
    2. Receives audio from React Native
    3. Transcribes using Whisper (Speech-to-Text)
    4. Runs Custom ML Pipeline (Intent + Entities + State)
    5. Calls NestJS Backend on behalf of Farmer
    6. Generates TTS audio response
    """
    
    # 1. JWT Authentication
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or Invalid Authorization Header")
        
    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, NESTJS_JWT_SECRET, algorithms=["HS256"])
        farmer_id = payload.get("sub")
        if not farmer_id:
            raise ValueError("No user ID in token")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid Token: {str(e)}")

    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")

    # 2. Save uploaded file
    input_audio_path = TEMP_DIR / f"input_{audio.filename}"
    with open(input_audio_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    try:
        print(f"Processing audio for {farmer_id}...")
        
        # 3. STT (Speech to Text) using self-hosted Whisper
        if whisper_model:
            print("Transcribing with Whisper...")
            result = whisper_model.transcribe(str(input_audio_path))
            transcribed_text = result["text"].strip()
            detected_lang = result.get("language", "en")
        else:
            transcribed_text = "I want to sell 100 kg of tomato."
            detected_lang = "en"
            
        print(f"Transcribed [{detected_lang}]: {transcribed_text}")

        # 4. Custom AI Inference & NestJS Integration
        # We pass the raw token so the AI can securely make HTTP requests on their behalf
        agent_response = nlp_engine.process_message(farmer_id, transcribed_text, token)
        print(f"Agent Reply: {agent_response}")

        # 5. TTS (Text to Speech)
        output_audio_path = TEMP_DIR / f"output_{audio.filename}.mp3"
        generate_speech(agent_response, detected_lang, str(output_audio_path))

        # Return audio file to frontend
        return FileResponse(
            path=output_audio_path,
            media_type="audio/mpeg",
            filename="response.mp3"
        )
    except Exception as e:
        print(f"Error processing voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup input (keep output for a bit if needed)
        if input_audio_path.exists():
            os.remove(input_audio_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
