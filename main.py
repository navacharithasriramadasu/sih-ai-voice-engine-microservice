from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from pathlib import Path
import jwt
import wave
import json
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
from tts import generate_speech
from nlp_engine import nlp_engine
from typing import Optional

app = FastAPI(title="AgriConnect Custom Voice Assistant API")

# Load the extremely lightweight Vosk model into RAM on boot
print("Loading Vosk STT model (Edge AI)...")
try:
    if not os.path.exists("model"):
        print("Warning: Vosk model not found. Render will download it during build.")
        vosk_model = None
    else:
        vosk_model = Model("model")
        print("Vosk model loaded successfully! Memory usage is tiny.")
except Exception as e:
    print(f"Error loading Vosk: {e}")
    vosk_model = None

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
    return {"status": "Custom Edge Voice AI Engine Online (Vosk)"}

@app.post("/api/v1/voice/chat")
async def process_voice(
    request: Request,
    audio: UploadFile = File(...)
):
    """
    1. Extracts JWT to identify Farmer
    2. Receives audio from Flutter
    3. Converts audio to 16kHz Mono WAV using pydub
    4. Transcribes using Vosk (Edge STT)
    5. Runs Custom ML Pipeline (Intent + Entities + State)
    6. Calls NestJS Backend on behalf of Farmer
    7. Generates TTS audio response
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

    # 2. Save raw uploaded file
    input_audio_path = TEMP_DIR / f"input_{audio.filename}"
    with open(input_audio_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    try:
        print(f"Processing audio for {farmer_id}...")
        
        # 3. Fast Audio Conversion for Vosk (16kHz Mono WAV)
        print("Converting audio format...")
        wav_path = TEMP_DIR / f"converted_{audio.filename}.wav"
        audio_segment = AudioSegment.from_file(str(input_audio_path))
        audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
        audio_segment.export(str(wav_path), format="wav")
        
        # 4. STT (Speech to Text) using offline Vosk
        transcribed_text = ""
        if vosk_model:
            print("Transcribing with Vosk...")
            wf = wave.open(str(wav_path), "rb")
            rec = KaldiRecognizer(vosk_model, wf.getframerate())
            
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data)
                
            result = json.loads(rec.FinalResult())
            transcribed_text = result.get("text", "").strip()
        else:
            transcribed_text = "I want to sell 100 kg of tomato."
            
        print(f"Transcribed Text: {transcribed_text}")
        
        # We enforce English for this hackathon demo with the small-en model
        detected_lang = "en"

        # 5. Custom AI Inference & NestJS Integration
        agent_response = nlp_engine.process_message(farmer_id, transcribed_text, token)
        print(f"Agent Reply: {agent_response}")

        # 6. TTS (Text to Speech)
        output_audio_path = TEMP_DIR / f"output_{audio.filename}.mp3"
        generate_speech(agent_response, detected_lang, str(output_audio_path))
        
        # Clean up temp files to save disk space
        os.remove(input_audio_path)
        os.remove(wav_path)

        return FileResponse(str(output_audio_path), media_type="audio/mpeg")

    except Exception as e:
        print(f"Error processing voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup input (keep output for a bit if needed)
        if 'input_audio_path' in locals() and input_audio_path.exists():
            os.remove(input_audio_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
