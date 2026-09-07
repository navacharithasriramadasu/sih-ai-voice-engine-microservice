from gtts import gTTS
import os

def generate_speech(text: str, lang: str, output_path: str):
    """
    Generates an MP3 audio file using Google TTS.
    Supports English ('en'), Hindi ('hi'), Telugu ('te').
    """
    print(f"Generating TTS for: {text[:30]}... in {lang}")
    
    # Fallback to English if language is not perfectly detected or supported
    safe_lang = lang if lang in ['hi', 'te', 'en'] else 'en'
    
    try:
        tts = gTTS(text=text, lang=safe_lang, slow=False)
        tts.save(output_path)
    except Exception as e:
        print(f"TTS Error: {e}. Generating English fallback.")
        tts = gTTS(text="Sorry, I am facing a technical issue.", lang="en")
        tts.save(output_path)
