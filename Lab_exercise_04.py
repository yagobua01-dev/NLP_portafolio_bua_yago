from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pyttsx3
from faster_whisper import WhisperModel
import speech_recognition as sr
from gtts import gTTS

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore
 
# Check if the audio file exists in the given path  
def ensure_file_exists(path: str) -> Path:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Audio file not found: {p}")
    return p

# Print results in a clear format
def pretty_print(title: str, payload: Any) -> None:
    print(f"\n=== {title} ===")
    if isinstance(payload, (dict, list)):
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(payload)
        
# Create OpenAI client using API key
def build_openai_client():
    if OpenAI is None:
        raise RuntimeError("Install openai package first.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    return OpenAI()

# 1. Local Speech to Text
# Convert speech to text using a local Whisper model
def local_speech_to_text(audio_path: str) -> Dict[str, Any]:
    audio_file = ensure_file_exists(audio_path)

    # Load local model
    model = WhisperModel("base", device="cpu", compute_type="int8")

    # Transcribe audio
    segments, info = model.transcribe(audio_file)

    text_parts = []

    # Collect all text parts
    for seg in segments:
        text_parts.append(seg.text.strip())

    return {
        "task": "local_stt",
        "text": " ".join(text_parts)
    }

# 2. Local Text to Speech
# Convert text to speech using local engine (offline)
def local_text_to_speech(text: str) -> None:
    engine = pyttsx3.init()

    # Set speaking speed
    engine.setProperty("rate", 180)

    # Speak the text
    engine.say(text)
    engine.runAndWait()
    
# 3. API Speech to Text
# Convert speech to text using OpenAI API
def api_speech_to_text(audio_path: str) -> Dict[str, Any]:
    audio_file = ensure_file_exists(audio_path)
    client = build_openai_client()

    # Send audio file to API
    with open(audio_file, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f
        )

    return {
        "task": "api_stt",
        "text": transcript.text
    }
    
# 4. API Text to Speech
# Convert text to speech using OpenAI API
def api_text_to_speech(text: str, output_file: str) -> None:
    client = build_openai_client()

    # Generate speech and save it to a file
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=text
    ) as response:
        response.stream_to_file(output_file)
        
# 5. Translation
# Translate audio speech into English text
def translate_audio_to_english(audio_path: str) -> Dict[str, Any]:
    audio_file = ensure_file_exists(audio_path)
    client = build_openai_client()

    # Send audio to translation model
    with open(audio_file, "rb") as f:
        result = client.audio.translations.create(
            model="whisper-1",
            file=f
        )

    return {
        "task": "translation",
        "translated_text": result.text
    }

# Free API STT
def free_api_speech_to_text(audio_path: str):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
        return {
            "task": "free_api_stt",
            "text": text
        }
    except sr.UnknownValueError:
        return {"task": "free_api_stt", "error": "Could not understand audio"}
    except sr.RequestError:
        return {"task": "free_api_stt", "error": "API unavailable"}


# Main program
# Read command from terminal and run correct function
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("command")  # command to execute
    parser.add_argument("input")    # input text or audio file

    args = parser.parse_args()

    # Select task based on command
    if args.command == "local-stt":
        result = local_speech_to_text(args.input)
        pretty_print("LOCAL STT", result)

    elif args.command == "local-tts":
        local_text_to_speech(args.input)

    elif args.command == "api-stt":
        result = api_speech_to_text(args.input)
        pretty_print("API STT", result)

    elif args.command == "api-tts":
        api_text_to_speech(args.input, "output.mp3")
        print("Audio saved as output.mp3")

    elif args.command == "translate-audio":
        result = translate_audio_to_english(args.input)
        pretty_print("TRANSLATION", result)
        
    elif args.command == "free-api-stt":
        result = free_api_speech_to_text(args.input)
        pretty_print("FREE API STT", result)

    else:
        print("Unknown command")
        
# Run the program
if __name__ == "__main__":
    main()