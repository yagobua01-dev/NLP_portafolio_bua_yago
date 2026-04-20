from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pyttsx3
from faster_whisper import WhisperModel

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