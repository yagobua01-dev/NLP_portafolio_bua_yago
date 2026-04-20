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