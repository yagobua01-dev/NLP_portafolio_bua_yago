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
    
def ensure_file_exists(path: str) -> Path:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Audio file not found: {p}")
    return p