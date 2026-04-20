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