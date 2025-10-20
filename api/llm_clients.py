# -*- coding: utf-8 -*-
"""
Thin Ollama client for deepseek-r1:1.5b.

Defaults:
  OLLAMA_HOST=http://localhost:11434
"""

import os
import requests
import json  
from typing import Optional, Generator

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL_NAME = os.getenv("R1_MODEL", "deepseek-r1:1.5b")

def r1_generate(prompt: str, system: Optional[str] = None, stream: bool = False, timeout: int = 120) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt if not system else f"<<SYS>>\n{system}\n<</SYS>>\n\n{prompt}",
        "stream": stream,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_ctx": 4096,
        }
    }
    url = f"{OLLAMA_HOST}/api/generate"
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

def r1_generate_stream(prompt: str, system: Optional[str] = None, timeout: int = 120) -> Generator[str, None, None]:
    """Streaming generator. Yields text chunks from Ollama."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt if not system else f"<<SYS>>\n{system}\n<</SYS>>\n\n{prompt}",
        "stream": True,
        "options": {  # keep decoding consistent with non-stream
            "temperature": 0.2,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_ctx": 4096,
        }
    }
    url = f"{OLLAMA_HOST}/api/generate"
    with requests.post(url, json=payload, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("done"):
                break
            chunk = obj.get("response")
            if chunk:
                yield chunk
