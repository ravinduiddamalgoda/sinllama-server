#!/usr/bin/env python3
"""
Sinhala TTS Server (Standalone)
===============================
Runs the VITS TTS model as a separate microservice on port 8001.
Requires Python 3.11 with its own venv (tts_env).

Endpoints:
  POST /synthesize       - JSON body: {"text": "...", "speaker": "oshadi"}  → WAV audio
  GET  /synthesize       - Query params: ?text=...&speaker=oshadi           → WAV audio
  GET  /speakers         - List available speakers
  GET  /health           - Health check

Usage:
  source /workspace/tts_env/bin/activate
  python tts_server.py
  python tts_server.py --port 8001
  python tts_server.py --model best_model.pth --config config.json
"""

import os
import sys
import io
import time
import logging
import argparse
import numpy as np
import torch
import soundfile as sf
from scipy.signal import resample
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Query, HTTPException, Depends, Security
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import uvicorn

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("sinhala-tts")

# ── Configuration ────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "best_model.pth")
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")

# ── Authentication ───────────────────────────────────────────────────
API_KEY = os.environ.get("SINLLAMA_API_KEY", "sinllama-default-key-change-me")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: Optional[str] = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key. Pass X-API-Key header.",
        )
    return key


# ── Globals (loaded once at startup) ─────────────────────────────────
synthesizer = None
is_multi_speaker = False
speaker_names: list[str] = []
sample_rate = 22050

# Will be set from CLI args or defaults
_model_path: str = DEFAULT_MODEL_PATH
_config_path: str = DEFAULT_CONFIG_PATH


# ── Pydantic models ─────────────────────────────────────────────────
class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="Romanized Sinhala text to synthesize")
    speaker: Optional[str] = Field(default=None, description="Speaker name (for multi-speaker models)")
    speed: Optional[float] = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier")


# ── Model Loading ───────────────────────────────────────────────────
def load_model(model_path: str, config_path: str):
    """Load the TTS model into the global synthesizer."""
    global synthesizer, is_multi_speaker, speaker_names, sample_rate

    from TTS.utils.synthesizer import Synthesizer as TTSSynthesizer

    logger.info(f"Loading model: {model_path}")
    logger.info(f"Config: {config_path}")

    synthesizer = TTSSynthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        use_cuda=torch.cuda.is_available(),
    )

    sample_rate = synthesizer.tts_config.audio.sample_rate

    # Detect speakers
    if (
        hasattr(synthesizer.tts_model, "num_speakers")
        and synthesizer.tts_model.num_speakers > 1
    ):
        is_multi_speaker = True
        if (
            hasattr(synthesizer.tts_model, "speaker_manager")
            and synthesizer.tts_model.speaker_manager
        ):
            sm = synthesizer.tts_model.speaker_manager
            if hasattr(sm, "name_to_id"):
                speaker_names = list(sm.name_to_id.keys())
            elif hasattr(sm, "speaker_names"):
                speaker_names = list(sm.speaker_names)
        if not speaker_names:
            speaker_names = ["oshadi", "mettananda"]  # fallback
    else:
        is_multi_speaker = False
        speaker_names = []

    device = "CUDA" if torch.cuda.is_available() else "CPU"
    logger.info(f"Model loaded on {device} | Multi-speaker: {is_multi_speaker} | Speakers: {speaker_names}")
    logger.info(f"Sample rate: {sample_rate}")


# ── Lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load TTS model on startup."""
    logger.info("=" * 60)
    logger.info("🔊 Starting Sinhala TTS Server")
    logger.info("=" * 60)

    if not os.path.exists(_model_path):
        logger.error(f"Model file not found: {_model_path}")
        logger.error("Specify --model and --config, or place best_model.pth and config.json next to this script.")
        sys.exit(1)
    if not os.path.exists(_config_path):
        logger.error(f"Config file not found: {_config_path}")
        sys.exit(1)

    start = time.time()
    load_model(_model_path, _config_path)
    elapsed = time.time() - start

    logger.info(f"✅ TTS model ready in {elapsed:.1f}s")
    logger.info("=" * 60)

    yield

    logger.info("🧹 Shutting down TTS server...")
    del synthesizer
    torch.cuda.empty_cache()
    logger.info("👋 TTS server stopped.")


# ── FastAPI app ──────────────────────────────────────────────────────
app = FastAPI(
    title="Sinhala TTS API",
    description="Sinhala Text-to-Speech API powered by VITS",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper ───────────────────────────────────────────────────────────
DEFAULT_SPEED = 0.85  # < 1.0 = slower, > 1.0 = faster


def _synthesize(text: str, speaker: str = None, speed: float = DEFAULT_SPEED) -> io.BytesIO:
    """Run TTS and return WAV bytes in a BytesIO buffer."""
    global synthesizer, is_multi_speaker, sample_rate

    if synthesizer is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded.")

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    speaker_name = None
    if is_multi_speaker:
        speaker_name = speaker or "oshadi"
        if speaker_name not in speaker_names:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown speaker '{speaker_name}'. Available: {speaker_names}",
            )

    t0 = time.time()
    wav = synthesizer.tts(text=text, speaker_name=speaker_name)
    elapsed = time.time() - t0
    logger.info(f"Synthesized {len(text)} chars in {elapsed:.2f}s | speaker={speaker_name}")

    wav_array = np.array(wav, dtype=np.float32)

    # Apply speed adjustment via resampling (pitch is preserved via sample_rate trick)
    if speed != 1.0:
        new_length = int(len(wav_array) / speed)
        wav_array = resample(wav_array, new_length).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, wav_array, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok" if synthesizer is not None else "loading",
        "model_loaded": synthesizer is not None,
        "multi_speaker": is_multi_speaker,
        "speakers": speaker_names,
        "sample_rate": sample_rate,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.get("/speakers")
def get_speakers():
    return {
        "speakers": speaker_names,
        "default": "oshadi" if speaker_names else None,
        "multi_speaker": is_multi_speaker,
    }


@app.post(
    "/synthesize",
    dependencies=[Depends(verify_api_key)],
    summary="Text-to-Speech (POST)",
    description="Synthesize Sinhala speech from romanized text. Returns WAV audio.",
)
def synthesize_post(req: SynthesizeRequest):
    """Synthesize speech from text (JSON body)."""
    buf = _synthesize(req.text, req.speaker, req.speed or DEFAULT_SPEED)
    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"},
    )


@app.get(
    "/synthesize",
    dependencies=[Depends(verify_api_key)],
    summary="Text-to-Speech (GET)",
    description="Synthesize Sinhala speech from romanized text via query params. Returns WAV audio.",
)
def synthesize_get(
    text: str = Query(..., description="Romanized Sinhala text to synthesize"),
    speaker: Optional[str] = Query(None, description="Speaker name (for multi-speaker models)"),
    speed: float = Query(DEFAULT_SPEED, ge=0.5, le=2.0, description="Speech speed multiplier"),
):
    """Synthesize speech from text (query parameters)."""
    buf = _synthesize(text, speaker, speed)
    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"},
    )


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sinhala TTS FastAPI Server")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to model checkpoint (.pth)")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to config.json")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001, help="Port (default: 8001)")
    args = parser.parse_args()

    _model_path = args.model
    _config_path = args.config

    logger.info(f"Starting TTS server on {args.host}:{args.port}")
    logger.info(f"API docs: http://{args.host}:{args.port}/docs")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
