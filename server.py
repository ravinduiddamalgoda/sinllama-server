#!/usr/bin/env python3
"""
SinLlama API Server — 2-GPU Version
Runs on Vast.ai (2x RTX 5060 Ti), serves Sinhala text generation via REST API.
Model is sharded across both GPUs using device_map='auto'.
"""

import torch
import time
import logging
import os
import io
from fastapi import FastAPI, HTTPException, Depends, Security, UploadFile, File, Form, Query
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import Optional
import tempfile
import shutil
import json as _json
import numpy as np
import torchaudio
import httpx

# === Configuration ===
BASE_MODEL_PATH = "/workspace/models/llama-3-8b-base"
ADAPTER_PATH = "/workspace/models/sinllama-adapter"
TOKENIZER_PATH = "/workspace/models/sinhala-tokenizer"
VOCAB_SIZE = 139336

# TTS Microservice URL (tts_server.py runs separately on port 8001)
TTS_SERVICE_URL = os.environ.get("TTS_SERVICE_URL", "http://127.0.0.1:8001")

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("sinllama")

# === Authentication ===
API_KEY = os.environ.get("SINLLAMA_API_KEY", "sinllama-default-key-change-me")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: Optional[str] = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key. Pass X-API-Key header.",
        )
    return key


# === Global References ===
model = None
tokenizer = None
input_device = None  # device where input tensors should be placed
asr_pipe = None  # Whisper ASR pipeline
diar_pipeline = None  # pyannote speaker diarization pipeline


# === Model Loading ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model, tokenizer, input_device, asr_pipe, diar_pipeline

    # Set memory config for better allocation across 2 GPUs
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

    # Enable CUDA optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    logger.info("=" * 60)
    logger.info("🚀 Starting SinLlama API Server (2-GPU Mode)")
    logger.info("=" * 60)

    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        logger.info(f"   GPU {i}: {name} ({total:.1f} GB)")

    start = time.time()

    # --- 4-bit Quantization Config ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # --- Step 1: Load Base Model (shard across both GPUs) ---
    logger.info("📦 Loading Meta-Llama-3-8B (4-bit, sharded across 2 GPUs)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "14GiB", 1: "14GiB", "cpu": "64GiB"},
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # use "flash_attention_2" if flash-attn is installed
    )

    # Log device map summary
    dm = getattr(base_model, "hf_device_map", {})
    gpu0_layers = sum(1 for v in dm.values() if str(v) == "0")
    gpu1_layers = sum(1 for v in dm.values() if str(v) == "1")
    logger.info(f"   Device map: GPU 0 has {gpu0_layers} modules, GPU 1 has {gpu1_layers} modules")

    for i in range(num_gpus):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        logger.info(f"   GPU {i} VRAM after base model: {alloc:.2f} GB")

    # --- Step 2: Load Extended Sinhala Tokenizer ---
    logger.info("📝 Loading Extended Sinhala tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    base_model.resize_token_embeddings(VOCAB_SIZE)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"   Tokenizer loaded. Vocab size: {len(tokenizer)}")

    # --- Step 3: Apply SinLlama LoRA Adapter & Merge ---
    logger.info("🔗 Applying SinLlama LoRA adapter...")
    peft_model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

    # Merge LoRA weights into base model — eliminates CPU↔GPU transfer overhead
    logger.info("🔀 Merging LoRA weights into base model for faster inference...")
    model = peft_model.merge_and_unload()
    model.eval()

    for i in range(num_gpus):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        logger.info(f"   GPU {i} VRAM after merge: {alloc:.2f} GB")

    # --- Step 4: Determine input device ---
    # Find the first CUDA device in the model's device map
    input_device = torch.device("cuda:0")  # default fallback
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for _, d in model.hf_device_map.items():
            if isinstance(d, (int, str)):
                dev_str = str(d)
                if dev_str.isdigit():
                    input_device = torch.device(f"cuda:{dev_str}")
                    break
                elif dev_str.startswith("cuda"):
                    input_device = torch.device(dev_str)
                    break

    # Warmup: run a dummy forward pass to initialize CUDA kernels and KV cache
    logger.info("🔥 Running warmup inference...")
    warmup_ids = tokenizer("warmup", return_tensors="pt").input_ids.to(input_device)
    with torch.no_grad():
        model.generate(warmup_ids, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    del warmup_ids
    torch.cuda.empty_cache()
    logger.info("   Warmup complete.")

    elapsed = time.time() - start

    logger.info("=" * 60)
    logger.info(f"✅ Model ready in {elapsed:.1f} seconds")
    logger.info(f"   Input device: {input_device}")
    for i in range(num_gpus):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        logger.info(f"   GPU {i}: {alloc:.2f} / {total:.1f} GB used")
    logger.info(
        f"   API Key: {'Custom' if API_KEY != 'sinllama-default-key-change-me' else '⚠️  DEFAULT (change this!)'}"
    )
    logger.info("=" * 60)

    # --- Step 5: Load Whisper ASR Pipeline (Sinhala) ---
    logger.info("🎙️ Loading Whisper Small Sinhala ASR pipeline...")
    from transformers import pipeline as hf_pipeline
    # Place Whisper on GPU 1 to avoid competing with LLM on GPU 0
    asr_device = 1 if num_gpus > 1 else 0
    asr_pipe = hf_pipeline(
        "automatic-speech-recognition",
        model="Lingalingeswaran/whisper-small-sinhala",
        device=asr_device,
        torch_dtype=torch.float16,
    )
    logger.info(f"   Whisper ASR loaded on GPU {asr_device}")
    for i in range(num_gpus):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        logger.info(f"   GPU {i}: {alloc:.2f} / {total_mem:.1f} GB used (after ASR)")

    # --- Step 6: Load pyannote Speaker Diarization Pipeline ---
    logger.info("🗣️ Loading pyannote speaker diarization pipeline...")
    try:
        from pyannote.audio import Pipeline as PyannotePipeline

        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            logger.warning("   ⚠️ HF_TOKEN env var not set. Diarization may fail if models aren't cached.")

        diar_pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token if hf_token else None,
        )
        # Place on GPU 1 alongside Whisper to keep GPU 0 free for LLM
        diar_device = 1 if num_gpus > 1 else 0
        diar_pipeline.to(torch.device(f"cuda:{diar_device}"))
        logger.info(f"   Diarization pipeline loaded on GPU {diar_device}")
        for i in range(num_gpus):
            alloc = torch.cuda.memory_allocated(i) / 1e9
            total_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"   GPU {i}: {alloc:.2f} / {total_mem:.1f} GB used (after diarization)")
    except Exception as e:
        logger.error(f"   ❌ Failed to load diarization pipeline: {e}")
        logger.error("   Diarization endpoint (/diarize) will be unavailable.")
        logger.error("   Install: pip install pyannote.audio")
        logger.error("   Set HF_TOKEN env var with your HuggingFace token.")
        diar_pipeline = None

    logger.info(f"📡 TTS microservice expected at: {TTS_SERVICE_URL}")

    yield  # Server runs here

    # Cleanup
    logger.info("🧹 Shutting down, releasing GPU memory...")
    del model, tokenizer, asr_pipe, diar_pipeline
    torch.cuda.empty_cache()
    logger.info("👋 Server stopped.")


# === FastAPI App ===
app = FastAPI(
    title="SinLlama API",
    description="Sinhala LLM inference API for IoT child conversation toy (2-GPU)",
    version="1.1.0",
    lifespan=lifespan,
)


# === Request/Response Schemas ===
class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="Romanized Sinhala text to synthesize")
    speaker: Optional[str] = Field(default=None, description="Speaker name (for multi-speaker models)")
    speed: Optional[float] = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt")
    max_new_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0, le=200)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)


class GenerateResponse(BaseModel):
    generated_text: str
    tokens_generated: int
    inference_time_ms: float


class AskRequest(BaseModel):
    message: str = Field(..., description="User message / question")
    system_prompt: str = Field(
        default="ඔබ සිංහල භාෂාවෙන් උත්තර දෙන බුද්ධිමත් සහකාරයෙකි. කෙටිව සහ පැහැදිලිව උත්තර දෙන්න. එක හෝ දෙක වාක්‍ය පමණක් භාවිතා කරන්න.",
        description="System prompt to guide the model's behaviour",
    )
    conversation_history: list[dict] = Field(
        default=[],
        description='Previous turns as [{"role":"user"|"assistant","content":"..."}]',
    )
    max_new_tokens: int = Field(default=80, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0, le=200)
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0)


class AskResponse(BaseModel):
    reply: str
    tokens_generated: int
    inference_time_ms: float


class ChildChatRequest(BaseModel):
    child_utterance: str = Field(..., description="What the child said (ASR transcription from Raspberry Pi)")
    conversation_history: list[str] = Field(default=[], description="Previous turns")
    child_name: str = Field(default="ළමයා", description="Child's name")
    target_phoneme: str = Field(default="", description="Target phoneme for practice")
    current_word: str = Field(default="", description="The Sinhala word currently being practiced (e.g. 'බල්ලා')")
    pronunciation_result: str = Field(
        default="",
        description="ASR pronunciation result: 'correct', 'close', 'retry', 'no_speech', 'skipped', or 'new_word' (introducing a new word)",
    )
    attempt_number: int = Field(default=1, ge=1, le=10, description="Which attempt this is for the current word")
    session_words: list[str] = Field(default=[], description="All words in this session (e.g. ['බල්ලා','පූසා','අලියා'])")
    next_word: str = Field(default="", description="The next word to move to (sent when current word is correct or skipped)")
    words_completed: int = Field(default=0, ge=0, description="Number of words completed so far in this session")
    is_session_complete: bool = Field(default=False, description="True when all 3 words are done")


class ChildChatResponse(BaseModel):
    response: str
    inference_time_ms: float


class TranscribeResponse(BaseModel):
    text: str
    language: str = "si"
    inference_time_ms: float


class ChildSessionResponse(BaseModel):
    """Combined ASR + child-chat response returned by /child-session."""
    transcribed_text: str = Field(description="What Whisper heard the child say")
    teacher_response: str = Field(description="Teacher's conversational reply")
    asr_time_ms: float = Field(description="Time taken for speech-to-text (ms)")
    chat_time_ms: float = Field(description="Time taken for response generation (ms)")
    total_time_ms: float = Field(description="Total end-to-end time (ms)")


class WordScore(BaseModel):
    """Per-word LM probability score indicating how meaningful/expected a word is."""
    word: str = Field(description="The word (after correction if corrected)")
    original_word: str = Field(description="Original ASR word before correction (same as word if not corrected)")
    probability: float = Field(description="LM probability score (0.0-1.0). Higher = more expected/meaningful")
    is_meaningful: bool = Field(description="Whether the word passed tokenizer validation as a real Sinhala word")
    was_corrected: bool = Field(description="Whether this word was corrected by the transformer")


class DiarizeSegment(BaseModel):
    """A single speaker-labeled transcript segment with both raw and AI-corrected text."""
    speaker: str = Field(description="Speaker label (e.g. SPEAKER_00)")
    start: float = Field(description="Segment start time in seconds")
    end: float = Field(description="Segment end time in seconds")
    duration: float = Field(description="Segment duration in seconds")
    asr_text: str = Field(description="Original Whisper ASR transcription (raw, uncorrected)")
    ai_corrected_text: str = Field(description="LLM-corrected transcription (AI-optimized)")
    text: str = Field(description="Best available text (ai_corrected_text if available, else asr_text)")
    word_scores: list[WordScore] = Field(default=[], description="Per-word probability scores showing which words are meaningful/garbled")
    segment_accuracy_pct: float = Field(default=0.0, description="Percentage of words in this segment that are meaningful (probability > 0.01)")
    total_words: int = Field(default=0, description="Total number of words in this segment")
    meaningful_words: int = Field(default=0, description="Number of words that passed as meaningful")
    corrected_words: int = Field(default=0, description="Number of words corrected by transformer")


class OverlappingRegion(BaseModel):
    """A region where multiple speakers are talking simultaneously."""
    start: float = Field(description="Overlap start time in seconds")
    end: float = Field(description="Overlap end time in seconds")
    duration: float = Field(description="Overlap duration in seconds")
    speakers: list[str] = Field(description="Speakers involved in this overlap")


class DiarizeResponse(BaseModel):
    """Speaker diarization + AI-optimized transcription response."""
    num_speakers: int = Field(description="Number of distinct speakers detected")
    speakers: list[str] = Field(description="List of speaker labels")
    total_segments: int = Field(description="Total number of transcript segments")
    segments: list[DiarizeSegment] = Field(description="Speaker-labeled segments with both raw ASR and AI-corrected text")
    full_audio_transcript: str = Field(description="Full-audio Whisper ASR transcript (used as reference for optimization)")
    overlapping_regions: list[OverlappingRegion] = Field(default=[], description="Regions where speakers overlap")
    transcript_text: str = Field(description="AI-optimized full transcript with speaker labels, one line per segment")
    raw_transcript_text: str = Field(description="Raw ASR full transcript with speaker labels, one line per segment")
    audio_duration_s: float = Field(description="Total audio duration in seconds")
    diarization_time_ms: float = Field(description="Time for speaker diarization (ms)")
    transcription_time_ms: float = Field(description="Time for full-audio + per-segment ASR (ms)")
    ai_optimization_time_ms: float = Field(description="Time for LLM-based transcript correction (ms)")
    total_time_ms: float = Field(description="Total end-to-end processing time (ms)")
    overall_accuracy_pct: float = Field(default=0.0, description="Overall percentage of words across all segments that are meaningful")
    total_words: int = Field(default=0, description="Total words across all segments")
    total_meaningful_words: int = Field(default=0, description="Total meaningful words across all segments")
    total_corrected_words: int = Field(default=0, description="Total words corrected by transformer across all segments")


# === Helpers ===

import re


# === Diarization Helper Functions ===

def _detect_overlapping_regions(diar_segments: list[dict]) -> list[OverlappingRegion]:
    """Detect time regions where multiple speakers talk simultaneously.
    
    Args:
        diar_segments: list of {"speaker": str, "start": float, "end": float}
    
    Returns:
        list of OverlappingRegion objects
    """
    if len(diar_segments) < 2:
        return []

    overlaps: list[OverlappingRegion] = []
    for i in range(len(diar_segments)):
        for j in range(i + 1, len(diar_segments)):
            seg_a = diar_segments[i]
            seg_b = diar_segments[j]

            # Skip if same speaker
            if seg_a["speaker"] == seg_b["speaker"]:
                continue

            # Calculate overlap
            overlap_start = max(seg_a["start"], seg_b["start"])
            overlap_end = min(seg_a["end"], seg_b["end"])

            if overlap_start < overlap_end:
                duration = round(overlap_end - overlap_start, 3)
                if duration >= 0.2:  # Only flag overlaps >= 200ms
                    overlaps.append(OverlappingRegion(
                        start=round(overlap_start, 3),
                        end=round(overlap_end, 3),
                        duration=duration,
                        speakers=sorted(set([seg_a["speaker"], seg_b["speaker"]])),
                    ))

    # Merge adjacent/overlapping overlap regions
    if overlaps:
        overlaps.sort(key=lambda x: x.start)
        merged = [overlaps[0]]
        for ovl in overlaps[1:]:
            prev = merged[-1]
            if ovl.start <= prev.end:
                # Merge
                merged[-1] = OverlappingRegion(
                    start=prev.start,
                    end=max(prev.end, ovl.end),
                    duration=round(max(prev.end, ovl.end) - prev.start, 3),
                    speakers=sorted(set(prev.speakers + ovl.speakers)),
                )
            else:
                merged.append(ovl)
        overlaps = merged

    return overlaps


def _align_chunks_to_speakers(
    whisper_chunks: list[dict],
    diar_segments: list[dict],
) -> list[dict]:
    """Align Whisper timestamp chunks to diarization speaker segments.
    
    For each diarization segment, find all Whisper chunks whose midpoint
    falls within that segment's time range, and concatenate their text.
    
    Args:
        whisper_chunks: [{"text": str, "timestamp": (start, end)}, ...]
        diar_segments: [{"speaker": str, "start": float, "end": float}, ...]
    
    Returns:
        list of {"speaker": str, "start": float, "end": float, "text": str}
    """
    aligned = []

    for seg in diar_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        matched_texts = []

        for chunk in whisper_chunks:
            ts = chunk.get("timestamp")
            if ts is None or ts[0] is None or ts[1] is None:
                continue
            chunk_start, chunk_end = ts
            # Use chunk midpoint for assignment
            chunk_mid = (chunk_start + chunk_end) / 2.0
            if seg_start <= chunk_mid <= seg_end:
                text = chunk.get("text", "").strip()
                if text:
                    matched_texts.append(text)

        aligned_text = " ".join(matched_texts).strip()
        aligned.append({
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "text": aligned_text,
        })

    return aligned


def _llm_generate_internal(
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
) -> str:
    """Run LLM inference directly (internal helper, no HTTP overhead).
    
    Returns the generated text, or empty string if model isn't loaded.
    """
    if model is None or tokenizer is None:
        return ""

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=50,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_length:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    except Exception as e:
        logger.warning(f"LLM internal generation failed: {e}")
        return ""


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def _word_similarity(w1: str, w2: str) -> float:
    """Return similarity score 0.0-1.0 between two words using normalized edit distance."""
    if not w1 or not w2:
        return 0.0
    if w1 == w2:
        return 1.0
    max_len = max(len(w1), len(w2))
    dist = _levenshtein_distance(w1, w2)
    return 1.0 - (dist / max_len)


def _word_level_fuse(text_a: str, text_b: str) -> str:
    """Fuse two ASR outputs word-by-word, picking the best word from each.
    
    text_a = per-segment ASR (isolated segment, may be garbled)
    text_b = aligned full-audio ASR (has more context, fewer chunks)
    
    Strategy:
    - Align words by position
    - For similar words (>0.5 similarity), pick the longer/more complete one
    - For very different words, prefer text_b (full-audio context is better)
    - Keep the word count close to the shorter text to avoid inflation
    """
    if not text_a and not text_b:
        return ""
    if not text_a:
        return text_b
    if not text_b:
        return text_a

    words_a = text_a.split()
    words_b = text_b.split()

    # If one is empty, return the other
    if not words_a:
        return text_b
    if not words_b:
        return text_a

    # Use dynamic programming to align the two word sequences
    # Simple approach: iterate through both, matching similar words
    result = []
    i, j = 0, 0

    while i < len(words_a) and j < len(words_b):
        wa = words_a[i]
        wb = words_b[j]
        sim = _word_similarity(wa, wb)

        if sim >= 0.6:
            # Similar words — pick the better one using tokenizer validation
            if sim >= 0.9:
                # Almost identical — prefer full-audio version (text_b)
                result.append(wb)
            else:
                # Moderately similar — prefer the more meaningful word
                a_meaningful = _is_meaningful_word(wa)
                b_meaningful = _is_meaningful_word(wb)
                if b_meaningful and not a_meaningful:
                    result.append(wb)
                elif a_meaningful and not b_meaningful:
                    result.append(wa)
                else:
                    # Both meaningful or both garbled — prefer longer (more complete)
                    result.append(wb if len(wb) >= len(wa) else wa)
            i += 1
            j += 1
        else:
            # Very different words — check if skipping one creates better alignment
            # Look ahead to see if next word matches
            skip_a_sim = _word_similarity(words_a[min(i + 1, len(words_a) - 1)], wb) if i + 1 < len(words_a) else 0
            skip_b_sim = _word_similarity(wa, words_b[min(j + 1, len(words_b) - 1)]) if j + 1 < len(words_b) else 0

            if skip_a_sim > 0.6:
                # Word in A is extra/garbled, skip it
                i += 1
            elif skip_b_sim > 0.6:
                # Word in B is extra, skip it
                j += 1
            else:
                # Both are different — prefer full-audio version
                result.append(wb)
                i += 1
                j += 1

    # Don't append remaining words — they're likely ASR artifacts

    return " ".join(result)


def _is_meaningful_word(word: str) -> bool:
    """Check if a word is likely a real Sinhala word using SinLlama tokenizer.
    
    A well-formed Sinhala word tokenizes efficiently into few subword tokens
    because the tokenizer has learned common Sinhala morphemes.
    Garbled/nonsense words fragment into many small character-level pieces,
    indicating they're not in the model's learned vocabulary.
    
    Returns True if the word appears meaningful, False if likely garbled.
    """
    if tokenizer is None:
        return True

    if not word or len(word) <= 1:
        return True  # Single chars / particles are fine

    # Strip punctuation for analysis
    clean = word.strip(".,!?;:\"'()[]{}។")
    if not clean:
        return True

    # Tokenize the word in isolation
    tokens = tokenizer.encode(clean, add_special_tokens=False)
    token_count = len(tokens)
    char_count = len(clean)

    if token_count == 0:
        return False  # Can't tokenize at all

    # Heuristic thresholds based on Sinhala word structure:
    # Real Sinhala words have efficient tokenization (model learned them).
    # Garbled words have high token/char ratio because the tokenizer
    # falls back to character-level or byte-level encoding.
    #
    # Thresholds are intentionally STRICT to catch more garbled words.
    # Words that pass here might still be caught by probability scoring.

    if char_count <= 4:
        return token_count <= 2
    elif char_count <= 8:
        return token_count <= 4
    elif char_count <= 14:
        return token_count <= 5
    else:
        # For longer words, use ratio
        ratio = token_count / char_count
        return ratio < 0.45


def _score_all_words(sentence: str) -> list[tuple[str, float]]:
    """Score how likely each word is in context using a SINGLE forward pass.
    
    Uses the causal LM property: predictions at position i only depend on
    tokens 0..i-1. A single forward pass gives per-position probability.
    
    Words the model didn't expect get low scores — these are likely garbled.
    
    Returns: [(word, min_token_probability), ...] for each word.
    """
    words = sentence.split()
    if not words or model is None or tokenizer is None:
        return [(w, 1.0) for w in words]
        scored_words: list[tuple[str, float]] = []
    try:
        # Tokenize word by word to track token boundaries per word
        all_token_ids: list[int] = []
        word_boundaries: list[tuple[int, int]] = []  # (start, end) in token list

        for idx, word in enumerate(words):
            prefix = " " if idx > 0 else ""
            tokens = tokenizer.encode(prefix + word, add_special_tokens=False)
            start = len(all_token_ids)
            all_token_ids.extend(tokens)
            word_boundaries.append((start, len(all_token_ids)))

        if not all_token_ids:
            return [(w, 1.0) for w in words]

        # Single forward pass — no generation, just get logits
        input_tensor = torch.tensor([all_token_ids], device=input_device)
        with torch.no_grad():
            outputs = model(input_ids=input_tensor)
            logits = outputs.logits[0]  # [seq_len, vocab_size]

        probs = torch.softmax(logits, dim=-1)

        # Score each word by the MINIMUM token probability within it.
        # If any token in the word was very unexpected, the word is suspicious.
        results = []
        for word, (start, end) in zip(words, word_boundaries):
            token_probs = []
            for pos in range(start, end):
                if pos > 0 and pos < len(all_token_ids):
                    actual_token = all_token_ids[pos]
                    p = probs[pos - 1, actual_token].item()
                    token_probs.append(p)

            if token_probs:
                word_score = min(token_probs)
            else:
                word_score = 1.0

            results.append((word, word_score))

        return results

    except Exception as e:
        logger.warning(f"Word probability scoring failed: {e}")
        return [(w, 1.0) for w in words]


def _predict_correct_word(
    left_context: str,
    garbled_word: str,
    right_context: str = "",
) -> str:
    """Use SinLlama transformer to predict the correct word given context.
    
    Algorithm:
    1. Feed left context to the model as a prompt
    2. Beam search top-8 candidate next-word continuations
    3. Extract first word from each candidate beam
    4. Score each candidate by:
       - Character-level similarity to the garbled ASR word (phonetic proxy)
       - Tokenizer efficiency (real words tokenize cleanly)
       - Length compatibility (shouldn't be wildly different length)
       - Beam rank (higher model probability = higher rank bonus)
    5. Return highest-scoring candidate if it passes threshold
    
    Uses the model's knowledge of Sinhala word sequences to predict
    what word SHOULD come at this position, then picks the candidate
    most similar to what ASR heard.
    
    Args:
        left_context: Words before the garbled word
        garbled_word: The word to correct
        right_context: Words after (for future bidirectional use)
    
    Returns:
        The predicted correct word, or original garbled_word if no match
    """
    if model is None or tokenizer is None:
        return garbled_word

    # Need at least some context for meaningful prediction
    prompt = left_context.rstrip()
    if not prompt or len(prompt.split()) < 1:
        return garbled_word

    prompt += " "

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        # Beam search: generate top-8 candidate continuations
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=6,     # Enough tokens for one Sinhala word
                num_beams=8,
                num_return_sequences=8,
                early_stopping=True,
                do_sample=False,       # Pure beam search (deterministic)
                pad_token_id=tokenizer.eos_token_id,
            )

        # Extract first word from each beam's output
        candidates = []
        for seq in outputs:
            new_tokens = seq[input_len:]
            new_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            if new_text:
                first_word = new_text.split()[0].strip(".,!?;:\"'()[]{}។")
                if first_word and first_word not in candidates:
                    candidates.append(first_word)

        if not candidates:
            return garbled_word

        # Score each candidate
        best_word = garbled_word
        best_score = 0.0

        for rank, cand in enumerate(candidates):
            # (a) Similarity to garbled word (phonetic proximity)
            sim = _word_similarity(cand, garbled_word)

            # (b) Bonus for being a meaningful word (clean tokenization)
            meaningful_bonus = 1.3 if _is_meaningful_word(cand) else 0.75

            # (c) Length compatibility — similar length words are preferred
            max_l = max(len(cand), len(garbled_word), 1)
            len_ratio = min(len(cand), len(garbled_word)) / max_l
            len_bonus = 0.5 + 0.5 * len_ratio  # 0.5 to 1.0

            # (d) Beam rank bonus — earlier beams have higher model probability
            rank_bonus = 1.0 + 0.1 * max(0, 5 - rank)  # 1.5 for rank 0, 1.0 for rank 5+

            score = sim * meaningful_bonus * len_bonus * rank_bonus

            if score > best_score:
                best_score = score
                best_word = cand

        # Accept prediction if score is reasonable
        # Lowered threshold to be more willing to correct
        if best_score >= 0.20:
            logger.info(
                f"      Predict: '{garbled_word}' → '{best_word}' "
                f"(score={best_score:.3f}, top3={candidates[:3]})"
            )
            return best_word

        return garbled_word

    except Exception as e:
        logger.warning(f"Word prediction failed for '{garbled_word}': {e}")
        return garbled_word


def _llm_correct_transcript(
    segments: list[dict],
    full_audio_text: str,
    overlapping_regions: list[OverlappingRegion],
) -> list[dict]:
    """Correct ASR transcript using word-level fusion + LM probability scoring
    + transformer beam-search prediction.
    
    Pipeline:
    1. WORD-LEVEL FUSION: Merge per-segment ASR and full-audio aligned ASR
       word-by-word using edit distance.
    2. LM PROBABILITY SCORING: Single forward pass through SinLlama to score
       every word. Words with low probability are flagged as suspicious —
       the model didn't expect them in this context.
    3. TRANSFORMER PREDICTION: For each suspicious word, beam search top-8
       candidate continuations, pick the one most similar to the garbled word.
    4. LENGTH GUARD: Reject output if word count explodes vs input.
    
    This pipeline uses the LLM as a language model — exactly how a transformer
    predicts the next word. Garbled words have low probability because the model
    has never seen them in its Sinhala training data.
    
    Args:
        segments: [{"speaker": str, "start": float, "end": float, "text": str, "aligned_text": str}, ...]
        full_audio_text: Full-audio Whisper transcript
        overlapping_regions: Detected overlap regions
    
    Returns:
        list of {"speaker": str, "start": float, "end": float, "text": str}
    """
    corrected = []

    for i, seg in enumerate(segments):
        raw_text = seg["text"]  # per-segment ASR
        aligned_text = seg.get("aligned_text", "")  # full-audio aligned

        # If both empty, skip
        if not raw_text and not aligned_text:
            corrected.append({
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "text": "",
                "word_details": [],
            })
            continue

        # ── Step 1: Word-level fusion of both ASR sources ──
        fused_text = _word_level_fuse(raw_text, aligned_text)
        if not fused_text:
            fused_text = raw_text or aligned_text

        # ── Step 2: LM probability scoring (single forward pass) ──
        word_details: list[dict] = []  # Track per-word info for response

        if model is not None and tokenizer is not None:
            word_scores = _score_all_words(fused_text)

            logger.info(
                f"   Seg {i+1} word scores: "
                + ", ".join(f"'{w}'={p:.4f}" for w, p in word_scores)
            )

            # ── Step 3: Predict corrections for low-scoring words ──
            corrected_words = []
            prediction_count = 0
            MAX_PREDICTIONS = 12  # Max beam searches per segment

            for word_idx, (word, prob) in enumerate(word_scores):
                meaningful = _is_meaningful_word(word)
                was_corrected = False
                original_word = word
                final_word = word

                # Skip very short words (particles, connectors)
                if len(word) <= 2:
                    corrected_words.append(word)
                # HIGH probability → model expected this word → likely correct
                elif prob > 0.05:
                    corrected_words.append(word)
                # MEDIUM probability + passes tokenizer validation → probably OK
                elif prob > 0.01 and meaningful:
                    corrected_words.append(word)
                # LOW probability → model didn't expect this word → try to correct
                elif prediction_count < MAX_PREDICTIONS:
                    # Build left context from already-corrected words
                    left_ctx = " ".join(corrected_words[-8:])

                    # Build right context from upcoming words
                    right_ctx = " ".join(
                        w for w, _ in word_scores[word_idx + 1:word_idx + 4]
                    )

                    predicted = _predict_correct_word(left_ctx, word, right_ctx)
                    corrected_words.append(predicted)
                    prediction_count += 1
                    final_word = predicted
                    was_corrected = (predicted != word)

                    if was_corrected:
                        logger.info(
                            f"      [{seg['speaker']}] Word {word_idx}: "
                            f"'{word}' (p={prob:.4f}) → '{predicted}'"
                        )
                else:
                    corrected_words.append(word)  # Budget exceeded

                word_details.append({
                    "word": corrected_words[-1],
                    "original_word": original_word,
                    "probability": round(prob, 6),
                    "is_meaningful": meaningful,
                    "was_corrected": was_corrected,
                })

            final_text = " ".join(corrected_words)

            if prediction_count > 0:
                logger.info(
                    f"   Segment {i+1}: {prediction_count} word(s) "
                    f"corrected by transformer prediction"
                )
        else:
            final_text = fused_text
            # No LLM — still produce word details with default scores
            for w in fused_text.split():
                word_details.append({
                    "word": w,
                    "original_word": w,
                    "probability": 1.0,
                    "is_meaningful": True,
                    "was_corrected": False,
                })

        # ── Step 4: Length guard ──
        input_words = len((raw_text or aligned_text).split())
        output_words = len(final_text.split())
        if input_words > 0 and output_words > input_words * 2:
            logger.warning(
                f"   Length guard: output ({output_words}w) > 2x input ({input_words}w), "
                f"using fused text instead"
            )
            final_text = fused_text

        corrected.append({
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "text": final_text,
            "word_details": word_details,
        })

        logger.info(
            f"   Correction [{i+1}/{len(segments)}] {seg['speaker']} "
            f"({seg['start']:.1f}-{seg['end']:.1f}s): "
            f"'{(raw_text or '')[:50]}' → '{final_text[:50]}'"
        )

    return corrected


def _truncate_to_sentences(text: str, max_sentences: int = 1) -> str:
    """Truncate text to at most `max_sentences` complete Sinhala sentences.
    Splits on common Sinhala/Unicode sentence-ending punctuation."""
    if not text:
        return text
    # Split on sentence-ending punctuation (. ! ? and Sinhala full stop ।)
    parts = re.split(r'(?<=[.!?।])\s*', text.strip())
    # Take only the requested number of sentences
    selected = [p for p in parts[:max_sentences] if p.strip()]
    if selected:
        return " ".join(selected).strip()
    # If no sentence boundary found, cap at ~60 chars on a word boundary
    if len(text) > 60:
        truncated = text[:60].rsplit(" ", 1)[0]
        return truncated.strip() + "."
    return text.strip()


# === Word Knowledge Base ===
# Pre-built child-friendly descriptions for each word.
# These are injected directly into the prompt so the model doesn't waste tokens generating them.
WORD_KNOWLEDGE: dict[str, str] = {
    # සතුන් (Animals)
    "බල්ලා": "බල්ලා වව් වව් කියනවා! බල්ලා ගෙදර රකිනවා, බල්ලා අපේ යාළුවා.",
    "පූසා": "පූසා මියාව් මියාව් කියනවා! පූසා කිරි බොනවා, පූසා මීයන් අල්ලනවා.",
    "අලියා": "අලියා ඉතාම ලොකු සතෙක්! අලියාට දිග උඹලක් තියෙනවා, අලියා වතුර ගහනවා.",
    "සිංහයා": "සිංහයා වනයේ රජු! සිංහයා ගර්ජනා කරනවා, සිංහයා ඉතා බලවත්.",
    "හාවා": "හාවා පොඩි සතෙක්! හාවා පැන පැන යනවා, හාවාට දිග කන් තියෙනවා.",
    "සමනළයා": "සමනළයා ලස්සන පියාපත් තියෙනවා! සමනළයා මල් මතින් පියාඹනවා.",
    "කුරුල්ලා": "කුරුල්ලා ගහ උඩ ඉන්නවා! කුරුල්ලා ලස්සනට ගීත ගයනවා.",
    "මාළුවා": "මාළුවා වතුරේ පීනනවා! මාළුවා ගංගාවේ සහ මුහුදේ ඉන්නවා.",
    "ගවයා": "ගවයා මූ කියනවා! ගවයා කිරි දෙනවා, ගවයා තණකොළ කනවා.",
    "එළුවා": "එළුවා මේ මේ කියනවා! එළුවා කරදරකාර සතෙක්.",
    "ඉබ්බා": "ඉබ්බා සෙමින් යනවා! ඉබ්බාට ගල් කබොලක් තියෙනවා.",
    "කිඹුලා": "කිඹුලා වතුරේ ඉන්නවා! කිඹුලාට ලොකු දෙඩ තියෙනවා.",
    "කපුටා": "කපුටා කා කා කියනවා! කපුටා කළු පාටයි.",
    "කොකා": "කොකා වතුරේ මාළු අල්ලනවා! කොකා සුදු පාටයි.",
    "මදුරුවා": "මදුරුවා පොඩි සතෙක්! මදුරුවා රාත්‍රියේ පියාඹනවා.",
    "මීයා": "මීයා පොඩි සතෙක්! මීයා චී චී කියනවා.",
    "කුකුළා": "කුකුළා උදේට කූ කූ කියනවා! කුකුළා බිත්තර දෙනවා.",
    "මොනරා": "මොනරා ලස්සන පිය දිග හරිනවා! මොනරා ලංකාවේ ජාතික පක්ෂියා.",
    "මකුළුවා": "මකුළුවා දැල් ගහනවා! මකුළුවා පොඩි පොඩි බෝනික්කන් අල්ලනවා.",
    "බැබළුවා": "බැබළුවා රාත්‍රියේ එළිය දෙනවා! බැබළුවා පොඩි පොඩි බල්බ වගේ.",
    "උදුරුවා": "උදුරුවා ගස් උඩ ඉන්නවා! උදුරුවාට දිග වලිගයක් තියෙනවා.",
    "මුවා": "මුවා වනයේ ඉන්නවා! මුවාට ලස්සන ඇම් තියෙනවා.",
    # ශරීරය (Body parts)
    "අත": "ඔයාගේ අත බලන්න! අතින් ලියනවා, අතින් කනවා, අතින් ආයුබෝවන් කියනවා.",
    "ඇස": "ඇසින් බලනවා! ඇස් දෙකයි, ඇසින් ලස්සන දේවල් දකිනවා.",
    "කන": "කනින් අහනවා! කන් දෙකයි, කනින් සින්දු අහනවා.",
    "නාසය": "නාසයෙන් හුස්ම ගන්නවා! නාසයෙන් මල් සුවඳ දැනෙනවා.",
    "මුඛය": "මුඛයෙන් කතා කරනවා! මුඛයෙන් කන්නවා, මුඛයෙන් සිනාසෙනවා.",
    "පාදය": "පාදයෙන් ඇවිදිනවා! පාද දෙකයි, පාදයෙන් දුවනවා.",
    "බඩ": "බඩ තියෙන්නේ මැද! බත් කාපු ගමන් බඩ පිරෙනවා.",
    "ඇඟිල්ල": "ඇඟිලි දහයක් තියෙනවා! ඇඟිල්ලෙන් පෙන්වනවා.",
    "දත": "දතින් කනවා! දත් මදින්න ඕනේ, එතකොට දත් ලස්සනයි.",
    "බෙල්ල": "බෙල්ල හිස සහ ඇඟ යා කරනවා! බෙල්ල හරවලා බලන්න පුළුවන්.",
    "කොණ්ඩය": "කොණ්ඩය හිසේ තියෙනවා! කොණ්ඩය පීරන්න ඕනේ.",
    "දිව": "දිවෙන් රස බලනවා! දිවෙන් අයිස්ක්‍රීම් රස බලනවා.",
    # කෑම සහ බීම (Food & Drinks)
    "බත්": "බත් ගෙදර හදනවා! අම්මා ලස්සන බත් උයනවා.",
    "වතුර": "වතුර බොනවා! වතුර පිපාසය නිවනවා.",
    "කිරි": "කිරි සුදු පාටයි! කිරි බොන්න ඇඟට හොඳයි, ආච්චි කිරි දෙනවා.",
    "පාන්": "පාන් උදේට කනවා! පාන් මෘදු සහ රසයි.",
    "බිත්තරය": "බිත්තරය කුකුළා දෙනවා! බිත්තර බැදලා කනවා, රසයි!",
    "අඹ": "අඹ රසවත් පලතුරක්! අඹ කහ පාටයි, ඉතාම පැණි රසයි.",
    "කෙසෙල්": "කෙසෙල් කහ පාටයි! කෙසෙල් කෑම හොඳයි, වඳුරා කෙසෙල් කනවා.",
    "දුරු": "දුරු ගැනීමට හොඳයි!",
    # දේවල් සහ ස්ථාන (Things & Places)
    "පොත": "පොතෙන් කියවනවා! පොතේ කතා තියෙනවා, පොත් කියවීම හොඳයි.",
    "ගස": "ගස ලොකුයි, ගසේ කොළ තියෙනවා! ගස අපට සෙවන දෙනවා.",
    "මල": "මල ලස්සනයි! මල් ලස්සන පාට තියෙනවා, මල සුවඳයි.",
    "ගෙදර": "ගෙදර අපි ඉන්නවා! අම්මා තාත්තා ගෙදර ඉන්නවා.",
    "ගම": "ගමේ ගස් තියෙනවා, ගමේ බල්ලෝ ඉන්නවා, ගම ලස්සනයි.",
    "වත්ත": "වත්තේ මල් තියෙනවා! වත්තේ ගස් සහ පලතුරු හැදෙනවා.",
    "පාසල": "පාසලේ ඉගෙන ගන්නවා! පාසලේ ටීචර් ඉන්නවා, යාළුවෝ ඉන්නවා.",
    "කෝච්චි": "කෝච්චි පීපී කියනවා! කෝච්චි පීල්ල උඩ යනවා, ඉතාම වේගවත්.",
    "බස්": "බස් එකේ යනවා! බස් එකේ මිනිස්සු ගොඩක් යනවා.",
    "කාර්": "කාර් එකේ යනවා! තාත්තා කාර් එක පදිනවා.",
    "බෝලය": "බෝලයෙන් ක්‍රීඩා කරනවා! බෝලය ගහලා එලියට යනවා.",
    "රෝදය": "රෝදය කරකැවෙනවා! බස් එකේ, කාර් එකේ රෝද තියෙනවා.",
    # පවුල (Family)
    "අම්මා": "අම්මා අපට ආදරේ! අම්මා කන්න හදනවා, අම්මා ළඟ ඉන්න හොඳයි.",
    "තාත්තා": "තාත්තා අපව රකිනවා! තාත්තා වැඩ කරනවා, තාත්තා ශක්තිමත්.",
    "අක්කා": "අක්කා ලොකු නංගිට උදව් කරනවා! අක්කා එක්ක සෙල්ලම් කරනවා.",
    "අයියා": "අයියා ලොකුයි! අයියා එක්ක ක්‍රිකට් ගහනවා.",
    "නංගි": "නංගි පොඩියි, නංගි ලස්සනයි! නංගිත් එක්ක සෙල්ලම් කරනවා.",
    "මල්ලි": "මල්ලි පොඩියි! මල්ලි එක්ක සෙල්ලම් කරනවා.",
    "ආච්චි": "ආච්චි කතා කියනවා! ආච්චි ළඟ කිරි තොෆී තියෙනවා.",
    "සීයා": "සීයා ගෙවත්තේ වැඩ කරනවා! සීයා කතා කියනවා.",
    # ස්වභාවය (Nature)
    "ඉර": "ඉර උදේට එනවා! ඉර එළිය දෙනවා, ඉර උෂ්ණයි.",
    "සඳ": "සඳ රාත්‍රියේ එනවා! සඳ ලස්සනයි, සඳ එළිය දෙනවා.",
    "තරු": "තරු රාත්‍රියේ පායනවා! තරු ඉතාම ලස්සනයි, බොහොම ඈතයි.",
    "වැස්ස": "වැස්ස වහිනකොට වතුර එනවා! වැස්සට කුඩයක් ගන්නවා.",
    # පාට (Colors)
    "රතු": "රතු පාට ලස්සනයි! රෝස මල රතු පාටයි.",
    "නිල්": "නිල් පාට අහසේ පාටයි! මුහුදත් නිල් පාටයි.",
    "කහ": "කහ පාට ඉරගේ පාටයි! කෙසෙල් කහ පාටයි.",
    "කොළ": "කොළ පාට ගස්වල පාටයි! කොළ පාට තණකොළ වගේ.",
    "සුදු": "සුදු පාට වලාකුළු වගේ! කිරිත් සුදු පාටයි.",
    # වෙනත් (Other common words from word list)
    "ඇලේ": "ඇලේ වතුර ගලනවා! ඇලේ මාළු ඉන්නවා.",
    "අං": "අං තියෙන්නේ ගවයාට, අලියාට! අං ශක්තිමත්.",
    "නළ": "නළ ඇතුලෙන් වතුර යනවා! නළ දිගට තියෙනවා.",
    "බල": "බල ශක්තියට කියනවා! 'බලන්න' කිව්වම ඇසින් බලනවා.",
    "රබර්": "රබර් ගසෙන් හදනවා! රබර් ඇඟිලිවලට දාන බෑන්ඩ් වගේ.",
    "නළු": "නළු නාට්‍යයේ රඟ පාන්නවා! නළු සිනමාවේ ඉන්නවා.",
    "හූනා": "හූනා වතුරේ ඉන්නවා! හූනා දිගට සහ සිනිඳුයි.",
    "අත්ත": "අත්ත ගසේ ශාඛාවක්! ගස්වල අත්ත ගොඩක් තියෙනවා.",
    "අරියා": "අරියා බුද්ධිමත් කෙනෙක්!",
    "නාගි": "නාගි ඇඳට ඇඳලම ගන්නවා!",
    "රේද්ද": "රේද්ද ඇඳගන්නවා! රේද්ද ලස්සන පාට පාට.",
    "නාරං": "නාරං පලතුරක්! නාරං තැඹිලි පාටයි, රසයි.",
    "දුන්": "දුන් කිව්වම යමක් දීම! අම්මා කෑම දුන්නා.",
    "අපණ": "අපණ බ්‍රෂ් එකක්!",
    "බලම": "බලම කියන්නේ බලන්නම කියන එක!",
    "මෙහෙණි": "මෙහෙණි භික්ෂුණියක්!",
    "කුළ": "කුළ ජාතිය!",
    "කොන්": "කොන් කෙළවර!",
    "අප්ප": "අප්ප රසවත් කෑමක්! අප්ප ගෙදර හදනවා.",
    "කුළුබඩු": "කුළුබඩු කෑමට දානවා! කුළුබඩු වලින් කෑම රසවත්.",
    "අල": "අල මුල් බෝගයක්! අල කරි හදනවා, අල රසයි.",
    "මෙංහෑන්": "මෙංහෑන් සුවඳ හොඳයි!",
    "කුළ": "කුළ ගොඩනැගිල්ලක කුළුණක්!",
    "පෙට්ටිය": "පෙට්ටිය ඇතුලේ දේවල් තියනවා! පෙට්ටිය වහන්න, අරින්න පුළුවන්.",
    "සබන්": "සබනින් අත් සෝදනවා! සබන් පොගුරු හදනවා.",
    "පේන": "පේනෙන් ලියනවා! පේන්සල් වගේ.",
    "කතුර": "කතුරෙන් කපනවා! කඩදාසි කපන්න කතුර ගන්නවා.",
    "කෝප්ප": "කෝප්පයෙන් වතුර බොනවා! කෝප්ප මේසේ තියෙනවා.",
    "පාර": "පාරේ ඇවිදිනවා! පාරේ කාර්, බස් යනවා.",
    "හැන්ද": "හැන්දෙන් බත් කනවා! හැන්ද මේසේ තියෙනවා.",
    "බුද්ධි": "බුද්ධි කියන්නේ හිතන බලය!",
    "පෝට්": "පෝට් එකේ බෝට්ටු තියෙනවා!",
    "රතනලය": "රතනලය ලස්සන තැනක්!",
    "භාණ්ඩය": "භාණ්ඩය ඇතුලේ දේවල් තබනවා!",
    "බිස්ලේරි": "බිස්ලේරි වතුර බෝතලයක්!",
    "රකුර": "රකුර රන් වගේ බබලනවා!",
    "පෑන": "පෑනෙන් ලියනවා! පෑන් වලින් ලස්සනට ලියන්න පුළුවන්.",
    "පේට්": "පේට් තැපැල් නිලධාරියා!",
    "පොටි": "පොටි පාටයි, ලස්සනයි!",
    "සබන්": "සබනින් අත් සෝදනවා! සබන් සුවඳයි.",
    "කොළඹ": "කොළඹ ලංකාවේ ලොකුම නගරය!",
    "හෝස්පිටලය": "හෝස්පිටලයේ ඩොක්ටර් ඉන්නවා! අසනීප උනාම හෝස්පිටලයට යනවා.",
    "රෝහල": "රෝහලේ ඩොක්ටර් බෙහෙත් දෙනවා!",
    "ග්‍රීස්": "ග්‍රීස් ලිස්සනවා!",
    "බුරුළු": "බුරුළු රස කෑමක්!",
    "සතුර": "සතුර කියන්නේ හතුරා!",
    "හැඳ": "හැඳ ඇඳලා ගන්නවා! ළමයි හැඳ ඇඳගන්නවා.",
    "සාමය": "සාමය තියෙන්න ඕනේ! සාමය හොඳයි.",
}


def _get_word_description(word: str) -> str:
    """Get pre-built description for a word, or return empty string."""
    return WORD_KNOWLEDGE.get(word, "")


import random


def _build_child_response(req) -> str | None:
    """Build a template-based response for structured pronunciation scenarios.
    Returns None if the scenario needs LLM generation (free conversation)."""

    name = req.child_name or "ළමයා"
    word = req.current_word
    nxt = req.next_word
    desc = _get_word_description(word) if word else ""
    next_desc = _get_word_description(nxt) if nxt else ""
    total = len(req.session_words) if req.session_words else 3
    completed = req.words_completed if req.words_completed else 0

    # --- Auto-detect session complete ---
    # If is_session_complete is explicitly set, OR if the last word was correct/skipped
    # with no next word and completed >= total, treat as session complete
    session_done = req.is_session_complete
    if not session_done and req.pronunciation_result in ("correct", "skipped") and not nxt:
        if completed >= total or (req.session_words and word == req.session_words[-1]):
            session_done = True

    # --- Session Complete ---
    if session_done:
        # If last word was correct, praise it first then end session
        praise = ""
        if req.pronunciation_result == "correct" and word:
            praise = f"වාව් {name}! '{word}' හරියටම කිව්වා! "
            if desc:
                praise += f"{desc} "
        templates = [
            f"{praise}සුපිරි {name}! අද session එක ඉවරයි! වචන {completed}/{total}ක් හරියට කිව්වා. ඔයා champion කෙනෙක්! හෙට ආයෙත් ඉගෙන ගමු!",
            f"{praise}වාව් {name}! ඔයා අද වචන {completed}ක් ඉගෙන ගත්තා! ඔයා හරිම දක්ෂයි! හෙට ආයෙත් එමුද?",
            f"{praise}හොඳයි {name}! අද අපි වචන {completed}ක් ඉගෙන ගත්තා! ඔයාට පුළුවන් කියලා පෙන්නුවා! හෙට තව වචන ඉගෙන ගමු!",
        ]
        return random.choice(templates)

    # --- Correct + Transition to Next Word ---
    if req.pronunciation_result == "correct" and nxt:
        praise_parts = [
            f"වාව් {name}! '{word}' හරියටම කිව්වා!",
            f"සුපිරි {name}! '{word}' නිවැරදියි!",
            f"හරිම හොඳයි {name}! '{word}' හරියට කිව්වා!",
        ]
        praise = random.choice(praise_parts)
        if desc:
            praise += f" {desc}"
        # Introduce next word
        intro = f" දැන් අපි '{nxt}' ගැන ඉගෙන ගමු."
        if next_desc:
            intro += f" {next_desc}"
        intro += f" {name}, '{nxt}' කියන්න පුළුවන්ද?"
        return praise + intro

    # --- Correct (mid-session, no next word provided — ask client to send next_word) ---
    if req.pronunciation_result == "correct":
        templates = [
            f"වාව් {name}! '{word}' හරියටම කිව්වා! {desc} ඔයා හරිම දක්ෂයි! ඊළඟ වචනයට යමුද?" if desc else f"වාව් {name}! '{word}' හරියටම කිව්වා! ඔයා හරිම දක්ෂයි! ඊළඟ වචනයට යමුද?",
            f"සුපිරි {name}! '{word}' නිවැරදියි! {desc} ඔයාට පුළුවන්! ඊළඟ වචනයට යමු!" if desc else f"සුපිරි {name}! '{word}' නිවැරදියි! ඔයාට පුළුවන්! ඊළඟ වචනයට යමු!",
        ]
        return random.choice(templates)

    # --- Close (almost correct) ---
    if req.pronunciation_result == "close":
        templates = [
            f"හොඳයි {name}! '{word}' ආසන්නයි! {desc} තව පොඩ්ඩක් උත්සාහ කරමු. '{word}' ආයේ කියන්නකෝ!" if desc else f"හොඳයි {name}! '{word}' ආසන්නයි! තව පොඩ්ඩක් උත්සාහ කරමු. '{word}' ආයේ කියන්නකෝ!",
            f"වාව් {name}! ඔයා ආසන්නයි! {desc} '{word}' ආයෙත් කියන්න, ඔයාට පුළුවන්!" if desc else f"වාව් {name}! ඔයා ආසන්නයි! '{word}' ආයෙත් කියන්න, ඔයාට පුළුවන්!",
        ]
        return random.choice(templates)

    # --- Retry (incorrect) ---
    if req.pronunciation_result == "retry":
        templates = [
            f"බය වෙන්න එපා {name}! '{word}' ගැන බලමු. {desc} ආයෙත් '{word}' කියන්නකෝ!" if desc else f"බය වෙන්න එපා {name}! ආයෙත් '{word}' කියන්නකෝ, ඔයාට පුළුවන්!",
            f"කමක් නැහැ {name}! අපි ආයෙත් උත්සාහ කරමු. {desc} '{word}' කියන්න පුළුවන්ද?" if desc else f"කමක් නැහැ {name}! ආයෙත් උත්සාහ කරමු. '{word}' කියන්න පුළුවන්ද?",
            f"{name}, ටිකක් අමාරුද? {desc} හරි, ආයෙත් '{word}' කියන්නකෝ!" if desc else f"{name}, ටිකක් අමාරුද? හරි, ආයෙත් '{word}' කියන්නකෝ!",
        ]
        return random.choice(templates)

    # --- No Speech ---
    if req.pronunciation_result == "no_speech":
        templates = [
            f"{name}, බය වෙන්න එපා! අපි එකට ඉගෙන ගමු. {desc} '{word}' කියන්න පුළුවන්ද?" if desc else f"{name}, බය වෙන්න එපා! '{word}' කියන්න උත්සාහ කරන්නකෝ!",
            f"හරි {name}, බය වෙන්න එපා! {desc} '{word}' කියන්න පුළුවන්ද? ඔයාට පුළුවන්!" if desc else f"හරි {name}, බය වෙන්න එපා! '{word}' කියන්න, ඔයාට පුළුවන්!",
        ]
        return random.choice(templates)

    # --- Skipped + Next Word ---
    if req.pronunciation_result == "skipped" and nxt:
        intro = f"කමක් නැහැ {name}! පස්සේ '{word}' ආයෙත් උත්සාහ කරමු. දැන් අපි '{nxt}' ගැන ඉගෙන ගමු."
        if next_desc:
            intro += f" {next_desc}"
        intro += f" {name}, '{nxt}' කියන්න පුළුවන්ද?"
        return intro

    # --- Skipped (no next) ---
    if req.pronunciation_result == "skipped":
        return f"කමක් නැහැ {name}! පස්සේ ආයෙත් උත්සාහ කරමු. ඔයා හොඳට කරනවා!"

    # --- New Word Introduction ---
    if req.pronunciation_result == "new_word":
        templates = [
            f"{name}, අද අපි '{word}' ගැන ඉගෙන ගමු! {desc} '{word}' කියන්න පුළුවන්ද?" if desc else f"{name}, අද අපි '{word}' ගැන ඉගෙන ගමු! '{word}' කියන්න පුළුවන්ද?",
            f"හරි {name}! අලුත් වචනයක්! {desc} '{word}' කියන්නකෝ!" if desc else f"හරි {name}! අලුත් වචනයක්! '{word}' කියන්නකෝ!",
        ]
        return random.choice(templates)

    # --- No matching scenario → use LLM for free conversation ---
    return None


@app.get("/health")
async def health_check():
    """Health check — no auth required."""
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        gpu_info.append(
            {
                "gpu_id": i,
                "name": torch.cuda.get_device_name(i),
                "vram_used_gb": round(torch.cuda.memory_allocated(i) / 1e9, 2),
                "vram_reserved_gb": round(torch.cuda.memory_reserved(i) / 1e9, 2),
                "vram_total_gb": round(
                    torch.cuda.get_device_properties(i).total_memory / 1e9, 1
                ),
            }
        )

    return {
        "status": "healthy" if model is not None else "loading",
        "model_loaded": model is not None,
        "diarization_loaded": diar_pipeline is not None,
        "tts_service_url": TTS_SERVICE_URL,
        "gpu_count": torch.cuda.device_count(),
        "input_device": str(input_device) if input_device else None,
        "gpus": gpu_info,
    }


# === TTS Proxy Endpoints (forwards to tts_server.py on port 8001) ===

@app.get("/tts/speakers")
async def tts_get_speakers():
    """List available TTS speakers (proxied from TTS microservice)."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{TTS_SERVICE_URL}/speakers")
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="TTS service is not running. Start tts_server.py on port 8001.")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS service error: {str(e)}")


@app.post(
    "/synthesize",
    dependencies=[Depends(verify_api_key)],
    summary="Text-to-Speech (POST)",
    description="Synthesize Sinhala speech from romanized text. Proxied to TTS microservice. Returns WAV audio.",
)
async def synthesize_post(req: SynthesizeRequest):
    """Synthesize speech from text (JSON body). Proxied to TTS microservice."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{TTS_SERVICE_URL}/synthesize",
                json={"text": req.text, "speaker": req.speaker, "speed": req.speed},
                headers={"X-API-Key": API_KEY},
            )
            resp.raise_for_status()
            return StreamingResponse(
                io.BytesIO(resp.content),
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=output.wav"},
            )
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="TTS service is not running. Start tts_server.py on port 8001.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS service error: {str(e)}")


@app.get(
    "/synthesize",
    dependencies=[Depends(verify_api_key)],
    summary="Text-to-Speech (GET)",
    description="Synthesize Sinhala speech from romanized text via query params. Proxied to TTS microservice. Returns WAV audio.",
)
async def synthesize_get(
    text: str = Query(..., description="Romanized Sinhala text to synthesize"),
    speaker: Optional[str] = Query(None, description="Speaker name (for multi-speaker models)"),
):
    """Synthesize speech from text (query parameters). Proxied to TTS microservice."""
    try:
        params = {"text": text}
        if speaker:
            params["speaker"] = speaker
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(
                f"{TTS_SERVICE_URL}/synthesize",
                params=params,
                headers={"X-API-Key": API_KEY},
            )
            resp.raise_for_status()
            return StreamingResponse(
                io.BytesIO(resp.content),
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=output.wav"},
            )
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="TTS service is not running. Start tts_server.py on port 8001.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS service error: {str(e)}")


@app.post(
    "/generate",
    response_model=GenerateResponse,
    dependencies=[Depends(verify_api_key)],
)
async def generate_text(req: GenerateRequest):
    """Raw text generation endpoint."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        start = time.time()

        # Tokenize and move to the correct GPU
        inputs = tokenizer(
            req.prompt, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only newly generated tokens
        new_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        tokens_generated = len(new_tokens)

        elapsed_ms = (time.time() - start) * 1000
        tps = tokens_generated / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

        logger.info(
            f"⚡ {tokens_generated} tokens in {elapsed_ms:.0f}ms "
            f"({tps:.1f} tok/s) | prompt={input_length} tokens"
        )

        return GenerateResponse(
            generated_text=generated_text,
            tokens_generated=tokens_generated,
            inference_time_ms=round(elapsed_ms, 1),
        )

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(
            status_code=507,
            detail="GPU out of memory. Try shorter prompt or fewer tokens.",
        )
    except Exception as e:
        logger.error(f"Generation error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/ask",
    response_model=AskResponse,
    dependencies=[Depends(verify_api_key)],
)
async def ask(req: AskRequest):
    """General-purpose LLM endpoint. Send any message with an optional system prompt and conversation history."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Build the prompt from system prompt + history + current message
    parts = [f"{req.system_prompt}\n"]

    for turn in req.conversation_history[-10:]:  # keep last 10 turns
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            parts.append(f"පරිශීලකයා: {content}")
        else:
            parts.append(f"සහකාරයා: {content}")

    parts.append(f"පරිශීලකයා: {req.message}")
    parts.append("සහකාරයා:")

    prompt = "\n".join(parts)

    gen_req = GenerateRequest(
        prompt=prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
    )

    result = await generate_text(gen_req)

    # Clean response — stop at continuation markers
    response_text = result.generated_text
    for stop_token in ["පරිශීලකයා:", "\n\n", "සහකාරයා:", "\nපරිශීලකයා", "\nසහකාරයා"]:
        if stop_token in response_text:
            response_text = response_text.split(stop_token)[0].strip()

    # Truncate to first 2 sentences max for conversational brevity
    response_text = _truncate_to_sentences(response_text, max_sentences=2)

    return AskResponse(
        reply=response_text,
        tokens_generated=result.tokens_generated,
        inference_time_ms=result.inference_time_ms,
    )


@app.post(
    "/child-chat",
    response_model=ChildChatResponse,
    dependencies=[Depends(verify_api_key)],
)
async def child_chat(req: ChildChatRequest):
    """Child conversation endpoint — uses pre-built templates for pronunciation
    scenarios (instant, reliable) and falls back to LLM only for free conversation."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    start_time = time.time()

    # Try template-based response first (covers all structured scenarios)
    template_response = _build_child_response(req)

    if template_response is not None:
        # Template hit — return instantly, no LLM call needed
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"💬 Child-chat (template): {len(template_response)} chars in {elapsed_ms:.0f}ms "
            f"| result={req.pronunciation_result} word={req.current_word}"
        )
        return ChildChatResponse(
            response=template_response,
            inference_time_ms=round(elapsed_ms, 1),
        )

    # --- Free conversation fallback (no pronunciation_result) → use LLM ---
    history_text = ""
    if req.conversation_history:
        history_text = "\n".join(req.conversation_history[-4:])

    # If there's a current word, mention it in context
    word_context = ""
    if req.current_word:
        desc = _get_word_description(req.current_word)
        if desc:
            word_context = f"දැන් ඉගෙන ගන්න වචනය: '{req.current_word}'. {desc}"
        else:
            word_context = f"දැන් ඉගෙන ගන්න වචනය: '{req.current_word}'."

    prompt = f"""ඔබ කුඩා දරුවන්ට සිංහල වචන උගන්වන මිත්‍රශීලී ගුරුවරියකි.
{req.child_name}ට කතා කරන්න. කෙටි වාක්‍ය 2ක්. සරල සිංහල.
{word_context}

{history_text}
දරුවා: {req.child_utterance}
ගුරුවරිය:"""

    gen_req = GenerateRequest(
        prompt=prompt,
        max_new_tokens=40,
        temperature=0.4,
        top_p=0.85,
        top_k=40,
        repetition_penalty=1.3,
    )

    result = await generate_text(gen_req)

    # Clean response
    response_text = result.generated_text
    for stop_token in ["දරුවා:", "\n\n", "ගුරුවරිය:", "\nදරුවා", "\nගුරුවරිය"]:
        if stop_token in response_text:
            response_text = response_text.split(stop_token)[0].strip()

    response_text = _truncate_to_sentences(response_text, max_sentences=2)

    return ChildChatResponse(
        response=response_text,
        inference_time_ms=result.inference_time_ms,
    )


# === ASR Endpoint ===
@app.post(
    "/transcribe",
    response_model=TranscribeResponse,
    dependencies=[Depends(verify_api_key)],
)
async def transcribe_audio(audio: UploadFile = File(..., description="Audio file (wav, mp3, flac, ogg, webm)")):
    """Transcribe Sinhala speech to text using Whisper Small Sinhala model."""
    if asr_pipe is None:
        raise HTTPException(status_code=503, detail="ASR model not loaded yet.")

    # Validate file type
    allowed_types = {
        "audio/wav", "audio/x-wav", "audio/wave",
        "audio/mpeg", "audio/mp3",
        "audio/flac",
        "audio/ogg",
        "audio/webm",
        "application/octet-stream",  # fallback for unknown mime
    }
    if audio.content_type and audio.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio type: {audio.content_type}. Use wav, mp3, flac, ogg, or webm.",
        )

    # Save uploaded file to a temp file (Whisper pipeline needs a file path)
    suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(audio.file, tmp)

        start = time.time()

        # Run Whisper inference
        result = asr_pipe(tmp_path)
        transcribed_text = result["text"].strip()

        elapsed_ms = (time.time() - start) * 1000

        logger.info(
            f"🎙️ ASR: '{transcribed_text[:50]}...' in {elapsed_ms:.0f}ms "
            f"| file={audio.filename} size={audio.size or 'unknown'}"
        )

        return TranscribeResponse(
            text=transcribed_text,
            language="si",
            inference_time_ms=round(elapsed_ms, 1),
        )

    except Exception as e:
        logger.error(f"ASR error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# === Combined ASR + Child Chat Endpoint ===
@app.post(
    "/child-session",
    response_model=ChildSessionResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Combined ASR + Child Chat (single round-trip)",
    description="""
Send the child's audio recording along with session metadata.
The server will:
1. Transcribe the audio using Whisper Sinhala ASR
2. Generate the teacher's response (template-based for pronunciation scenarios, LLM for free chat)
3. Return both the transcription and teacher response in one call

**All metadata fields are sent as form fields alongside the audio file.**

### Pronunciation Result Values
| Value | Meaning |
|---|---|
| `new_word` | Introducing a new word to the child |
| `correct` | Child said the word correctly |
| `close` | Child was close but not exact |
| `retry` | Child's pronunciation needs more work |
| `no_speech` | No speech detected from the child |
| `skipped` | Word was skipped |
| *(empty)* | Free conversation (uses LLM) |

### Session Flow
1. Send `pronunciation_result=new_word` with `current_word` to introduce first word
2. Send audio + `pronunciation_result` based on how the child did
3. When correct, include `next_word` to transition
4. When all words done, set `is_session_complete=true`
""",
)
async def child_session(
    audio: UploadFile = File(..., description="Child's audio recording (wav, mp3, flac, ogg, webm)"),
    child_name: str = Form(default="ළමයා", description="Child's name"),
    current_word: str = Form(default="", description="Word currently being practiced"),
    pronunciation_result: str = Form(
        default="",
        description="ASR result: 'correct', 'close', 'retry', 'no_speech', 'skipped', 'new_word', or empty for free chat",
    ),
    attempt_number: int = Form(default=1, description="Which attempt for current word (1-10)"),
    session_words: str = Form(
        default="[]",
        description='JSON array of session words, e.g. ["බල්ලා","පූසා","අලියා"]',
    ),
    next_word: str = Form(default="", description="Next word to transition to (when correct/skipped)"),
    words_completed: int = Form(default=0, description="Words completed so far"),
    is_session_complete: bool = Form(default=False, description="True when all words are done"),
    conversation_history: str = Form(
        default="[]",
        description='JSON array of previous turns, e.g. ["ගුරුවරිය: ...","දරුවා: ..."]',
    ),
    target_phoneme: str = Form(default="", description="Target phoneme for practice"),
):
    """Combined endpoint: transcribes child audio + generates teacher response in one call."""
    if asr_pipe is None:
        raise HTTPException(status_code=503, detail="ASR model not loaded yet.")
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="LLM not loaded yet.")

    total_start = time.time()

    # --- Parse JSON form fields ---
    try:
        parsed_session_words = _json.loads(session_words) if session_words else []
    except (ValueError, TypeError):
        parsed_session_words = []
    try:
        parsed_history = _json.loads(conversation_history) if conversation_history else []
    except (ValueError, TypeError):
        parsed_history = []

    # --- Step 1: ASR — Transcribe audio ---
    allowed_types = {
        "audio/wav", "audio/x-wav", "audio/wave",
        "audio/mpeg", "audio/mp3",
        "audio/flac", "audio/ogg", "audio/webm",
        "application/octet-stream",
    }
    if audio.content_type and audio.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio type: {audio.content_type}. Use wav, mp3, flac, ogg, or webm.",
        )

    suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
    tmp_path = None
    transcribed_text = ""
    asr_time_ms = 0.0

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(audio.file, tmp)

        asr_start = time.time()
        asr_result = asr_pipe(tmp_path)
        transcribed_text = asr_result["text"].strip()
        asr_time_ms = (time.time() - asr_start) * 1000

        logger.info(
            f"🎙️ ASR: '{transcribed_text[:60]}' in {asr_time_ms:.0f}ms"
        )
    except Exception as e:
        logger.error(f"ASR error in child-session: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # --- Step 2: Generate teacher response ---
    # Build a ChildChatRequest-like object for _build_child_response
    class _SessionReq:
        pass

    req = _SessionReq()
    req.child_utterance = transcribed_text
    req.conversation_history = parsed_history
    req.child_name = child_name
    req.target_phoneme = target_phoneme
    req.current_word = current_word
    req.pronunciation_result = pronunciation_result
    req.attempt_number = attempt_number
    req.session_words = parsed_session_words
    req.next_word = next_word
    req.words_completed = words_completed
    req.is_session_complete = is_session_complete

    chat_start = time.time()

    # Try template first
    template_response = _build_child_response(req)

    if template_response is not None:
        chat_time_ms = (time.time() - chat_start) * 1000
        total_ms = (time.time() - total_start) * 1000
        logger.info(
            f"💬 Child-session (template): asr={asr_time_ms:.0f}ms chat={chat_time_ms:.0f}ms "
            f"total={total_ms:.0f}ms | result={pronunciation_result} word={current_word}"
        )
        return ChildSessionResponse(
            transcribed_text=transcribed_text,
            teacher_response=template_response,
            asr_time_ms=round(asr_time_ms, 1),
            chat_time_ms=round(chat_time_ms, 1),
            total_time_ms=round(total_ms, 1),
        )

    # Free conversation fallback → use LLM
    history_text = ""
    if parsed_history:
        history_text = "\n".join(parsed_history[-4:])

    word_context = ""
    if current_word:
        desc = _get_word_description(current_word)
        if desc:
            word_context = f"දැන් ඉගෙන ගන්න වචනය: '{current_word}'. {desc}"
        else:
            word_context = f"දැන් ඉගෙන ගන්න වචනය: '{current_word}'."

    prompt = f"""ඔබ කුඩා දරුවන්ට සිංහල වචන උගන්වන මිත්‍රශීලී ගුරුවරියකි.
{child_name}ට කතා කරන්න. කෙටි වාක්‍ය 2ක්. සරල සිංහල.
{word_context}

{history_text}
දරුවා: {transcribed_text}
ගුරුවරිය:"""

    gen_req = GenerateRequest(
        prompt=prompt,
        max_new_tokens=40,
        temperature=0.4,
        top_p=0.85,
        top_k=40,
        repetition_penalty=1.3,
    )

    result = await generate_text(gen_req)

    response_text = result.generated_text
    for stop_token in ["දරුවා:", "\n\n", "ගුරුවරිය:", "\nදරුවා", "\nගුරුවරිය"]:
        if stop_token in response_text:
            response_text = response_text.split(stop_token)[0].strip()

    response_text = _truncate_to_sentences(response_text, max_sentences=2)

    chat_time_ms = (time.time() - chat_start) * 1000
    total_ms = (time.time() - total_start) * 1000

    logger.info(
        f"💬 Child-session (LLM): asr={asr_time_ms:.0f}ms chat={chat_time_ms:.0f}ms "
        f"total={total_ms:.0f}ms"
    )

    return ChildSessionResponse(
        transcribed_text=transcribed_text,
        teacher_response=response_text,
        asr_time_ms=round(asr_time_ms, 1),
        chat_time_ms=round(chat_time_ms, 1),
        total_time_ms=round(total_ms, 1),
    )


# === Speaker Diarization + ASR Endpoint ===
@app.post(
    "/diarize",
    response_model=DiarizeResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Speaker Diarization + Sinhala ASR",
    description="""
Upload an audio file to get a speaker-labeled Sinhala transcript with AI-optimized correction.

The server will:
1. Run **pyannote speaker diarization** to identify who spoke when
2. Run **full-audio Whisper ASR** (with timestamps) for high-quality transcription
3. **Align** Whisper chunks to speaker segments by timestamp overlap
4. Run **per-segment ASR** as a secondary reference
5. **Detect overlapping** speaker regions
6. Use the **SinLlama LLM** to correct ASR errors and produce an optimized transcript

Typically handles **4-5 speakers** in a single recording.
Supports wav, mp3, flac, ogg, and webm formats.

**Form fields:**
- `audio` (required): Audio file upload
- `num_speakers` (optional): Exact number of speakers if known (skips auto-detection)
- `min_speakers` (default 2): Minimum speakers for auto-detection
- `max_speakers` (default 8): Maximum speakers for auto-detection
- `enable_llm_optimization` (default true): Enable LLM-based transcript correction
""",
)
async def diarize_audio(
    audio: UploadFile = File(..., description="Audio file (wav, mp3, flac, ogg, webm)"),
    num_speakers: Optional[int] = Form(
        default=None,
        description="Known number of speakers (leave empty for auto-detect, typically 4-5)",
    ),
    min_speakers: int = Form(default=2, description="Minimum expected speakers for auto-detection"),
    max_speakers: int = Form(default=8, description="Maximum expected speakers for auto-detection"),
    enable_llm_optimization: bool = Form(default=True, description="Enable LLM-based transcript correction"),
):
    """Speaker diarization with per-speaker Sinhala transcription."""
    if diar_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Diarization model not loaded. "
                "Ensure pyannote.audio is installed and HF_TOKEN env var is set. "
                "You must also accept the model licenses on HuggingFace."
            ),
        )
    if asr_pipe is None:
        raise HTTPException(status_code=503, detail="ASR model not loaded yet.")

    # Validate file type
    allowed_types = {
        "audio/wav", "audio/x-wav", "audio/wave",
        "audio/mpeg", "audio/mp3",
        "audio/flac", "audio/ogg", "audio/webm",
        "application/octet-stream",
    }
    if audio.content_type and audio.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio type: {audio.content_type}. Use wav, mp3, flac, ogg, or webm.",
        )

    total_start = time.time()
    suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
    tmp_path = None
    processed_path = None

    try:
        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(audio.file, tmp)

        # Ensure 16kHz mono (required by both diarization and ASR)
        waveform, sr = torchaudio.load(tmp_path)
        changed = False
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            changed = True
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            changed = True

        if changed:
            processed_path = tmp_path.replace(suffix, "_16k.wav")
            torchaudio.save(processed_path, waveform, 16000)
            wav_path = processed_path
        else:
            wav_path = tmp_path

        audio_duration = waveform.shape[1] / 16000
        logger.info(
            f"🗣️ Diarization request: {audio.filename} "
            f"({audio_duration:.1f}s, {len(waveform[0]) / 1024:.0f} KB samples)"
        )

        # ── Step 1: Speaker Diarization ──
        diar_start = time.time()

        if num_speakers is not None and num_speakers > 0:
            diarization_result = diar_pipeline(wav_path, num_speakers=num_speakers)
        else:
            diarization_result = diar_pipeline(
                wav_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

        # Collect speaker segments (skip very short bursts < 0.4s)
        # pyannote.audio v4.x returns DiarizeOutput with .speaker_diarization attribute
        annotation = None
        if hasattr(diarization_result, "itertracks"):
            # Direct Annotation object (v3.x style)
            annotation = diarization_result
        elif hasattr(diarization_result, "speaker_diarization"):
            # pyannote.audio v4.x DiarizeOutput
            annotation = diarization_result.speaker_diarization
        elif hasattr(diarization_result, "to_annotation"):
            annotation = diarization_result.to_annotation()
        else:
            annotation = None

        if annotation is None or not hasattr(annotation, "itertracks"):
            logger.error(
                f"Cannot extract annotation from {type(diarization_result).__name__}. "
                f"Available attrs: {[a for a in dir(diarization_result) if not a.startswith('_')]}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Unsupported diarization output type: {type(diarization_result).__name__}",
            )

        raw_segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            duration = turn.end - turn.start
            if duration < 0.4:
                continue
            raw_segments.append({
                "speaker": speaker,
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
            })

        speakers_found = list(set(s["speaker"] for s in raw_segments))
        diar_time_ms = (time.time() - diar_start) * 1000
        logger.info(
            f"   Diarization: {len(speakers_found)} speakers, "
            f"{len(raw_segments)} segments in {diar_time_ms:.0f}ms"
        )

        # ── Step 2: Full-Audio Whisper ASR (with timestamps) ──
        # Full-audio transcription is higher quality than per-segment because
        # Whisper has the full conversational context
        asr_start = time.time()
        logger.info("   Running full-audio Whisper ASR with timestamps...")

        audio_array_full = waveform.squeeze().numpy().astype(np.float32)
        full_asr_result = asr_pipe(
            {"raw": audio_array_full, "sampling_rate": 16000},
            return_timestamps=True,
        )
        full_audio_text = full_asr_result.get("text", "").strip()
        whisper_chunks = full_asr_result.get("chunks", [])
        logger.info(
            f"   Full-audio ASR: '{full_audio_text[:60]}...' "
            f"({len(whisper_chunks)} chunks)"
        )

        # ── Step 3: Align Whisper chunks to speaker segments ──
        aligned_segments = _align_chunks_to_speakers(whisper_chunks, raw_segments)
        logger.info(f"   Aligned {len(whisper_chunks)} Whisper chunks to {len(raw_segments)} speaker segments")

        # ── Step 4: Per-Segment ASR (secondary reference) ──
        logger.info("   Running per-segment ASR for secondary reference...")
        segments_for_llm: list[dict] = []

        for idx, seg in enumerate(raw_segments):
            try:
                # Extract audio slice for this speaker turn
                start_frame = int(seg["start"] * 16000)
                num_frames = int((seg["end"] - seg["start"]) * 16000)
                seg_waveform = waveform[:, start_frame : start_frame + num_frames]

                if seg_waveform.shape[1] < 1600:  # < 0.1s — skip
                    continue

                # Convert to numpy for ASR pipeline
                audio_array = seg_waveform.squeeze().numpy().astype(np.float32)

                # Transcribe using existing Whisper Sinhala pipeline
                asr_result = asr_pipe(
                    {"raw": audio_array, "sampling_rate": 16000},
                )
                text = asr_result["text"].strip()

                # Prepare for LLM correction: combine aligned + per-segment text
                aligned_text = aligned_segments[idx]["text"] if idx < len(aligned_segments) else ""
                segments_for_llm.append({
                    "speaker": seg["speaker"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": text if text else "",
                    "aligned_text": aligned_text,
                })

            except Exception as e:
                logger.warning(
                    f"   ASR failed for segment "
                    f"{seg['start']:.1f}-{seg['end']:.1f}s: {e}"
                )
                # Still add aligned text if available
                aligned_text = aligned_segments[idx]["text"] if idx < len(aligned_segments) else ""
                if aligned_text:
                    segments_for_llm.append({
                        "speaker": seg["speaker"],
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": "",
                        "aligned_text": aligned_text,
                    })
                continue

        asr_time_ms = (time.time() - asr_start) * 1000
        logger.info(
            f"   ASR complete: {len(segments_for_llm)} segments in {asr_time_ms:.0f}ms"
        )

        # ── Step 5: Detect Overlapping Speaker Regions ──
        overlapping_regions = _detect_overlapping_regions(raw_segments)
        if overlapping_regions:
            logger.info(
                f"   Detected {len(overlapping_regions)} overlapping region(s): "
                + ", ".join(f"{o.start:.1f}-{o.end:.1f}s" for o in overlapping_regions)
            )

        # ── Step 6: LLM-Based Transcript Correction ──
        llm_start = time.time()
        final_segments: list[DiarizeSegment] = []

        if enable_llm_optimization and segments_for_llm:
            logger.info(
                f"🤖 Running LLM transcript correction on {len(segments_for_llm)} segments..."
            )
            corrected = _llm_correct_transcript(
                segments_for_llm,
                full_audio_text,
                overlapping_regions,
            )

            for i, seg_data in enumerate(corrected):
                raw_asr = segments_for_llm[i]["text"] if i < len(segments_for_llm) else ""
                ai_text = seg_data["text"]
                wd = seg_data.get("word_details", [])
                total_w = len(wd)
                meaningful_w = sum(1 for d in wd if d["is_meaningful"] or d["probability"] > 0.01)
                corrected_w = sum(1 for d in wd if d["was_corrected"])
                accuracy = round((meaningful_w / total_w * 100) if total_w > 0 else 0.0, 1)
                ws_list = [
                    WordScore(
                        word=d["word"],
                        original_word=d["original_word"],
                        probability=d["probability"],
                        is_meaningful=d["is_meaningful"],
                        was_corrected=d["was_corrected"],
                    ) for d in wd
                ]
                if raw_asr or ai_text:
                    final_segments.append(
                        DiarizeSegment(
                            speaker=seg_data["speaker"],
                            start=seg_data["start"],
                            end=seg_data["end"],
                            duration=round(seg_data["end"] - seg_data["start"], 3),
                            asr_text=raw_asr if raw_asr else "(empty)",
                            ai_corrected_text=ai_text if ai_text else raw_asr,
                            text=ai_text if ai_text else raw_asr,
                            word_scores=ws_list,
                            segment_accuracy_pct=accuracy,
                            total_words=total_w,
                            meaningful_words=meaningful_w,
                            corrected_words=corrected_w,
                        )
                    )
        else:
            # No LLM optimization — asr_text and ai_corrected_text are the same
            for seg_data in segments_for_llm:
                raw_asr = seg_data.get("text", "")
                aligned = seg_data.get("aligned_text", "")
                best = aligned or raw_asr
                if best:
                    words_in_seg = best.split()
                    default_ws = [
                        WordScore(
                            word=w, original_word=w,
                            probability=1.0, is_meaningful=True, was_corrected=False,
                        ) for w in words_in_seg
                    ]
                    final_segments.append(
                        DiarizeSegment(
                            speaker=seg_data["speaker"],
                            start=seg_data["start"],
                            end=seg_data["end"],
                            duration=round(seg_data["end"] - seg_data["start"], 3),
                            asr_text=raw_asr if raw_asr else "(empty)",
                            ai_corrected_text=best,
                            text=best,
                            word_scores=default_ws,
                            segment_accuracy_pct=100.0,
                            total_words=len(words_in_seg),
                            meaningful_words=len(words_in_seg),
                            corrected_words=0,
                        )
                    )

        llm_time_ms = (time.time() - llm_start) * 1000
        total_ms = (time.time() - total_start) * 1000

        # Build human-readable transcript texts
        ai_transcript_lines = []
        raw_transcript_lines = []
        for seg in final_segments:
            ts = f"[{seg.start:.1f}s - {seg.end:.1f}s]"
            ai_transcript_lines.append(f"{ts} {seg.speaker}: {seg.ai_corrected_text}")
            raw_transcript_lines.append(f"{ts} {seg.speaker}: {seg.asr_text}")
        transcript_text = "\n".join(ai_transcript_lines)
        raw_transcript_text = "\n".join(raw_transcript_lines)

        all_speakers = sorted(set(s.speaker for s in final_segments))

        # Calculate overall accuracy stats
        grand_total_words = sum(s.total_words for s in final_segments)
        grand_meaningful = sum(s.meaningful_words for s in final_segments)
        grand_corrected = sum(s.corrected_words for s in final_segments)
        grand_accuracy = round(
            (grand_meaningful / grand_total_words * 100) if grand_total_words > 0 else 0.0, 1
        )

        logger.info(
            f"🗣️ Diarization complete: {len(all_speakers)} speakers, "
            f"{len(final_segments)} segments "
            f"| diar={diar_time_ms:.0f}ms asr={asr_time_ms:.0f}ms "
            f"llm={llm_time_ms:.0f}ms total={total_ms:.0f}ms"
        )

        return DiarizeResponse(
            num_speakers=len(all_speakers),
            speakers=all_speakers,
            total_segments=len(final_segments),
            segments=final_segments,
            full_audio_transcript=full_audio_text,
            overlapping_regions=overlapping_regions,
            transcript_text=transcript_text,
            raw_transcript_text=raw_transcript_text,
            audio_duration_s=round(audio_duration, 2),
            diarization_time_ms=round(diar_time_ms, 1),
            transcription_time_ms=round(asr_time_ms, 1),
            ai_optimization_time_ms=round(llm_time_ms, 1),
            total_time_ms=round(total_ms, 1),
            overall_accuracy_pct=grand_accuracy,
            total_words=grand_total_words,
            total_meaningful_words=grand_meaningful,
            total_corrected_words=grand_corrected,
        )

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(
            status_code=507,
            detail="GPU out of memory during diarization. Try shorter audio.",
        )
    except Exception as e:
        logger.error(f"Diarization error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Diarization failed: {str(e)}")
    finally:
        # Clean up temp files
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if processed_path and os.path.exists(processed_path):
            os.unlink(processed_path)


# === Main ===
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
