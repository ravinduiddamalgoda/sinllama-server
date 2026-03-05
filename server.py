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
from fastapi import FastAPI, HTTPException, Depends, Security, UploadFile, File, Form
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import Optional
import tempfile
import shutil
import json as _json

# === Configuration ===
BASE_MODEL_PATH = "/workspace/models/llama-3-8b-base"
ADAPTER_PATH = "/workspace/models/sinllama-adapter"
TOKENIZER_PATH = "/workspace/models/sinhala-tokenizer"
VOCAB_SIZE = 139336

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


# === Model Loading ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model, tokenizer, input_device, asr_pipe

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

    yield  # Server runs here

    # Cleanup
    logger.info("🧹 Shutting down, releasing GPU memory...")
    del model, tokenizer, asr_pipe
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


# === Helpers ===

import re


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

    # --- Session Complete ---
    if req.is_session_complete:
        total = len(req.session_words) if req.session_words else 3
        templates = [
            f"වාව් {name}! ඔයා අද වචන {req.words_completed}ක් හරියට කිව්වා! ඔයා හරිම දක්ෂයි! හෙට ආයෙත් ඉගෙන ගමු!",
            f"සුපිරි {name}! අද session එක ඉවරයි! {req.words_completed}/{total} වචන හරියට කිව්වා. ඔයා champion කෙනෙක්! හෙට ආයෙත් එමුද?",
            f"හොඳයි {name}! අද අපි වචන {req.words_completed}ක් ඉගෙන ගත්තා! ඔයාට පුළුවන් කියලා පෙන්නුවා! හෙට තව වචන ඉගෙන ගමු!",
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

    # --- Correct (last word, no next) ---
    if req.pronunciation_result == "correct":
        templates = [
            f"වාව් {name}! '{word}' හරියටම කිව්වා! {desc} ඔයා හරිම දක්ෂයි!" if desc else f"වාව් {name}! '{word}' හරියටම කිව්වා! ඔයා හරිම දක්ෂයි!",
            f"සුපිරි {name}! '{word}' නිවැරදියි! {desc} ඔයාට පුළුවන්!" if desc else f"සුපිරි {name}! '{word}' නිවැරදියි! ඔයාට පුළුවන්!",
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
        "gpu_count": torch.cuda.device_count(),
        "input_device": str(input_device) if input_device else None,
        "gpus": gpu_info,
    }


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


# === Main ===
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
