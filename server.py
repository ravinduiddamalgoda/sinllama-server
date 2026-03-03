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
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import Optional

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


# === Model Loading ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model, tokenizer, input_device

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

    yield  # Server runs here

    # Cleanup
    logger.info("🧹 Shutting down, releasing GPU memory...")
    del model, tokenizer
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
        description="ASR pronunciation result: 'correct', 'close', 'retry', 'no_speech', or 'skipped'",
    )
    attempt_number: int = Field(default=1, ge=1, le=10, description="Which attempt this is for the current word")
    session_words: list[str] = Field(default=[], description="All words in this session (e.g. ['බල්ලා','පූසා','අලියා'])")


class ChildChatResponse(BaseModel):
    response: str
    inference_time_ms: float


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


# === API Endpoints ===


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
    """Child conversation endpoint with Sinhala prompt formatting."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Build conversation history (keep last 4 turns for speed)
    history_text = ""
    if req.conversation_history:
        history_text = "\n".join(req.conversation_history[-4:])

    # Build context lines
    context_lines = []
    if req.current_word:
        context_lines.append(f"දැන් පුහුණු කරන වචනය: '{req.current_word}'")
    if req.target_phoneme:
        context_lines.append(f"ඉලක්ක ශබ්දය: {req.target_phoneme}")
    if req.pronunciation_result:
        result_map = {
            "correct": "දරුවා වචනය නිවැරදිව කිව්වා! ප්‍රශංසා කරන්න.",
            "close": "දරුවා ආසන්නව කිව්වා. දිරිමත් කරන්න, නැවත උත්සාහ කරන්න කියන්න.",
            "retry": "දරුවාට වචනය හරියට කියන්න බැරි උනා. බය නොවී ආයෙත් කියන්න කියන්න.",
            "no_speech": "දරුවා කතා කළේ නැහැ. දිරිමත් කරන්න, බය වෙන්න එපා කියන්න.",
            "skipped": "වචනය මඟ හැරියා. කමක් නැත කියන්න, ඊළඟ වචනයට යන්න.",
        }
        result_hint = result_map.get(req.pronunciation_result, "")
        if result_hint:
            context_lines.append(result_hint)
    if req.attempt_number > 1:
        context_lines.append(f"මෙය {req.attempt_number} වන උත්සාහයයි.")
    if req.session_words:
        context_lines.append(f"අද session වචන: {', '.join(req.session_words)}")

    context_text = "\n".join(context_lines)

    prompt = f"""ඔබ වයස 3ත් 10ත් අතර කුඩා දරුවන්ට සිංහල වචන උච්චාරණය කිරීමට උගන්වන මිත්‍රශීලී ගැහැණු ගුරුවරියකි.
ඔබේ නම 'අක්කා' කියලා දරුවන් කතා කරනවා. ඔබ ඉතා කරුණාවන්ත, ඉවසිලිවන්ත, දිරිමත් කරන ගුරුවරියකි.
දරුවාගේ නම: {req.child_name}
{context_text}
වැදගත් නීති:
- එක කෙටි වාක්‍යයක් පමණක් කියන්න.
- සරල, දරුවාට තේරෙන සිංහල වචන පමණක් භාවිතා කරන්න.
- 'හොඳයි!', 'වාව්!', 'ඔයාට පුළුවන්!', 'බය වෙන්න එපා!' වැනි දිරිගැන්වීම් භාවිතා කරන්න.
- දරුවා නිවැරදිව කිව්වොත් ප්‍රශංසා කරන්න.
- දරුවාට අමාරු නම් උච්චාරණ උපදෙස් දෙන්න.

පෙර සංවාදය:
{history_text}

දරුවා: {req.child_utterance}
ගුරුවරිය:"""

    # Generate — keep max_new_tokens low for fast conversational responses
    gen_req = GenerateRequest(
        prompt=prompt,
        max_new_tokens=35,
        temperature=0.3,
        top_p=0.85,
        top_k=40,
        repetition_penalty=1.3,
    )

    result = await generate_text(gen_req)

    # Clean response — stop at continuation markers
    response_text = result.generated_text
    for stop_token in ["දරුවා:", "\n\n", "ගුරුවරිය:", "\nදරුවා", "\nගුරුවරිය"]:
        if stop_token in response_text:
            response_text = response_text.split(stop_token)[0].strip()

    # Truncate to first sentence for quick chat-style reply
    response_text = _truncate_to_sentences(response_text, max_sentences=1)

    return ChildChatResponse(
        response=response_text,
        inference_time_ms=result.inference_time_ms,
    )


# === Main ===
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
