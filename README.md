# 🇱🇰 SinLlama Cloud API Server

> **Sinhala Large Language Model deployed on Vast.ai GPU cloud, serving a Raspberry Pi-based IoT conversational toy for children's phonology education.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Hardware Requirements](#hardware-requirements)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [Setup Guide](#setup-guide)
  - [1. Vast.ai Account Setup](#1-vastai-account-setup)
  - [2. Instance Rental & SSH](#2-instance-rental--ssh)
  - [3. Environment Setup](#3-environment-setup)
  - [4. HuggingFace & Model Download](#4-huggingface--model-download)
  - [5. Server Deployment](#5-server-deployment)
  - [6. Raspberry Pi Client Setup](#6-raspberry-pi-client-setup)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Cost Analysis](#cost-analysis)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project deploys **SinLlama** — the first large language model specifically extended for Sinhala — as a cloud-hosted REST API on Vast.ai GPU infrastructure. The API is consumed by a **Raspberry Pi-based IoT toy** designed to help Sri Lankan children (ages 3–5) practice Sinhala phonology through interactive conversation.

The system follows an **edge-cloud architecture**: the Raspberry Pi handles real-time audio I/O (speech-to-text, text-to-speech, LCD display), while the cloud GPU server handles computationally expensive LLM inference, returning generated Sinhala text responses in 1–2 seconds.

### Key Features

- **SinLlama 8B** inference with 4-bit quantization (NF4) for efficient GPU memory usage
- **Dual-GPU sharding** across 2× RTX 5060 Ti (16GB VRAM each)
- **FastAPI REST server** with authenticated endpoints for text generation and child conversation
- **Raspberry Pi client** with automatic offline fallback when the cloud is unreachable
- **Child-optimized conversation** prompting with target phoneme support
- **Conversation history management** for contextual multi-turn dialogue

---

## System Architecture

```
┌──────────────────────────────────────┐       HTTPS / REST API       ┌──────────────────────────────────────┐
│          RASPBERRY PI (Edge)         │ ◄──────────────────────────► │      VAST.AI GPU SERVER (Cloud)       │
│                                      │                              │                                      │
│  ┌──────────┐    ┌───────────────┐   │   POST /chat                 │   ┌──────────────────────────────┐   │
│  │Microphone│───►│  STT Engine   │   │   {child_utterance,          │   │       SinLlama 8B (4-bit)    │   │
│  └──────────┘    │(Whisper/AAAI) │   │    conversation_history,     │   │                              │   │
│                  └──────┬────────┘   │    child_name,               │   │  ┌────────────────────────┐  │   │
│                         │            │    target_phoneme}            │   │  │  GPU 0: RTX 5060 Ti    │  │   │
│                         ▼            │                              │   │  │  Layers 0-7 (~4.5 GB)  │  │   │
│              ┌──────────────────┐    │   Response:                   │   │  └────────────────────────┘  │   │
│              │  cloud_client.py │────┼──►{response,                  │   │  ┌────────────────────────┐  │   │
│              │  (REST Client)   │◄───┼── inference_time_ms}          │   │  │  GPU 1: RTX 5060 Ti    │  │   │
│              └──────┬───────────┘    │                              │   │  │  Layers 8-31 (~7.7 GB) │  │   │
│                     │                │                              │   │  └────────────────────────┘  │   │
│                     ▼                │                              │   │                              │   │
│              ┌──────────────┐        │                              │   │  FastAPI + Uvicorn           │   │
│              │  TTS Engine  │        │                              │   │  Port 8000                   │   │
│              │  (VITS)      │        │                              │   └──────────────────────────────┘   │
│              └──────┬───────┘        │                              │                                      │
│                     │                │                              │  Instance: #28332477                  │
│          ┌──────────┴──────────┐     │                              │  Location: Vietnam                    │
│          ▼                    ▼      │                              │  Cost: $0.148/hr                      │
│    ┌──────────┐        ┌─────────┐   │                              │                                      │
│    │ Speaker  │        │  LCD    │   │                              │                                      │
│    └──────────┘        │ Display │   │                              │                                      │
│                        └─────────┘   │                              │                                      │
└──────────────────────────────────────┘                              └──────────────────────────────────────┘
```

### Latency Breakdown (Sri Lanka ↔ Vietnam)

| Stage | Time |
|-------|------|
| Network: Pi → Vast.ai | ~100 ms |
| Model inference (4-bit, 150 tokens) | ~500–1500 ms |
| Network: Vast.ai → Pi | ~100 ms |
| **Total round-trip** | **~700–1700 ms** |

---

## Technology Stack

### Cloud Server (Vast.ai)

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.12 |
| API Framework | FastAPI | 0.115.0 |
| ASGI Server | Uvicorn | 0.30.0 |
| ML Framework | PyTorch | 2.1+ (CUDA 12.x) |
| Model Loading | Transformers (HuggingFace) | 4.44.0 |
| LoRA Adapter | PEFT | 0.12.0 |
| Quantization | bitsandbytes | 0.41.1 |
| Validation | Pydantic | 2.9.0 |

### Edge Device (Raspberry Pi)

| Component | Technology |
|-----------|-----------|
| HTTP Client | requests |
| STT | Whisper / AssemblyAI |
| TTS | VITS (Sinhala) |
| Display | LCD (I2C) |
| Language | Python 3.11+ |

---

## Hardware Requirements

### Cloud Server (Vast.ai Instance)

| Spec | Minimum | Recommended (Our Setup) |
|------|---------|------------------------|
| GPU | 1× 16GB VRAM | 2× RTX 5060 Ti (16GB each) |
| GPU VRAM Usage | ~5 GB (4-bit) | ~12 GB across 2 GPUs |
| System RAM | 32 GB | 64 GB |
| Disk | 50 GB | 80+ GB |
| Network | 100 Mbps | 500+ Mbps |
| CUDA | 12.0+ | 12.8+ |

### Raspberry Pi (Edge)

| Spec | Requirement |
|------|-------------|
| Model | Raspberry Pi 4B / 5 |
| RAM | 4 GB+ |
| Storage | 32 GB+ microSD |
| Network | WiFi or Ethernet |
| Peripherals | USB microphone, speaker, LCD |

---

## Model Details

### SinLlama v01

| Property | Value |
|----------|-------|
| **Base Model** | Meta-Llama-3-8B |
| **Adapter Type** | LoRA (PEFT) |
| **Architecture** | Decoder-only autoregressive transformer |
| **Parameters** | ~8 billion |
| **Language** | Sinhala (සිංහල) |
| **Tokenizer** | Extended Sinhala LLaMA (139,336 vocab) |
| **Training Data** | 10.7M Sinhala sentences (MADLAD-400 + CulturaX) |
| **Quantization** | NF4 (4-bit, double quantization) |
| **VRAM (4-bit)** | ~5 GB per GPU (sharded) |
| **Instruct-tuned** | ❌ No — base model, requires prompting or fine-tuning |
| **License** | Meta Llama 3 License |

**HuggingFace Repositories:**

| Component | Repository | Size |
|-----------|-----------|------|
| Base Model | [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) | ~15 GB |
| LoRA Adapter | [polyglots/SinLlama_v01](https://huggingface.co/polyglots/SinLlama_v01) | ~200 MB |
| Tokenizer | [polyglots/Extended-Sinhala-LLaMA](https://huggingface.co/polyglots/Extended-Sinhala-LLaMA) | ~5 MB |

**Citation:**
```bibtex
@article{sinllama2024,
  title={SinLlama: A Large Language Model for Sinhala},
  author={Aravinda, H.W.K. and Sirajudeen, Rashad and Karunathilake, Samith 
          and de Silva, Nisansa and Kaur, Rishemjit and Ranathunga, Surangika},
  journal={arXiv preprint arXiv:2508.09115},
  year={2024}
}
```

---

## Project Structure

```
sinllama-cloud-api/
│
├── README.md                    # This file
│
├── cloud-server/                # Runs on Vast.ai GPU instance
│   ├── server.py                # FastAPI server (2-GPU version)
│   ├── quick_setup.sh           # One-click setup script for new instances
│   └── requirements.txt         # Python dependencies
│
├── pi-client/                   # Runs on Raspberry Pi
│   ├── cloud_client.py          # REST API client with offline fallback
│   └── requirements.txt         # Python dependencies
│
└── docs/
    └── SinLlama_VastAI_Setup_Guide.md   # Detailed step-by-step guide
```

---

## Setup Guide

### Prerequisites

- Vast.ai account with $5+ credits
- HuggingFace account with Meta Llama 3 license accepted
- SSH key pair generated on your local machine
- Raspberry Pi with Python 3.11+ and network access

### 1. Vast.ai Account Setup

1. Sign up at [cloud.vast.ai](https://cloud.vast.ai)
2. Add minimum **$5** credit (Billing → Add Credit)
3. Add your SSH public key (Account → SSH Public Keys):
   ```bash
   # Generate key if needed
   ssh-keygen -t ed25519 -C "sinllama-vastai"
   cat ~/.ssh/id_ed25519.pub
   # Copy and paste into Vast.ai dashboard
   ```

### 2. Instance Rental & SSH

**Recommended instance:** Vietnam, 2× RTX 5060 Ti (#28332477) — $0.148/hr

1. Go to [cloud.vast.ai/create](https://cloud.vast.ai/create/)
2. Filter: GPU = RTX 5060 Ti, VRAM ≥ 16 GB
3. Select PyTorch template, set disk to **80 GB**
4. Click **RENT**
5. Connect via SSH:
   ```bash
   ssh -p <PORT> root@<VAST_IP>
   
   # Verify GPUs
   nvidia-smi
   # Should show 2× RTX 5060 Ti
   ```

### 3. Environment Setup

```bash
# Update and install dependencies
apt-get update && apt-get install -y nano wget curl

pip install --upgrade pip
pip install transformers==4.44.0 accelerate==0.33.0 peft==0.12.0 \
    bitsandbytes==0.41.1 fastapi==0.115.0 uvicorn==0.30.0 \
    pydantic==2.9.0 huggingface_hub

# Verify
python3 -c "
import torch, transformers, peft, bitsandbytes, fastapi
print(f'PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}')
print(f'Transformers {transformers.__version__} | PEFT {peft.__version__}')
"
```

### 4. HuggingFace & Model Download

**Before downloading — do these in your browser:**

1. Accept Meta Llama 3 license: [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) → "Agree and access repository"
2. Create a read token: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

```bash
# Login
huggingface-cli login
# Paste your hf_... token

# Download all 3 components (~15 GB total, takes 5-15 min)
mkdir -p /workspace/models

python3 -c "
from huggingface_hub import snapshot_download

print('📥 Downloading Meta-Llama-3-8B...')
snapshot_download('meta-llama/Meta-Llama-3-8B', local_dir='/workspace/models/llama-3-8b-base')

print('📥 Downloading SinLlama adapter...')
snapshot_download('polyglots/SinLlama_v01', local_dir='/workspace/models/sinllama-adapter')

print('📥 Downloading Sinhala tokenizer...')
snapshot_download('polyglots/Extended-Sinhala-LLaMA', local_dir='/workspace/models/sinhala-tokenizer')

print('✅ All models downloaded!')
"

# Verify
du -sh /workspace/models/*
```

### 5. Server Deployment

```bash
# Create server directory
mkdir -p /workspace/sinllama-server

# Copy server.py (download from this repo or create manually)
# See cloud-server/server.py

# Set environment variables
export SINLLAMA_API_KEY="your-secret-key-2024"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Start server (foreground for testing)
cd /workspace/sinllama-server
python3 server.py

# Or start in background (production)
nohup python3 server.py > /workspace/server.log 2>&1 &
```

**Test the server:**

```bash
# Health check
curl http://localhost:8000/health | python3 -m json.tool

# Text generation
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key-2024" \
  -d '{"prompt": "සිංහල භාෂාව ඉගෙන ගනිමු", "max_new_tokens": 100}'

# Child conversation
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key-2024" \
  -d '{
    "child_utterance": "මට බල්ලෙක් ඕනේ",
    "child_name": "කවිඳු",
    "target_phoneme": "බ"
  }'
```

### 6. Raspberry Pi Client Setup

```bash
# On the Raspberry Pi
pip3 install requests
mkdir -p ~/sinllama-toy

# Copy cloud_client.py from this repo to ~/sinllama-toy/
# Edit the configuration:
nano ~/sinllama-toy/cloud_client.py
```

Update these values in `cloud_client.py`:

```python
API_URL = "http://<VAST_PUBLIC_IP>:<MAPPED_PORT>"  # From Vast.ai dashboard
API_KEY = "your-secret-key-2024"                    # Must match server
```

**Test:**

```bash
python3 ~/sinllama-toy/cloud_client.py
# Interactive test mode — type Sinhala text and get responses
```

---

## API Reference

### `GET /health`

Health check endpoint. No authentication required.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_count": 2,
  "input_device": "cuda:0",
  "gpus": [
    {
      "gpu_id": 0,
      "name": "NVIDIA GeForce RTX 5060 Ti",
      "vram_used_gb": 4.53,
      "vram_reserved_gb": 12.07,
      "vram_total_gb": 16.0
    },
    {
      "gpu_id": 1,
      "name": "NVIDIA GeForce RTX 5060 Ti",
      "vram_used_gb": 7.71,
      "vram_reserved_gb": 8.85,
      "vram_total_gb": 16.0
    }
  ]
}
```

---

### `POST /generate`

Raw text generation. Requires `X-API-Key` header.

**Request:**
```json
{
  "prompt": "සිංහල භාෂාව",
  "max_new_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1
}
```

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `prompt` | string | *required* | — | Input text prompt |
| `max_new_tokens` | int | 256 | 1–1024 | Maximum tokens to generate |
| `temperature` | float | 0.7 | 0.1–2.0 | Sampling randomness |
| `top_p` | float | 0.9 | 0.0–1.0 | Nucleus sampling threshold |
| `top_k` | int | 50 | 0–200 | Top-k sampling |
| `repetition_penalty` | float | 1.1 | 1.0–2.0 | Penalize repeated tokens |

**Response:**
```json
{
  "generated_text": "සිංහල භාෂාවේ ඇති 'අ' සහ 'ආ' අකුරු දෙකේ වෙනස...",
  "tokens_generated": 87,
  "inference_time_ms": 1234.5
}
```

---

### `POST /chat`

Child conversation endpoint with Sinhala prompt formatting. Requires `X-API-Key` header.

**Request:**
```json
{
  "child_utterance": "මට බල්ලෙක් ඕනේ",
  "conversation_history": [
    "දරුවා: හෙලෝ",
    "සහකාරයා: හෙලෝ! කොහොමද?"
  ],
  "child_name": "කවිඳු",
  "target_phoneme": "බ"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `child_utterance` | string | *required* | Child's speech (Sinhala text from STT) |
| `conversation_history` | list[str] | `[]` | Previous turns (last 6 kept) |
| `child_name` | string | `"ළමයා"` | Child's name in Sinhala |
| `target_phoneme` | string | `""` | Phoneme being practiced |

**Response:**
```json
{
  "response": "බල්ලෙක් ඕනේද? ඒක හොඳ අදහසක්! බල්ලෙකුට මොකක්ද නමක් දාන්නේ?",
  "inference_time_ms": 892.3
}
```

---

### Error Codes

| Code | Meaning |
|------|---------|
| `200` | Success |
| `403` | Invalid or missing API key |
| `500` | Server error during generation |
| `503` | Model not yet loaded (server still starting) |
| `507` | GPU out of memory |

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SINLLAMA_API_KEY` | Yes | `sinllama-default-key-change-me` | API authentication key |
| `PYTORCH_CUDA_ALLOC_CONF` | Recommended | — | Set to `expandable_segments:True,max_split_size_mb:128` |
| `PORT` | No | `8000` | Server port |

### Model Paths (Configured in server.py)

| Path | Description |
|------|-------------|
| `/workspace/models/llama-3-8b-base` | Meta-Llama-3-8B base model |
| `/workspace/models/sinllama-adapter` | SinLlama LoRA adapter weights |
| `/workspace/models/sinhala-tokenizer` | Extended Sinhala tokenizer |

---

## Cost Analysis

### Instance: Vietnam 2× RTX 5060 Ti ($0.148/hr)

| Usage Pattern | Hours/Day | Monthly Cost | $10 Lasts |
|---------------|-----------|-------------|-----------|
| Light testing | 1 hr | $4.44 | ~67 hours |
| Active development | 3 hrs | $13.32 | ~22 days |
| Heavy development | 5 hrs | $22.20 | ~13 days |
| Demo day | 8 hrs (1 day) | $1.18 | — |

### Full Project Estimate (3-Month Capstone)

| Phase | Duration | Usage | Cost |
|-------|----------|-------|------|
| Setup & testing | 1 week | 2 hrs/day | ~$2.07 |
| Active development | 8 weeks | 3 hrs/day | ~$24.86 |
| Testing & refinement | 2 weeks | 2 hrs/day | ~$4.14 |
| Demo & presentation | 2 days | 5 hrs/day | ~$1.48 |
| Storage (stopped instances) | ~$1–3/month | — | ~$6.00 |
| **Total** | **~3 months** | — | **~$38.55** |

### Cost-Saving Tips

1. **STOP the instance** when not in use (Dashboard → Stop). Storage charges continue but GPU billing stops.
2. **DESTROY the instance** if you won't use it for days. Re-setup takes ~20 minutes.
3. **Monitor credits** at [cloud.vast.ai/billing](https://cloud.vast.ai/billing/)
4. **Start small** — $10 deposit gives ~67 hours, enough for initial development.

---

## Troubleshooting

### Server Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: triton.ops` | `pip install bitsandbytes==0.41.1 triton==2.1.0` |
| `CUDA out of memory` | Already using 4-bit quantization. Reduce `max_new_tokens` or prompt length. |
| Server starts but generation is slow (>10s) | Check `nvidia-smi` for other processes using GPU. Kill them. |
| `403 Forbidden` on API calls | API key mismatch. Ensure `X-API-Key` header matches `SINLLAMA_API_KEY` env var. |
| Model download returns `403` | Accept Meta Llama 3 license at HuggingFace first. |

### Pi Client Issues

| Problem | Solution |
|---------|----------|
| `ConnectionError` from Pi | Check Vast.ai instance is running. Verify IP and mapped port in dashboard. |
| High latency (>3s) | Normal for first request (model warmup). Subsequent requests should be 1–2s. |
| Fallback responses only | Server is unreachable. Check instance status, network, and firewall. |
| `Timeout` errors | Increase `TIMEOUT` in `cloud_client.py` or check server logs for errors. |

### Quick Diagnostic Commands

```bash
# On Vast.ai instance:
nvidia-smi                              # Check GPU usage
curl http://localhost:8000/health        # Check server status
tail -f /workspace/server.log           # View server logs
ps aux | grep python                    # Check running processes

# On Raspberry Pi:
curl http://<VAST_IP>:<PORT>/health     # Test connectivity
ping <VAST_IP>                          # Check network latency
python3 ~/sinllama-toy/cloud_client.py  # Interactive test
```

---

## Quick Command Reference

```bash
# ══════════════════════════════════════════
# VAST.AI INSTANCE — Start Session
# ══════════════════════════════════════════

# SSH into instance
ssh -p <PORT> root@<VAST_IP>

# Set environment
export SINLLAMA_API_KEY="your-secret-key-2024"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Start server (foreground)
cd /workspace/sinllama-server && python3 server.py

# Start server (background)
cd /workspace/sinllama-server
nohup python3 server.py > /workspace/server.log 2>&1 &

# Stop server
pkill -f "python3 server.py"

# ══════════════════════════════════════════
# RASPBERRY PI — Connect & Test
# ══════════════════════════════════════════

# Quick health check
curl http://<VAST_IP>:<PORT>/health

# Interactive test
python3 ~/sinllama-toy/cloud_client.py

# ══════════════════════════════════════════
# COST MANAGEMENT
# ══════════════════════════════════════════

# ALWAYS stop/destroy instance when done!
# Dashboard: cloud.vast.ai/instances → Stop or Destroy
```

---

## Contributors

| Name | Role |
|------|------|
| Ravindu | Senior Backend Software Engineer, SLIIT Student — System design, implementation, deployment |

---

## Acknowledgements

- **[SinLlama](https://huggingface.co/polyglots/SinLlama_v01)** by H.W.K. Aravinda, Rashad Sirajudeen, Samith Karunathilake, Nisansa de Silva, Rishemjit Kaur, Surangika Ranathunga — Polyglots Research Team
- **[Meta Llama 3](https://huggingface.co/meta-llama/Meta-Llama-3-8B)** by Meta AI — Base model
- **[Vast.ai](https://vast.ai)** — Affordable GPU cloud infrastructure
- **[SLIIT](https://www.sliit.lk/)** — Faculty of Computing, Sri Lanka Institute of Information Technology
- **[HuggingFace](https://huggingface.co)** — Model hosting and transformers library

---

## License

This project code is licensed under the MIT License. The SinLlama model follows the [Meta Llama 3 License](https://llama.meta.com/llama3/license/).

---

> **Built with ❤️ in Sri Lanka 🇱🇰 for Sinhala-speaking children**
