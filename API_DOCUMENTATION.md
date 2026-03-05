# SinLlama API Documentation

> **Version:** 2.0.0  
> **Description:** Sinhala LLM inference + ASR API for IoT child pronunciation education toy  
> **Hardware:** 2× NVIDIA RTX 5060 Ti (16 GB each) on Vast.ai  
> **Models:** Meta-Llama-3-8B + SinLlama LoRA (LLM) | Whisper Small Sinhala (ASR)  
> **Base URL:** `http://<server-host>:8000`

---

## Table of Contents

- [Authentication](#authentication)
- [Endpoints Overview](#endpoints-overview)
- [Endpoints](#endpoints)
  - [GET /health](#1-get-health)
  - [POST /generate](#2-post-generate)
  - [POST /ask](#3-post-ask)
  - [POST /child-chat](#4-post-child-chat)
  - [POST /transcribe](#5-post-transcribe)
  - [POST /child-session](#6-post-child-session-recommended)
- [Pronunciation Result Values](#pronunciation-result-values)
- [Session Flow Guide](#session-flow-guide)
- [Word Knowledge Base](#word-knowledge-base)
- [Error Responses](#error-responses)
- [Examples (cURL)](#examples-curl)
- [Examples (Python)](#examples-python)
- [Examples (Raspberry Pi Client)](#examples-raspberry-pi-client)
- [Environment Variables](#environment-variables)

---

## Authentication

All endpoints except `/health` require an API key via the `X-API-Key` header.

| Header      | Value        |
| ----------- | ------------ |
| `X-API-Key` | Your API key |

Set the key on the server: `export SINLLAMA_API_KEY="your-secret-key"`  
Default (dev only): `sinllama-default-key-change-me`

---

## Endpoints Overview

| Endpoint | Method | Auth | Content-Type | Description |
|---|---|---|---|---|
| `/health` | GET | No | — | Health check + GPU status |
| `/generate` | POST | Yes | `application/json` | Raw text generation |
| `/ask` | POST | Yes | `application/json` | General-purpose Sinhala chat |
| `/child-chat` | POST | Yes | `application/json` | Child pronunciation chat (text input) |
| `/transcribe` | POST | Yes | `multipart/form-data` | Speech-to-text (Sinhala ASR) |
| `/child-session` | POST | Yes | `multipart/form-data` | **Combined ASR + child chat (recommended)** |

---

## Endpoints

### 1. `GET /health`

Health check and GPU status. **No authentication required.**

#### Response `200 OK`

```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_count": 2,
  "input_device": "cuda:0",
  "gpus": [
    {
      "gpu_id": 0,
      "name": "NVIDIA RTX 5060 Ti",
      "vram_used_gb": 5.12,
      "vram_reserved_gb": 5.50,
      "vram_total_gb": 16.0
    },
    {
      "gpu_id": 1,
      "name": "NVIDIA RTX 5060 Ti",
      "vram_used_gb": 4.88,
      "vram_reserved_gb": 5.20,
      "vram_total_gb": 16.0
    }
  ]
}
```

---

### 2. `POST /generate`

Raw text generation. Sends a prompt and returns the model's continuation. **Authentication required.**

#### Request

```
POST /generate
Content-Type: application/json
X-API-Key: <your-key>
```

#### Request Body

```json
{
  "prompt": "ශ්‍රී ලංකාව",
  "max_new_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1
}
```

| Field | Type | Required | Default | Range | Description |
|---|---|---|---|---|---|
| `prompt` | string | **Yes** | — | — | Input text prompt |
| `max_new_tokens` | int | No | 256 | 1–1024 | Max tokens to generate |
| `temperature` | float | No | 0.7 | 0.1–2.0 | Sampling temperature |
| `top_p` | float | No | 0.9 | 0.0–1.0 | Nucleus sampling threshold |
| `top_k` | int | No | 50 | 0–200 | Top-K sampling |
| `repetition_penalty` | float | No | 1.1 | 1.0–2.0 | Repetition penalty |

#### Response `200 OK`

```json
{
  "generated_text": "ශ්‍රී ලංකාව ඉන්දියානු සාගරයේ පිහිටි දිවයිනකි.",
  "tokens_generated": 42,
  "inference_time_ms": 1250.3
}
```

---

### 3. `POST /ask`

General-purpose Sinhala chat with system prompt and conversation history. **Authentication required.**

#### Request

```
POST /ask
Content-Type: application/json
X-API-Key: <your-key>
```

#### Request Body

```json
{
  "message": "ශ්‍රී ලංකාවේ අගනුවර කුමක්ද?",
  "system_prompt": "ඔබ සිංහල භාෂාවෙන් උත්තර දෙන බුද්ධිමත් සහකාරයෙකි.",
  "conversation_history": [
    { "role": "user", "content": "හෙලෝ" },
    { "role": "assistant", "content": "ආයුබෝවන්!" }
  ],
  "max_new_tokens": 80,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.2
}
```

| Field | Type | Required | Default | Range | Description |
|---|---|---|---|---|---|
| `message` | string | **Yes** | — | — | User's message |
| `system_prompt` | string | No | Sinhala default | — | System behavior prompt |
| `conversation_history` | list[object] | No | `[]` | last 10 | `[{"role":"user"\|"assistant","content":"..."}]` |
| `max_new_tokens` | int | No | 80 | 1–1024 | Max tokens |
| `temperature` | float | No | 0.7 | 0.1–2.0 | Sampling temperature |
| `top_p` | float | No | 0.9 | 0.0–1.0 | Nucleus sampling |
| `top_k` | int | No | 50 | 0–200 | Top-K sampling |
| `repetition_penalty` | float | No | 1.2 | 1.0–2.0 | Repetition penalty |

#### Response `200 OK`

```json
{
  "reply": "ශ්‍රී ජයවර්ධනපුර කෝට්ටේ වේ.",
  "tokens_generated": 35,
  "inference_time_ms": 980.5
}
```

---

### 4. `POST /child-chat`

Child pronunciation practice endpoint (text input). Uses **pre-built templates** for pronunciation scenarios (instant response, <1ms) and falls back to LLM for free conversation. **Authentication required.**

> **Note:** For Raspberry Pi integration, prefer [`/child-session`](#6-post-child-session-recommended) which combines ASR + chat in one call.

#### Request

```
POST /child-chat
Content-Type: application/json
X-API-Key: <your-key>
```

#### Request Body

```json
{
  "child_utterance": "බල්ලා",
  "child_name": "කවිඳු",
  "current_word": "බල්ලා",
  "pronunciation_result": "correct",
  "attempt_number": 1,
  "session_words": ["බල්ලා", "පූසා", "අලියා"],
  "next_word": "පූසා",
  "words_completed": 1,
  "is_session_complete": false,
  "conversation_history": [],
  "target_phoneme": ""
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `child_utterance` | string | **Yes** | — | What the child said (text) |
| `child_name` | string | No | `"ළමයා"` | Child's name for personalization |
| `current_word` | string | No | `""` | Word being practiced |
| `pronunciation_result` | string | No | `""` | See [Pronunciation Result Values](#pronunciation-result-values) |
| `attempt_number` | int | No | 1 | Which attempt (1–10) |
| `session_words` | list[str] | No | `[]` | All words in this session |
| `next_word` | string | No | `""` | Next word to transition to |
| `words_completed` | int | No | 0 | Words completed so far |
| `is_session_complete` | bool | No | false | True when all words done |
| `conversation_history` | list[str] | No | `[]` | Previous conversation turns |
| `target_phoneme` | string | No | `""` | Target phoneme for practice |

#### Response `200 OK`

```json
{
  "response": "වාව් කවිඳු! 'බල්ලා' හරියටම කිව්වා! බල්ලා වව් වව් කියනවා! බල්ලා ගෙදර රකිනවා, බල්ලා අපේ යාළුවා. දැන් අපි 'පූසා' ගැන ඉගෙන ගමු. පූසා මියාව් මියාව් කියනවා! පූසා කිරි බොනවා, පූසා මීයන් අල්ලනවා. කවිඳු, 'පූසා' කියන්න පුළුවන්ද?",
  "inference_time_ms": 0.3
}
```

| Field | Type | Description |
|---|---|---|
| `response` | string | Teacher's child-friendly Sinhala response |
| `inference_time_ms` | float | Time in milliseconds (near-instant for templates) |

---

### 5. `POST /transcribe`

Sinhala speech-to-text using Whisper Small Sinhala model. **Authentication required.**

#### Request

```
POST /transcribe
Content-Type: multipart/form-data
X-API-Key: <your-key>
```

| Field | Type | Required | Description |
|---|---|---|---|
| `audio` | file | **Yes** | Audio file (wav, mp3, flac, ogg, webm) |

#### Response `200 OK`

```json
{
  "text": "බල්ලා",
  "language": "si",
  "inference_time_ms": 850.2
}
```

| Field | Type | Description |
|---|---|---|
| `text` | string | Transcribed Sinhala text |
| `language` | string | Language code (`"si"`) |
| `inference_time_ms` | float | ASR inference time (ms) |

---

### 6. `POST /child-session` (Recommended)

**Combined ASR + child chat in a single API call.** This is the **recommended endpoint for Raspberry Pi** — send the child's audio recording along with session metadata and receive both the transcription and teacher's response in one round trip.

This saves a full network round trip compared to calling `/transcribe` + `/child-chat` separately.

#### Request

```
POST /child-session
Content-Type: multipart/form-data
X-API-Key: <your-key>
```

All fields are sent as **multipart form fields** alongside the audio file.

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `audio` | file | **Yes** | — | Child's audio recording (wav, mp3, flac, ogg, webm) |
| `child_name` | string | No | `"ළමයා"` | Child's name |
| `current_word` | string | No | `""` | Word being practiced |
| `pronunciation_result` | string | No | `""` | See [Pronunciation Result Values](#pronunciation-result-values) |
| `attempt_number` | int | No | 1 | Which attempt (1–10) |
| `session_words` | string (JSON) | No | `"[]"` | JSON array: `'["බල්ලා","පූසා","අලියා"]'` |
| `next_word` | string | No | `""` | Next word to transition to |
| `words_completed` | int | No | 0 | Words completed so far |
| `is_session_complete` | bool | No | false | True when all words done |
| `conversation_history` | string (JSON) | No | `"[]"` | JSON array of previous turns |
| `target_phoneme` | string | No | `""` | Target phoneme for practice |

> **Important:** `session_words` and `conversation_history` must be **JSON-encoded strings** in form data (e.g., `'["බල්ලා","පූසා"]'`).

#### Response `200 OK`

```json
{
  "transcribed_text": "බල්ලා",
  "teacher_response": "වාව් කවිඳු! 'බල්ලා' හරියටම කිව්වා! බල්ලා වව් වව් කියනවා! බල්ලා ගෙදර රකිනවා, බල්ලා අපේ යාළුවා. දැන් අපි 'පූසා' ගැන ඉගෙන ගමු. පූසා මියාව් මියාව් කියනවා! පූසා කිරි බොනවා, පූසා මීයන් අල්ලනවා. කවිඳු, 'පූසා' කියන්න පුළුවන්ද?",
  "asr_time_ms": 850.2,
  "chat_time_ms": 0.3,
  "total_time_ms": 852.1
}
```

| Field | Type | Description |
|---|---|---|
| `transcribed_text` | string | What Whisper heard the child say |
| `teacher_response` | string | Teacher's conversational reply |
| `asr_time_ms` | float | Speech-to-text time (ms) |
| `chat_time_ms` | float | Response generation time (ms) |
| `total_time_ms` | float | Total end-to-end time (ms) |

---

## Pronunciation Result Values

| Value | When to Send | Teacher Behavior |
|---|---|---|
| `new_word` | Introducing a new word | Describes the word with fun facts + asks child to say it |
| `correct` | Child said it right | Praises + word description + transitions to `next_word` with description |
| `close` | Almost correct | Encourages + word description + asks to retry same word |
| `retry` | Incorrect pronunciation | Reassures + word description + asks to retry same word |
| `no_speech` | No speech detected | Encourages child + word description + asks to try |
| `skipped` | Word was skipped | Accepts + introduces `next_word` with description |
| *(empty)* | Free conversation | Uses LLM for open-ended chat |

---

## Session Flow Guide

A typical pronunciation practice session with 3 words:

```
Step 1: Introduce first word
  → pronunciation_result="new_word", current_word="බල්ලා"
  ← "කවිඳු, අද අපි 'බල්ලා' ගැන ඉගෙන ගමු! බල්ලා වව් වව් කියනවා!
     බල්ලා ගෙදර රකිනවා, බල්ලා අපේ යාළුවා. 'බල්ලා' කියන්න පුළුවන්ද?"

Step 2: Child attempts "බල්ලා" → CORRECT
  → pronunciation_result="correct", current_word="බල්ලා", next_word="පූසා", words_completed=1
  ← "වාව් කවිඳු! 'බල්ලා' හරියටම කිව්වා! බල්ලා වව් වව් කියනවා!
     දැන් අපි 'පූසා' ගැන ඉගෙන ගමු. පූසා මියාව් මියාව් කියනවා!
     කවිඳු, 'පූසා' කියන්න පුළුවන්ද?"

Step 2b: If WRONG → retry
  → pronunciation_result="retry", current_word="බල්ලා", attempt_number=2
  ← "කමක් නැහැ කවිඳු! අපි ආයෙත් උත්සාහ කරමු. බල්ලා වව් වව් කියනවා!
     'බල්ලා' කියන්න පුළුවන්ද?"

Step 2c: If NO SPEECH
  → pronunciation_result="no_speech", current_word="බල්ලා"
  ← "කවිඳු, බය වෙන්න එපා! අපි එකට ඉගෙන ගමු. බල්ලා වව් වව් කියනවා!
     'බල්ලා' කියන්න පුළුවන්ද?"

Step 3: Child attempts "පූසා" → CORRECT
  → pronunciation_result="correct", current_word="පූසා", next_word="අලියා", words_completed=2
  ← "සුපිරි කවිඳු! 'පූසා' නිවැරදියි! දැන් අපි 'අලියා' ගැන ඉගෙන ගමු.
     අලියා ඉතාම ලොකු සතෙක්! අලියාට දිග උඹලක් තියෙනවා.
     කවිඳු, 'අලියා' කියන්න පුළුවන්ද?"

Step 4: Child attempts "අලියා" → CORRECT (last word)
  → pronunciation_result="correct", current_word="අලියා", words_completed=3
  ← "වාව් කවිඳු! 'අලියා' හරියටම කිව්වා! ඔයා හරිම දක්ෂයි!"

Step 5: Session complete
  → is_session_complete=true, words_completed=3
  ← "සුපිරි කවිඳු! අද session එක ඉවරයි! 3/3 වචන හරියට කිව්වා.
     ඔයා champion කෙනෙක්! හෙට ආයෙත් එමුද?"
```

---

## Word Knowledge Base

The server has a built-in knowledge base of **100+ Sinhala words** with child-friendly descriptions. These are injected directly into responses (no LLM call needed), covering:

| Category | Example Words |
|---|---|
| **Animals** | බල්ලා, පූසා, අලියා, සිංහයා, හාවා, සමනළයා, කුරුල්ලා, මාළුවා, ගවයා, එළුවා, ඉබ්බා, කිඹුලා, කපුටා, කොකා, කුකුළා, මොනරා, මුවා |
| **Body Parts** | අත, ඇස, කන, නාසය, මුඛය, පාදය, බඩ, ඇඟිල්ල, දත, බෙල්ල, කොණ්ඩය, දිව |
| **Food & Drinks** | බත්, වතුර, කිරි, පාන්, බිත්තරය, අඹ, කෙසෙල් |
| **Family** | අම්මා, තාත්තා, අක්කා, අයියා, නංගි, මල්ලි, ආච්චි, සීයා |
| **Nature** | ඉර, සඳ, තරු, වැස්ස |
| **Colors** | රතු, නිල්, කහ, කොළ, සුදු |
| **Places & Things** | ගෙදර, පාසල, පොත, ගස, මල, කෝච්චි, බස්, පාර, බෝලය, ... |

Words in the knowledge base get **instant responses** (<1ms). Words not in the base still work with generic templates.

---

## Error Responses

| Status Code | Condition | Response |
|---|---|---|
| **400** | Unsupported audio type | `{"detail": "Unsupported audio type: ..."}` |
| **403** | Missing/invalid API key | `{"detail": "Invalid or missing API key. Pass X-API-Key header."}` |
| **500** | Internal error | `{"detail": "<error message>"}` |
| **503** | Model still loading | `{"detail": "Model not loaded yet."}`  or `{"detail": "ASR model not loaded yet."}` |
| **507** | GPU out of memory | `{"detail": "GPU out of memory. Try shorter prompt or fewer tokens."}` |

---

## Examples (cURL)

### Health Check

```bash
curl http://localhost:8000/health
```

### Raw Generate

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sinllama-default-key-change-me" \
  -d '{
    "prompt": "ශ්‍රී ලංකාව",
    "max_new_tokens": 100,
    "temperature": 0.7
  }'
```

### General Ask

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sinllama-default-key-change-me" \
  -d '{
    "message": "ශ්‍රී ලංකාවේ අගනුවර කුමක්ද?"
  }'
```

### Transcribe Audio

```bash
curl -X POST http://localhost:8000/transcribe \
  -H "X-API-Key: sinllama-default-key-change-me" \
  -F "audio=@recording.wav"
```

### Child Chat (text input)

```bash
curl -X POST http://localhost:8000/child-chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sinllama-default-key-change-me" \
  -d '{
    "child_utterance": "බල්ලා",
    "child_name": "කවිඳු",
    "current_word": "බල්ලා",
    "pronunciation_result": "correct",
    "next_word": "පූසා",
    "session_words": ["බල්ලා","පූසා","අලියා"],
    "words_completed": 1
  }'
```

### Child Session — Introduce New Word

```bash
curl -X POST http://localhost:8000/child-session \
  -H "X-API-Key: sinllama-default-key-change-me" \
  -F "audio=@silence.wav" \
  -F "child_name=කවිඳු" \
  -F "current_word=බල්ලා" \
  -F "pronunciation_result=new_word" \
  -F 'session_words=["බල්ලා","පූසා","අලියා"]'
```

### Child Session — Correct → Transition to Next Word

```bash
curl -X POST http://localhost:8000/child-session \
  -H "X-API-Key: sinllama-default-key-change-me" \
  -F "audio=@child_says_ballaa.wav" \
  -F "child_name=කවිඳු" \
  -F "current_word=බල්ලා" \
  -F "pronunciation_result=correct" \
  -F "next_word=පූසා" \
  -F "words_completed=1" \
  -F 'session_words=["බල්ලා","පූසා","අලියා"]'
```

### Child Session — Retry (Wrong Pronunciation)

```bash
curl -X POST http://localhost:8000/child-session \
  -H "X-API-Key: sinllama-default-key-change-me" \
  -F "audio=@child_attempt.wav" \
  -F "child_name=කවිඳු" \
  -F "current_word=පූසා" \
  -F "pronunciation_result=retry" \
  -F "attempt_number=2" \
  -F 'session_words=["බල්ලා","පූසා","අලියා"]'
```

### Child Session — Session Complete

```bash
curl -X POST http://localhost:8000/child-session \
  -H "X-API-Key: sinllama-default-key-change-me" \
  -F "audio=@child_audio.wav" \
  -F "child_name=කවිඳු" \
  -F "is_session_complete=true" \
  -F "words_completed=3" \
  -F 'session_words=["බල්ලා","පූසා","අලියා"]'
```

---

## Examples (Python)

```python
import requests
import json

BASE_URL = "http://localhost:8000"
API_KEY = "sinllama-default-key-change-me"
HEADERS = {"X-API-Key": API_KEY}
JSON_HEADERS = {**HEADERS, "Content-Type": "application/json"}


# --- Health Check ---
print(requests.get(f"{BASE_URL}/health").json())


# --- Raw Generate ---
resp = requests.post(f"{BASE_URL}/generate", headers=JSON_HEADERS, json={
    "prompt": "ශ්‍රී ලංකාව",
    "max_new_tokens": 100,
})
print(resp.json())


# --- General Ask ---
resp = requests.post(f"{BASE_URL}/ask", headers=JSON_HEADERS, json={
    "message": "ශ්‍රී ලංකාවේ අගනුවර කුමක්ද?",
})
print(resp.json())


# --- Transcribe Audio ---
with open("recording.wav", "rb") as f:
    resp = requests.post(f"{BASE_URL}/transcribe", headers=HEADERS,
                         files={"audio": ("recording.wav", f, "audio/wav")})
print(resp.json())


# --- Child Chat (text input) ---
resp = requests.post(f"{BASE_URL}/child-chat", headers=JSON_HEADERS, json={
    "child_utterance": "බල්ලා",
    "child_name": "කවිඳු",
    "current_word": "බල්ලා",
    "pronunciation_result": "correct",
    "next_word": "පූසා",
    "session_words": ["බල්ලා", "පූසා", "අලියා"],
    "words_completed": 1,
})
print(resp.json())


# --- Child Session (combined — recommended for Raspberry Pi) ---
with open("child_audio.wav", "rb") as f:
    resp = requests.post(
        f"{BASE_URL}/child-session",
        headers=HEADERS,
        files={"audio": ("child_audio.wav", f, "audio/wav")},
        data={
            "child_name": "කවිඳු",
            "current_word": "බල්ලා",
            "pronunciation_result": "correct",
            "next_word": "පූසා",
            "words_completed": 1,
            "session_words": json.dumps(["බල්ලා", "පූසා", "අලියා"]),
        },
    )
print(resp.json())
# {
#   "transcribed_text": "බල්ලා",
#   "teacher_response": "වාව් කවිඳු! 'බල්ලා' හරියටම කිව්වා! ...",
#   "asr_time_ms": 850.2,
#   "chat_time_ms": 0.3,
#   "total_time_ms": 852.1
# }
```

---

## Examples (Raspberry Pi Client)

```python
"""Minimal Raspberry Pi client for /child-session endpoint."""
import requests
import json

SERVER_URL = "http://<server-ip>:8000/child-session"
HEADERS = {"X-API-Key": "your-api-key"}


def send_to_server(audio_path, child_name, current_word, pronunciation_result,
                   session_words, next_word="", attempt_number=1,
                   words_completed=0, is_session_complete=False):
    """Send audio + metadata → get transcription + teacher response."""
    with open(audio_path, "rb") as f:
        resp = requests.post(
            SERVER_URL,
            headers=HEADERS,
            files={"audio": (audio_path, f, "audio/wav")},
            data={
                "child_name": child_name,
                "current_word": current_word,
                "pronunciation_result": pronunciation_result,
                "attempt_number": str(attempt_number),
                "session_words": json.dumps(session_words),
                "next_word": next_word,
                "words_completed": str(words_completed),
                "is_session_complete": str(is_session_complete).lower(),
            },
            timeout=30,
        )

    result = resp.json()
    print(f"Child said: {result['transcribed_text']}")
    print(f"Teacher:    {result['teacher_response']}")
    print(f"Time: ASR={result['asr_time_ms']}ms + Chat={result['chat_time_ms']}ms = {result['total_time_ms']}ms")
    return result


# --- Example session flow ---
words = ["බල්ලා", "පූසා", "අලියා"]

# 1. Introduce first word
send_to_server("silence.wav", "කවිඳු", "බල්ලා", "new_word", words)

# 2. Child says "බල්ලා" correctly → transition to next
send_to_server("child_ballaa.wav", "කවිඳු", "බල්ලා", "correct", words,
               next_word="පූසා", words_completed=1)

# 3. Child tries "පූසා" but wrong → retry
send_to_server("child_attempt.wav", "කවිඳු", "පූසා", "retry", words,
               attempt_number=2)

# 4. Child says "පූසා" correctly
send_to_server("child_puusaa.wav", "කවිඳු", "පූසා", "correct", words,
               next_word="අලියා", words_completed=2)

# 5. Session complete
send_to_server("child_final.wav", "කවිඳු", "", "", words,
               words_completed=3, is_session_complete=True)
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `SINLLAMA_API_KEY` | API key for authentication | `sinllama-default-key-change-me` |
| `PORT` | Server port | `8000` |

---

## Required Libraries (Server)

```bash
# Core ML
pip install torch transformers peft bitsandbytes accelerate

# API Server
pip install fastapi uvicorn python-multipart

# ASR (Whisper)
pip install librosa soundfile

# System dependency for audio processing
apt-get install -y ffmpeg
```

---

## Interactive Docs

Once the server is running:

- **Swagger UI:** `http://<server-host>:8000/docs`
- **ReDoc:** `http://<server-host>:8000/redoc`

These are auto-generated by FastAPI with live "Try it out" functionality for all endpoints.
