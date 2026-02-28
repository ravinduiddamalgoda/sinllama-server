# SinLlama API Documentation

> **Version:** 1.1.0  
> **Description:** Sinhala LLM inference API powered by Meta-Llama-3-8B with SinLlama LoRA adapter  
> **Base URL:** `http://<server-host>:8000`

---

## Table of Contents

- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [GET /health](#1-get-health)
  - [POST /generate](#2-post-generate)
  - [POST /ask](#3-post-ask)
  - [POST /chat](#4-post-chat)
- [Error Responses](#error-responses)
- [Examples (cURL)](#examples-curl)
- [Examples (Python)](#examples-python)
- [Examples (JavaScript / Fetch)](#examples-javascript--fetch)
- [Environment Variables](#environment-variables)

---

## Authentication

All endpoints except `/health` require an API key passed via the `X-API-Key` HTTP header.

| Header      | Value                   |
| ----------- | ----------------------- |
| `X-API-Key` | Your API key            |

Set the key on the server via the `SINLLAMA_API_KEY` environment variable.  
Default (for development only): `sinllama-default-key-change-me`

---

## Endpoints

### 1. `GET /health`

Health check and GPU status. **No authentication required.**

#### Request

```
GET /health
```

No query parameters or body required.

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

| Field          | Type    | Description                                  |
| -------------- | ------- | -------------------------------------------- |
| `status`       | string  | `"healthy"` if model loaded, else `"loading"` |
| `model_loaded` | boolean | Whether the model is ready for inference     |
| `gpu_count`    | integer | Number of available GPUs                     |
| `input_device` | string  | CUDA device used for input tensors           |
| `gpus`         | array   | Per-GPU VRAM usage details                   |

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

| Field                | Type   | Required | Default | Range     | Description                                |
| -------------------- | ------ | -------- | ------- | --------- | ------------------------------------------ |
| `prompt`             | string | **Yes**  | —       | —         | Input text prompt for the model            |
| `max_new_tokens`     | int    | No       | 256     | 1–1024    | Maximum number of tokens to generate       |
| `temperature`        | float  | No       | 0.7     | 0.1–2.0   | Sampling temperature (higher = more random)|
| `top_p`              | float  | No       | 0.9     | 0.0–1.0   | Nucleus sampling probability threshold     |
| `top_k`              | int    | No       | 50      | 0–200     | Top-K sampling (0 = disabled)              |
| `repetition_penalty` | float  | No       | 1.1     | 1.0–2.0   | Penalty for repeating tokens               |

#### Response `200 OK`

```json
{
  "generated_text": "ශ්‍රී ලංකාව ඉන්දියානු සාගරයේ පිහිටි දිවයිනකි.",
  "tokens_generated": 42,
  "inference_time_ms": 1250.3
}
```

| Field               | Type   | Description                          |
| ------------------- | ------ | ------------------------------------ |
| `generated_text`    | string | Model-generated continuation text    |
| `tokens_generated`  | int    | Number of new tokens generated       |
| `inference_time_ms` | float  | Total inference time in milliseconds |

---

### 3. `POST /ask`

General-purpose LLM chat endpoint. Send any message with an optional system prompt and multi-turn conversation history. **Authentication required.**

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
  "system_prompt": "ඔබ සිංහල භාෂාවෙන් උත්තර දෙන බුද්ධිමත් සහකාරයෙකි. පැහැදිලිව සහ නිවැරදිව උත්තර දෙන්න.",
  "conversation_history": [
    { "role": "user", "content": "හෙලෝ" },
    { "role": "assistant", "content": "ආයුබෝවන්! මට ඔබට උදව් කරන්න පුළුවන්." }
  ],
  "max_new_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1
}
```

| Field                  | Type         | Required | Default                         | Range   | Description                                                                 |
| ---------------------- | ------------ | -------- | ------------------------------- | ------- | --------------------------------------------------------------------------- |
| `message`              | string       | **Yes**  | —                               | —       | The user's current message / question                                       |
| `system_prompt`        | string       | No       | Sinhala default (see below)     | —       | System prompt to guide the model's behaviour                                |
| `conversation_history` | list[object] | No       | `[]`                            | last 10 | Previous turns as `[{"role": "user"|"assistant", "content": "..."}]`        |
| `max_new_tokens`       | int          | No       | 512                             | 1–1024  | Maximum number of tokens to generate                                        |
| `temperature`          | float        | No       | 0.7                             | 0.1–2.0 | Sampling temperature                                                        |
| `top_p`                | float        | No       | 0.9                             | 0.0–1.0 | Nucleus sampling threshold                                                  |
| `top_k`                | int          | No       | 50                              | 0–200   | Top-K sampling                                                              |
| `repetition_penalty`   | float        | No       | 1.1                             | 1.0–2.0 | Repetition penalty                                                          |

**Default system prompt:**
```
ඔබ සිංහල භාෂාවෙන් උත්තර දෙන බුද්ධිමත් සහකාරයෙකි. පැහැදිලිව සහ නිවැරදිව උත්තර දෙන්න.
```
*(Translation: "You are an intelligent assistant that answers in Sinhala. Answer clearly and accurately.")*

#### Conversation History Format

Each entry in `conversation_history` must have:

| Field     | Type   | Values                    | Description               |
| --------- | ------ | ------------------------- | ------------------------- |
| `role`    | string | `"user"` or `"assistant"` | Who said this turn        |
| `content` | string | —                         | The text of that turn     |

Only the last **10 turns** are used to keep the context window manageable.

#### Response `200 OK`

```json
{
  "reply": "ශ්‍රී ලංකාවේ අගනුවර ශ්‍රී ජයවර්ධනපුර කෝට්ටේ වේ.",
  "tokens_generated": 35,
  "inference_time_ms": 980.5
}
```

| Field               | Type   | Description                          |
| ------------------- | ------ | ------------------------------------ |
| `reply`             | string | Model's response text                |
| `tokens_generated`  | int    | Number of new tokens generated       |
| `inference_time_ms` | float  | Total inference time in milliseconds |

---

### 4. `POST /chat`

Child conversation endpoint designed for the IoT toy. Formats a Sinhala prompt optimized for interactions with young children, with optional phoneme targeting for speech therapy. **Authentication required.**

#### Request

```
POST /chat
Content-Type: application/json
X-API-Key: <your-key>
```

#### Request Body

```json
{
  "child_utterance": "මට කතාවක් කියන්න",
  "conversation_history": [
    "දරුවා: ආයුබෝවන්",
    "සහකාරයා: ආයුබෝවන් පුතේ! කොහොමද?"
  ],
  "child_name": "කවිඳු",
  "target_phoneme": "ක"
}
```

| Field                  | Type       | Required | Default  | Description                                        |
| ---------------------- | ---------- | -------- | -------- | -------------------------------------------------- |
| `child_utterance`      | string     | **Yes**  | —        | What the child said (Sinhala text)                  |
| `conversation_history` | list[str]  | No       | `[]`     | Previous turns as plain strings (last 6 kept)       |
| `child_name`           | string     | No       | `"ළමයා"` | The child's name for personalized responses         |
| `target_phoneme`       | string     | No       | `""`     | Target phoneme for speech therapy practice          |

**Conversation history format:** Each entry is a plain string like `"දරුවා: ආයුබෝවන්"` or `"සහකාරයා: ආයුබෝවන් පුතේ!"`.

**Internal prompt template:**
```
ඔබ කුඩා දරුවන් සමග සිංහලෙන් කතා කරන මිත්‍රශීලී සහකාරයෙකි.
දරුවාගේ නම: {child_name}
ඉලක්ක ශබ්දය: {target_phoneme}       ← only if target_phoneme is provided
සරල, කෙටි වාක්‍ය භාවිතා කරන්න. දරුවාට තේරෙන භාෂාවෙන් කතා කරන්න.

පෙර සංවාදය:
{conversation_history}

දරුවා: {child_utterance}
සහකාරයා:
```

#### Response `200 OK`

```json
{
  "response": "හරි පුතේ! කවුරුහරි කොල්ලෙක් ගැන කතාවක් කියන්නම්.",
  "inference_time_ms": 1100.2
}
```

| Field               | Type   | Description                                           |
| ------------------- | ------ | ----------------------------------------------------- |
| `response`          | string | Model's child-friendly Sinhala response               |
| `inference_time_ms` | float  | Total inference time in milliseconds                  |

---

## Error Responses

All protected endpoints may return the following errors:

| Status Code | Condition                      | Response Body                                                                 |
| ----------- | ------------------------------ | ----------------------------------------------------------------------------- |
| **403**     | Missing or invalid API key     | `{"detail": "Invalid or missing API key. Pass X-API-Key header."}`            |
| **503**     | Model still loading            | `{"detail": "Model not loaded yet."}`                                         |
| **507**     | GPU out of memory              | `{"detail": "GPU out of memory. Try shorter prompt or fewer tokens."}`        |
| **500**     | Internal server error          | `{"detail": "<error message>"}`                                               |

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

### General Chat (/ask) — Simple

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sinllama-default-key-change-me" \
  -d '{
    "message": "ශ්‍රී ලංකාවේ අගනුවර කුමක්ද?"
  }'
```

### General Chat (/ask) — With History & Custom System Prompt

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sinllama-default-key-change-me" \
  -d '{
    "message": "ඒක ගැන තව කියන්න",
    "system_prompt": "ඔබ ශ්‍රී ලංකාව ගැන විශේෂඥයෙකි. සවිස්තරව පිළිතුරු දෙන්න.",
    "conversation_history": [
      {"role": "user", "content": "ශ්‍රී ලංකාවේ අගනුවර කුමක්ද?"},
      {"role": "assistant", "content": "ශ්‍රී ජයවර්ධනපුර කෝට්ටේ වේ."}
    ],
    "max_new_tokens": 256
  }'
```

### Child Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sinllama-default-key-change-me" \
  -d '{
    "child_utterance": "මට කතාවක් කියන්න",
    "child_name": "කවිඳු",
    "target_phoneme": "ක"
  }'
```

### Child Chat — With History

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sinllama-default-key-change-me" \
  -d '{
    "child_utterance": "තව කියන්න",
    "child_name": "කවිඳු",
    "conversation_history": [
      "දරුවා: මට කතාවක් කියන්න",
      "සහකාරයා: හරි පුතේ! එක කොල්ලෙක් හිටියා...",
      "දරුවා: ඊට පස්සේ මොකද උනේ?"
    ]
  }'
```

---

## Examples (Python)

```python
import requests

BASE_URL = "http://localhost:8000"
HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": "sinllama-default-key-change-me",
}


# --- Health Check ---
resp = requests.get(f"{BASE_URL}/health")
print(resp.json())


# --- Raw Generate ---
resp = requests.post(
    f"{BASE_URL}/generate",
    headers=HEADERS,
    json={
        "prompt": "ශ්‍රී ලංකාව",
        "max_new_tokens": 100,
    },
)
print(resp.json())


# --- General Ask (simple) ---
resp = requests.post(
    f"{BASE_URL}/ask",
    headers=HEADERS,
    json={
        "message": "ශ්‍රී ලංකාවේ අගනුවර කුමක්ද?",
    },
)
print(resp.json())


# --- General Ask (with history) ---
resp = requests.post(
    f"{BASE_URL}/ask",
    headers=HEADERS,
    json={
        "message": "ඒක ගැන තව කියන්න",
        "conversation_history": [
            {"role": "user", "content": "ශ්‍රී ලංකාවේ අගනුවර කුමක්ද?"},
            {"role": "assistant", "content": "ශ්‍රී ජයවර්ධනපුර කෝට්ටේ වේ."},
        ],
        "max_new_tokens": 256,
    },
)
print(resp.json())


# --- Child Chat ---
resp = requests.post(
    f"{BASE_URL}/chat",
    headers=HEADERS,
    json={
        "child_utterance": "මට කතාවක් කියන්න",
        "child_name": "කවිඳු",
        "target_phoneme": "ක",
    },
)
print(resp.json())
```

---

## Examples (JavaScript / Fetch)

```javascript
const BASE_URL = "http://localhost:8000";
const API_KEY = "sinllama-default-key-change-me";

const headers = {
  "Content-Type": "application/json",
  "X-API-Key": API_KEY,
};

// --- Health Check ---
const healthRes = await fetch(`${BASE_URL}/health`);
console.log(await healthRes.json());

// --- Raw Generate ---
const genRes = await fetch(`${BASE_URL}/generate`, {
  method: "POST",
  headers,
  body: JSON.stringify({
    prompt: "ශ්‍රී ලංකාව",
    max_new_tokens: 100,
  }),
});
console.log(await genRes.json());

// --- General Ask (simple) ---
const askRes = await fetch(`${BASE_URL}/ask`, {
  method: "POST",
  headers,
  body: JSON.stringify({
    message: "ශ්‍රී ලංකාවේ අගනුවර කුමක්ද?",
  }),
});
console.log(await askRes.json());

// --- General Ask (with conversation history) ---
const askHistoryRes = await fetch(`${BASE_URL}/ask`, {
  method: "POST",
  headers,
  body: JSON.stringify({
    message: "ඒක ගැන තව කියන්න",
    system_prompt:
      "ඔබ ශ්‍රී ලංකාව ගැන විශේෂඥයෙකි. සවිස්තරව පිළිතුරු දෙන්න.",
    conversation_history: [
      { role: "user", content: "ශ්‍රී ලංකාවේ අගනුවර කුමක්ද?" },
      { role: "assistant", content: "ශ්‍රී ජයවර්ධනපුර කෝට්ටේ වේ." },
    ],
    max_new_tokens: 256,
  }),
});
console.log(await askHistoryRes.json());

// --- Child Chat ---
const chatRes = await fetch(`${BASE_URL}/chat`, {
  method: "POST",
  headers,
  body: JSON.stringify({
    child_utterance: "මට කතාවක් කියන්න",
    child_name: "කවිඳු",
    target_phoneme: "ක",
  }),
});
console.log(await chatRes.json());
```

---

## Environment Variables

| Variable            | Description                              | Default                             |
| ------------------- | ---------------------------------------- | ----------------------------------- |
| `SINLLAMA_API_KEY`  | API key for authenticating requests      | `sinllama-default-key-change-me`    |
| `PORT`              | Server port                              | `8000`                              |

---

## Interactive Docs

Once the server is running, visit:

- **Swagger UI:** `http://<server-host>:8000/docs`
- **ReDoc:** `http://<server-host>:8000/redoc`

These are automatically generated by FastAPI and reflect all endpoints with live "Try it out" functionality.
