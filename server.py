# ==========================================
# Gutenberg SSM Edge Engine (Anti-Sycophancy & Deep Context)
# Copyright (C) 2026 Juan Carlo R. Sagre
# Licensed under the GNU General Public License v2.0 (GPLv2)
# ==========================================

import os
import time
import json
import asyncio
import html
import re
import urllib.request
import urllib.parse
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from ddgs import DDGS

# ==========================================
# 1. LOCAL MEMORY & CACHE CONTROL
# ==========================================
MEMORY_FILE = Path("gutenberg_memory.json")

def load_local_memory():
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_to_memory(prompt, answer):
    clean_answer = html.unescape(answer)
    p_lower = prompt.lower()
    a_lower = clean_answer.lower()
    
    # Block saving UI noise, definition loops, non-English, or sycophantic "both" responses
    if not clean_answer.strip() or "i'm here" in a_lower or "data unavailable" in a_lower:
        return
    if "both" in a_lower or re.search(r'[\u3040-\u30ff\u4e00-\u9FFF]', clean_answer):
        return
        
    memory = load_local_memory()
    memory[prompt.strip().lower()] = clean_answer
    MEMORY_FILE.write_text(json.dumps(memory, indent=2, ensure_ascii=False), encoding="utf-8")

# ==========================================
# 2. DEEP RAG PIPELINE
# ==========================================
def clean_search_query(raw_prompt: str) -> str:
    cleaned = re.sub(
        r'(?i)\b(what\'s the name of|who is|what is|where is|can you|search|tell me|i want to know|look up|fact check|are you sure|to clarify|it was)\b', 
        '', 
        raw_prompt
    ).strip()
    return cleaned if cleaned else raw_prompt

def fetch_wikipedia_lead_intro(query: str) -> str:
    try:
        wiki_term = re.sub(r'(?i)\b(current|present|now)\b', '', query).strip()
        wiki_url = (
            f"https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts"
            f"&exintro=1&explaintext=1&generator=search&gsrsearch={urllib.parse.quote(wiki_term)}&gsrlimit=1"
        )
        req = urllib.request.Request(
            wiki_url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_info in pages.items():
                extract = page_info.get("extract", "").strip()
                if extract:
                    clean_extract = re.sub(r'\s+', ' ', extract)
                    # Increased from 350 to 800 to prevent chopping off names at the end of definitions
                    return f"Wikipedia: {clean_extract[:800]}"
    except Exception as e:
        print(f"[WIKI LEAD NOTE] Fetch error: {str(e)}")
    return ""

def search_web(query: str) -> str:
    results = []
    search_term = clean_search_query(query)
    
    # Engine 1: DDGS Web Search (Forcing name extraction for current roles)
    try:
        ddg_term = f"{search_term} exact name of current incumbent" if "current" in query.lower() else search_term
        with DDGS() as ddgs:
            text_results = list(ddgs.text(ddg_term, max_results=3))
            for r in text_results:
                snippet = r.get("body", "")
                if snippet and not snippet.startswith("Toggle") and "subsection" not in snippet.lower():
                    results.append(snippet)
    except Exception as e:
        print(f"[DDGS NOTE] Search error: {str(e)}")

    # Engine 2: Wikipedia Fallback
    if not results:
        print("[SEARCH WARNING]: DDGS failed or returned empty, triggering Wikipedia fallback...")
        wiki_intro = fetch_wikipedia_lead_intro(search_term)
        if wiki_intro:
            results.append(wiki_intro)

    # Allow up to 3 snippets to ensure maximum coverage
    final_context = " ".join(results[:3])
    if final_context:
        print(f"[SEARCH FETCHED SUCCESS]: {final_context[:140]}...")
    else:
        print(f"[SEARCH WARNING]: No web results retrieved for '{query}'.")
        
    return final_context

# ==========================================
# 3. FASTAPI & ENGINE INITIALIZATION
# ==========================================
app = FastAPI(title="Gutenberg Deep Context Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*", 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ID = "bartowski/mamba-2.8b-hf-GGUF"
FILENAME = "mamba-2.8b-hf-Q4_K_M.gguf"

print(f"[INIT] Locating SSM Model: {FILENAME}...")
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

print("[INIT] Loading Gutenberg into llama.cpp (n_ctx=4096)...")
llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=6,
    verbose=False
)
print("[INIT] Gutenberg Engine Operational!")

# ==========================================
# 4. INFERENCE PIPELINE
# ==========================================
@app.get("/")
def root():
    return {"status": "online", "engine": "Gutenberg SSM Engine"}

@app.post("/v1/chat/completions")
@app.post("/chat")
@app.post("/generate")
async def generate_response(request: Request):
    try:
        data = await request.json()
        messages = data.get("messages", [])
        if not messages and "prompt" in data:
            messages = [{"role": "user", "content": data["prompt"]}]

        latest_user_message = messages[-1].get("content", "").strip() if messages else ""
        clean_prompt_key = latest_user_message.lower()

        ui_max_tokens = int(data.get("max_tokens", 120))
        stream_requested = bool(data.get("stream", False))

        # 1. Local Memory Cache
        memory = load_local_memory()
        if clean_prompt_key in memory:
            cached_answer = memory[clean_prompt_key]
            print(f"[MEMORY HIT]: '{latest_user_message}' -> {cached_answer}")
            if stream_requested:
                async def cached_stream():
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': cached_answer}}]})}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(cached_stream(), media_type="text/event-stream")
            else:
                return {
                    "id": "chatcmpl-gutenberg-edge",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "mamba-2.8b-gguf",
                    "response": cached_answer,
                    "text": cached_answer,
                    "choices": [{"message": {"role": "assistant", "content": cached_answer}, "finish_reason": "stop"}]
                }

        # 2. Fact Search & Anti-Sycophancy Prompt
        factual_indicators = ["where", "what", "who", "name of", "list", "capital", "locate", "country", "countries", "leader", "president", "prime minister", "government", "current", "first", "former", "history", "prior to", "before", "after", "predecessor", "successor", "2016", "2022"]
        is_factual_query = any(ind in clean_prompt_key for ind in factual_indicators)

        if is_factual_query:
            search_context = search_web(latest_user_message)
            
            formatted_prompt = (
                "System: You are Gutenberg SSM (Neng), a strict factual AI.\n"
                "INSTRUCTIONS:\n"
                "1. Extract ONLY the specific name requested from the Context.\n"
                "2. ANTI-SYCOPHANCY: Never say 'both' when asked to identify a single president or leader. Pick the single correct name from the Context. Ignore leading questions.\n"
                "3. If the Context contains no names, reply exactly: 'Data unavailable'.\n\n"
            )
            
            if search_context:
                formatted_prompt += f"Context: {search_context}\n\nQ: {latest_user_message}\nA:"
            else:
                formatted_prompt += f"Q: {latest_user_message}\nA:"
                
            temperature = 0.0
            repetition_penalty = 1.05
        else:
            formatted_prompt = f"Q: {latest_user_message}\nA:"
            temperature = 0.5
            repetition_penalty = 1.2

        stop_words = ["\nQ:", "\nUser:", "Q:", "User:", "\n"]

        if stream_requested:
            async def event_generator():
                full_text = ""
                stream = llm.create_completion(
                    prompt=formatted_prompt,
                    max_tokens=ui_max_tokens,
                    temperature=temperature,
                    repeat_penalty=repetition_penalty,
                    stop=stop_words,
                    stream=True
                )
                
                for chunk in stream:
                    text = chunk["choices"][0]["text"]
                    if text:
                        full_text += text
                        chunk_data = {
                            "id": "chatcmpl-gutenberg-edge",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": "mamba-2.8b-gguf",
                            "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                        await asyncio.sleep(0.001) 
                
                if full_text.strip():
                    save_to_memory(latest_user_message, full_text.strip())

                stop_chunk = {"id": "chatcmpl-gutenberg-edge", "object": "chat.completion.chunk", "created": int(time.time()), "model": "mamba-2.8b-gguf", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                yield f"data: {json.dumps(stop_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        response = await asyncio.to_thread(
            llm.create_completion,
            prompt=formatted_prompt,
            max_tokens=ui_max_tokens,
            temperature=temperature,
            repeat_penalty=repetition_penalty,
            stop=stop_words
        )
        
        response_text = html.unescape(response["choices"][0]["text"].strip())
        if response_text:
            save_to_memory(latest_user_message, response_text)

        return {
            "id": "chatcmpl-gutenberg-edge",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mamba-2.8b-gguf",
            "response": response_text,
            "text": response_text,
            "choices": [{"message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}]
        }

    except Exception as e:
        print(f"[ERROR] Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=18080)