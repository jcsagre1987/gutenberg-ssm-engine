# ==========================================
# Gutenberg SSM Edge Engine (Hybrid DB RAG & JSON Social Engine)
# Copyright (C) 2026 Juan Carlo R. Sagre
# Licensed under the GNU General Public License v2.0 (GPLv2)
# ==========================================

import os
import time
import json
import asyncio
import html
import re
import random
import sqlite3
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from ddgs import DDGS

# Try loading Vector DB (ChromaDB)
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_CLIENT = chromadb.PersistentClient(path="./chroma_db")
    EMB_FN = embedding_functions.DefaultEmbeddingFunction()
    VECTOR_COLLECTION = CHROMA_CLIENT.get_or_create_collection(name="gutenberg_kb", embedding_function=EMB_FN)
    VECTOR_SEARCH_AVAILABLE = True
    print("[INIT] ChromaDB Vector Engine Initialized!")
except Exception as e:
    VECTOR_SEARCH_AVAILABLE = False
    print(f"[INIT NOTE] Vector search offline, using SQLite fallback. ({str(e)})")

# ==========================================
# 1. DATABASE & FILE PATHS
# ==========================================
DB_FILE = Path("gutenberg_kb.db")
CONVO_GUIDE_FILE = Path("conversationandsocialguidereference.json")
KB_REFERENCE_FILE = Path("knowledgebasereference.json")
CORRECTIONS_FILE = Path("gutenberg_corrections.json")

CASUAL_PATTERNS = [
    r'^(hi|hello|hey|yo|sup|good morning|good evening|good afternoon)\b',
    r'\bhow(\'s|\s+is|\s+are)\s+(you|your day|it going)\b',
    r'\b(what\'s\s+up|whats\s+up)\b',
    r'^(thanks|thank you|cool|nice|okay|ok|ready|proceed|doing great|lets start)\b'
]

CHALLENGE_PATTERNS = [
    r'\bare\s+you\s+(sure|certain)\b',
    r'\bis\s+that\s+(correct|right|true)\b',
    r'\bthat(\'s|\s+is)\s+(not|wrong|incorrect)\b',
    r'\byou\s+(made\s+a\s+mistake|are\s+wrong)\b'
]

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_key TEXT UNIQUE,
            response_text TEXT,
            category TEXT,
            source TEXT
        )
    """)
    
    # Auto-load facts from knowledgebasereference.json if available
    if KB_REFERENCE_FILE.exists():
        try:
            kb_data = json.loads(KB_REFERENCE_FILE.read_text(encoding="utf-8"))
            facts = kb_data.get("facts", {})
            for key, val in facts.items():
                cursor.execute(
                    "INSERT OR REPLACE INTO knowledge (prompt_key, response_text, category, source) VALUES (?, ?, ?, ?)",
                    (key.lower(), val, "reference", "JSON Knowledgebase")
                )
            conn.commit()
            print(f"[INIT] Loaded {len(facts)} facts from knowledgebasereference.json into SQLite!")
        except Exception as e:
            print(f"[INIT ERROR] Failed loading knowledgebasereference.json: {str(e)}")

    # Fallback default if empty
    cursor.execute("SELECT COUNT(*) FROM knowledge")
    if cursor.fetchone()[0] == 0:
        defaults = [
            ("socialism", "Socialism is an economic and political system based on public or collective ownership of the means of production.", "politics", "System Seed"),
            ("capitalism", "Capitalism is an economic system based on the private ownership of the means of production and their operation for profit.", "economics", "System Seed")
        ]
        cursor.executemany("INSERT OR IGNORE INTO knowledge (prompt_key, response_text, category, source) VALUES (?, ?, ?, ?)", defaults)
        conn.commit()
    conn.close()

def normalize_key(text: str) -> str:
    # Clean query by removing question prefixes so "What is meritocracy?" matches "meritocracy"
    cleaned = re.sub(r'(?i)\b(what is|what\'s|who is|who\'s|define|explain)\b', '', text)
    cleaned = re.sub(r'[\'"?!\.,;:]', '', cleaned.strip().lower())
    return re.sub(r'\s+', ' ', cleaned)

def is_casual_prompt(prompt: str) -> bool:
    clean_p = normalize_key(prompt)
    if not clean_p:
        return True
    if any(re.search(pat, clean_p) for pat in CASUAL_PATTERNS):
        return True
    if any(kw in clean_p for kw in ["where", "what", "who", "which", "part of asia", "countries", "list", "how many", "explain", "definition", "when", "compare"]):
        return False
    return False

def is_challenge_prompt(prompt: str) -> bool:
    clean_p = normalize_key(prompt)
    return any(re.search(pat, clean_p) for pat in CHALLENGE_PATTERNS)

# ==========================================
# 2. SOCIAL GUIDANCE & JSON PARSING
# ==========================================
def get_social_response(prompt: str) -> str:
    clean_p = normalize_key(prompt)
    if not CONVO_GUIDE_FILE.exists():
        return "Hello, Boss! How can I assist your workflow today?"

    try:
        data = json.loads(CONVO_GUIDE_FILE.read_text(encoding="utf-8"))
        greetings = data.get("greetings", {})
        acknowledgments = data.get("acknowledgments", {})
        steerers = data.get("active_listening_steerers", [])

        if clean_p in greetings:
            return greetings[clean_p]
        if clean_p in acknowledgments:
            return acknowledgments[clean_p]

        for key, val in {**greetings, **acknowledgments}.items():
            if key in clean_p or clean_p in key:
                return val

        if steerers:
            return random.choice(steerers)
            
    except Exception as e:
        print(f"[SOCIAL JSON ERROR]: {str(e)}")
        
    return "Hello, Boss! How can I assist your workflow today?"

# ==========================================
# 3. HYBRID RETRIEVAL & AUTO-PURGE CORRECTIONS
# ==========================================
def search_hybrid_kb(prompt: str) -> tuple[str, str, str]:
    clean_p = normalize_key(prompt)
    if not clean_p or is_casual_prompt(prompt):
        return "", "", ""

    if CORRECTIONS_FILE.exists():
        try:
            corrections = json.loads(CORRECTIONS_FILE.read_text(encoding="utf-8"))
            if clean_p in corrections.get("flagged_prompts", {}):
                return "", "", ""
        except Exception:
            pass

    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        # Direct key lookup
        cursor.execute("SELECT response_text, category, source FROM knowledge WHERE prompt_key = ?", (clean_p,))
        row = cursor.fetchone()
        
        # Fuzzy fallback lookup if direct key fails
        if not row:
            cursor.execute("SELECT response_text, category, source FROM knowledge WHERE ? LIKE '%' || prompt_key || '%'", (clean_p,))
            row = cursor.fetchone()

        conn.close()
        if row:
            return row[0], row[1], f"SQLite ({row[2]})"
    except Exception as e:
        print(f"[SQLITE SEARCH ERROR]: {str(e)}")

    return "", "", ""

def get_few_shot_examples() -> str:
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT prompt_key, response_text FROM knowledge LIMIT 2")
        rows = cursor.fetchall()
        conn.close()
        examples = ""
        for r in rows:
            examples += f"User: What is {r[0]}?\nAssistant: {r[1]}\n\n"
        return examples
    except Exception:
        return ""

def save_to_learned_db(prompt: str, response_text: str, category: str = "facts") -> bool:
    try:
        clean_key = normalize_key(prompt)
        clean_val = clean_factual_output(response_text.strip())
        
        if is_casual_prompt(clean_key) or len(clean_val.split()) < 3:
            return False

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO knowledge (prompt_key, response_text, category, source) VALUES (?, ?, ?, ?)",
            (clean_key, clean_val, category, "Learned KB")
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[KB SAVE ERROR]: {str(e)}")
        return False

def log_and_purge_correction(prompt: str, bad_response: str) -> bool:
    try:
        clean_key = normalize_key(prompt)
        if CORRECTIONS_FILE.exists():
            data = json.loads(CORRECTIONS_FILE.read_text(encoding="utf-8"))
        else:
            data = {"flagged_prompts": {}}

        data["flagged_prompts"][clean_key] = {
            "bad_response": bad_response,
            "flagged_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        CORRECTIONS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM knowledge WHERE prompt_key = ?", (clean_key,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[PURGE ERROR]: {str(e)}")
        return False

# ==========================================
# 4. STRICT OUTPUT GUARDRAILS & CONCISENESS
# ==========================================
def enforce_sentence_case(text: str) -> str:
    text = text.strip()
    if not text: return text
    for i, char in enumerate(text):
        if char.isalpha():
            return text[:i] + char.upper() + text[i+1:]
    return text

def clean_factual_output(text: str) -> str:
    text = re.sub(r'<\|.*?\|>', '', text)
    text = re.sub(r'[\u4e00-\u9fff]+', '', text)
    
    if "context:" in text.lower():
        text = text[:text.lower().find("context:")]
    
    sentences = re.findall(r'[^.!?]+[.!?]?', text)
    preamble_patterns = r'^(i will|here is|i am going|let me|sure|okay|the answer is)'
    noise_tokens = ['context:', 'http', 'www', 'thomas paine', 'plato', 'history', 'coined']
    
    valid_sentences = []
    for s in sentences:
        s_clean = s.strip()
        s_lower = s_clean.lower()
        
        if not s_clean:
            continue
        if re.search(preamble_patterns, s_lower):
            continue
        if any(token in s_lower for token in noise_tokens):
            continue
            
        valid_sentences.append(s_clean)
            
    if valid_sentences:
        result = valid_sentences[0]
    elif sentences:
        result = sentences[0].strip()
    else:
        result = text

    result = re.sub(r'\s{2,}', ' ', result).strip()
    
    if result and not result.endswith(('.', '!', '?')):
        result += '.'

    if not result or len(result.split()) < 3:
        return "I need a bit more specific context to answer that accurately."

    return enforce_sentence_case(result)

# ==========================================
# 5. PRECISION WEB RAG SEARCH PIPELINE
# ==========================================
def clean_search_query(raw_prompt: str) -> str:
    cleaned = re.sub(r'(?i)\b(i\'m asking|im asking|tell me|can you search up|can you search|search up|look up|find me|can you explain|then|again|what is|what\'s)\b', '', raw_prompt)
    cleaned = re.sub(r'[?!\. \'\"]+$', '', cleaned).strip()
    return cleaned if len(cleaned) > 2 else raw_prompt

def search_web(query: str) -> str:
    search_term = clean_search_query(query)
    keywords = [w.lower() for w in search_term.split() if len(w) > 3]
    noise_filters = ['thomas paine', 'plato', 'century', 'historical', 'born', 'died', 'pamphlet', 'monarchy']
    
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(search_term, max_results=3):
                snippet = r.get("body", "")
                if snippet:
                    sentences = re.findall(r'[^.!?]+[.!?]?', snippet)
                    if keywords:
                        relevant = [
                            s.strip() for s in sentences 
                            if any(kw in s.lower() for kw in keywords) 
                            and not any(nf in s.lower() for nf in noise_filters)
                        ]
                        if relevant:
                            return relevant[0]
    except Exception:
        pass
    return ""

# ==========================================
# 6. FASTAPI ENGINE INITIALIZATION
# ==========================================
app = FastAPI(title="Gutenberg Hybrid Edge Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*", 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ID = "bartowski/mamba-2.8b-hf-GGUF"
FILENAME = "mamba-2.8b-hf-Q6_K.gguf"

print(f"[INIT] Locating Model: {FILENAME}...")
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

print("[INIT] Loading Gutenberg into llama.cpp (n_ctx=4096)...")
llm = Llama(model_path=model_path, n_ctx=4096, n_threads=6, verbose=False)
print("[INIT] Gutenberg Hybrid Edge Engine Fully Operational!")

init_db()

@app.get("/")
def root():
    return {"status": "online", "engine": "Gutenberg SSM Edge Engine"}

@app.post("/api/kb/save")
async def save_to_kb_endpoint(request: Request):
    try:
        data = await request.json()
        success = save_to_learned_db(data.get("prompt"), data.get("response"), data.get("category", "facts"))
        if success:
            return {"status": "success", "message": "Saved to KB!"}
        raise HTTPException(status_code=400, detail="Ignored invalid content.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/kb/flag")
async def flag_error_endpoint(request: Request):
    try:
        data = await request.json()
        success = log_and_purge_correction(data.get("prompt"), data.get("response"))
        if success:
            return {"status": "success", "message": "Error logged and purged from DB!"}
        raise HTTPException(status_code=400, detail="Failed to log.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        stream_requested = bool(data.get("stream", False))
        is_challenge = is_challenge_prompt(latest_user_message)
        is_casual = is_casual_prompt(latest_user_message)

        # 1. Handle Casual Conversations via conversationandsocialguidereference.json
        if is_casual and not is_challenge:
            social_response = get_social_response(latest_user_message)
            if stream_requested:
                async def social_stream():
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': social_response}}], 'saveable': False})}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(social_stream(), media_type="text/event-stream")
            return {"response": social_response, "saveable": False, "choices": [{"message": {"role": "assistant", "content": social_response}}]}

        # 2. Check Knowledge Base first for factual queries
        if not is_challenge:
            kb_match, category, source_tag = search_hybrid_kb(latest_user_message)
            if kb_match:
                formatted_kb = clean_factual_output(kb_match)
                if stream_requested:
                    async def kb_stream():
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': formatted_kb}}], 'saveable': True})}\n\n"
                        yield "data: [DONE]\n\n"
                    return StreamingResponse(kb_stream(), media_type="text/event-stream")
                return {"response": formatted_kb, "saveable": True, "choices": [{"message": {"role": "assistant", "content": formatted_kb}}]}

        # 3. Build strict few-shot prompt for LLM generation
        stop_words = ["\nUser:", "\nInstructions:", "\nAssistant:", "User:", "Assistant:", "<|im_end|>"]
        few_shot_examples = get_few_shot_examples()

        search_context = search_web(latest_user_message)
        formatted_prompt = (
            "Instructions: State the exact definition directly in 1 concise sentence. Do not use preambles, history, or introductory phrases.\n\n"
            f"{few_shot_examples}"
        )
        if search_context:
            formatted_prompt += f"Context: {search_context}\n\n"
        formatted_prompt += f"User: {latest_user_message}\nAssistant:"
        temp_setting = 0.1

        # 4. Stream or return generated LLM response
        if stream_requested:
            async def event_generator():
                raw_chunk_buffer = ""
                try:
                    stream = llm.create_completion(
                        prompt=formatted_prompt, max_tokens=100, temperature=temp_setting, repeat_penalty=1.1, stop=stop_words, stream=True
                    )
                    for chunk in stream:
                        text = chunk["choices"][0]["text"]
                        if text:
                            raw_chunk_buffer += text
                            yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': text}}], 'saveable': True})}\n\n"
                            await asyncio.sleep(0.001)
                except Exception:
                    pass
                yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(event_generator(), media_type="text/event-stream")

        response = await asyncio.to_thread(
            llm.create_completion, prompt=formatted_prompt, max_tokens=100, temperature=temp_setting, repeat_penalty=1.1, stop=stop_words
        )
        response_text = clean_factual_output(html.unescape(response["choices"][0]["text"].strip()))

        save_to_learned_db(latest_user_message, response_text)

        return {"response": response_text, "saveable": True, "choices": [{"message": {"role": "assistant", "content": response_text}}]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=18080)