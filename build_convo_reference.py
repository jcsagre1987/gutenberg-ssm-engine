# ==========================================
# Gutenberg SSM Web Conversational Skills Scraper
# Copyright (C) 2026 Juan Carlo R. Sagre
# ==========================================

import json
import re
import urllib.request
from pathlib import Path

CONVO_GUIDE_FILE = Path("conversationandsocialguidereference.json")

# Hugging Face Datasets API Endpoint for DailyDialog / PersonaChat
HF_DIALOG_URL = "https://datasets-server.huggingface.co/rows?dataset=li2017dailydialog%2Fdailydialog&config=default&split=train&offset=0&length=100"

def normalize_text(text: str) -> str:
    text = re.sub(r'[\'"?!\.,;:]', '', text.strip().lower())
    return re.sub(r'\s+', ' ', text)

def fetch_web_dialogue_samples() -> dict:
    web_greetings = {}
    web_acknowledgments = {}
    
    print("[CONVO SCRAPER] Fetching open human dialogue corpora from Hugging Face API...")
    try:
        req = urllib.request.Request(HF_DIALOG_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            rows = data.get("rows", [])
            
            for row in rows:
                dialogue = row.get("row", {}).get("dialog", [])
                # Extract 2-turn exchanges (User -> Assistant response)
                if len(dialogue) >= 2:
                    turn1 = dialogue[0].strip()
                    turn2 = dialogue[1].strip()
                    
                    # Clean up space artifacts around punctuation
                    turn1_clean = re.sub(r'\s+([?!\.,;:])', r'\1', turn1)
                    turn2_clean = re.sub(r'\s+([?!\.,;:])', r'\1', turn2)
                    
                    norm_key = normalize_text(turn1_clean)
                    
                    # Keep short, natural human turns (under 120 chars)
                    if 4 <= len(norm_key) <= 60 and len(turn2_clean) <= 120:
                        if any(kw in norm_key for kw in ["hi", "hello", "morning", "evening", "how are you", "day"]):
                            web_greetings[norm_key] = turn2_clean
                        else:
                            web_acknowledgments[norm_key] = turn2_clean
                            
        print(f"  + Scraped {len(web_greetings)} greeting patterns and {len(web_acknowledgments)} conversational turns.")
    except Exception as e:
        print(f"[SCRAPER NOTE] Offline fallback used for web dialogue fetching: {str(e)}")

    return {"greetings": web_greetings, "acknowledgments": web_acknowledgments}

def build_convo_guide():
    # Load existing file or default base persona
    if CONVO_GUIDE_FILE.exists():
        existing_data = json.loads(CONVO_GUIDE_FILE.read_text(encoding="utf-8"))
    else:
        existing_data = {
            "persona": {
                "name": "Gutenberg SSM",
                "nickname": "Neng",
                "user_title": "Boss",
                "tone": "Warm, grounded, professional, sharp, concise, and observant local AI assistant."
            },
            "greetings": {},
            "acknowledgments": {},
            "active_listening_steerers": [
                "Understood! What's our next step, Boss?",
                "Got it! What would you like to focus on next?",
                "Sounds good! What shall we tackle next, Boss?"
            ]
        }

    # Fetch fresh dialogue turns from web API
    scraped = fetch_web_dialogue_samples()

    # Merge scraped data without overwriting existing manual entries
    for key, val in scraped.get("greetings", {}).items():
        if key not in existing_data.setdefault("greetings", {}):
            existing_data["greetings"][key] = val

    for key, val in scraped.get("acknowledgments", {}).items():
        if key not in existing_data.setdefault("acknowledgments", {}):
            existing_data["acknowledgments"][key] = val

    CONVO_GUIDE_FILE.write_text(json.dumps(existing_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[CONVO BUILDER SUCCESS] Updated {CONVO_GUIDE_FILE.name} with {len(existing_data['greetings'])} greetings and {len(existing_data['acknowledgments'])} interaction turns.")

if __name__ == "__main__":
    build_convo_guide()