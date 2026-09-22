# ==========================================
# Gutenberg SSM Bulk Wikipedia Knowledge Ingester (Rate-Limited)
# Copyright (C) 2026 Juan Carlo R. Sagre
# ==========================================

import json
import re
import time
import urllib.request
import urllib.parse
from pathlib import Path

KB_REF_FILE = Path("knowledgebasereference.json")

# Curated High-Value Topics Across Major Domains
WIKI_SEED_TOPICS = [
    # World History & Events
    "Ancient Egypt", "Roman Empire", "Middle Ages", "Renaissance", "Industrial Revolution",
    "World War I", "World War II", "Cold War", "Space Race", "French Revolution",
    
    # Science & Physics
    "Quantum mechanics", "General relativity", "Thermodynamics", "Photosynthesis",
    "DNA", "Plate tectonics", "Speed of light", "Periodic table", "Evolution",
    
    # Mathematics & Logic
    "Calculus", "Algebra", "Pythagorean theorem", "Prime number", "Statistics",
    
    # Technology & AI
    "Artificial intelligence", "Machine learning", "Large language model",
    "Operating system", "Cloud computing", "Cybersecurity", "Blockchain",
    
    # Health & Life
    "Immune system", "Circadian rhythm", "Aerobic exercise", "Metabolism",
    
    # Economics & Politics
    "Capitalism", "Socialism", "Democracy", "Monarchy", "Inflation", "Microeconomics"
]

def clean_wiki_summary(text: str) -> str:
    """Strips citation brackets, coordinate noise, and cleans up output into 2 crisp sentences."""
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\([^)]*see\s+map[^)]*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+°?\s*\d*\'?\s*(?:north|south|east|west|N|S|E|W)?', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.findall(r'[^.!?]+[.!?]', text)
    return " ".join([s.strip() for s in sentences[:2]]) if sentences else text

def fetch_wiki_summary(topic: str, retries: int = 2) -> tuple[str, str]:
    """Fetches lead paragraph via MediaWiki API with exponential backoff on HTTP 429."""
    url = (
        f"https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts"
        f"&exintro=1&explaintext=1&generator=search&gsrsearch={urllib.parse.quote(topic)}&gsrlimit=1"
    )
    headers = {'User-Agent': 'GutenbergEdgeBot/2.0 (gutenberg.ssm@example.com)'}

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                pages = data.get("query", {}).get("pages", {})
                for p_id, p_info in pages.items():
                    title = p_info.get("title", "").strip().lower()
                    extract = p_info.get("extract", "").strip()
                    if extract:
                        cleaned = clean_wiki_summary(extract)
                        return title, cleaned
                return "", ""
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"  [RATE LIMITED 429] Waiting 5 seconds before retry ({attempt + 1}/{retries})...")
                time.sleep(5)
            else:
                print(f"  [HTTP ERROR {e.code}] Skipped '{topic}'")
                break
        except Exception as e:
            print(f"  [SKIPPED] Could not fetch '{topic}': {str(e)}")
            break

    return "", ""

def run_bulk_download():
    if KB_REF_FILE.exists():
        kb_data = json.loads(KB_REF_FILE.read_text(encoding="utf-8"))
    else:
        kb_data = {"facts": {}, "art_styles": {}}

    kb_data.setdefault("facts", {})
    added_count = 0

    print(f"[BULK DOWNLOADER] Processing {len(WIKI_SEED_TOPICS)} Wikipedia seed topics...")

    for topic in WIKI_SEED_TOPICS:
        norm_topic = topic.lower().strip()
        
        # Skip if fact already exists in JSON
        if norm_topic in kb_data["facts"]:
            continue

        title, summary = fetch_wiki_summary(topic)
        if summary:
            kb_data["facts"][norm_topic] = summary
            if title and title != norm_topic:
                kb_data["facts"][title] = summary
            added_count += 1
            print(f"  ✓ Indexed: '{norm_topic}'")

        # Essential 1.2-second rate limit delay per Wikipedia API rules
        time.sleep(1.2)

    KB_REF_FILE.write_text(json.dumps(kb_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[BULK DOWNLOADER SUCCESS] Added {added_count} new entries to {KB_REF_FILE.name}.")

if __name__ == "__main__":
    run_bulk_download()