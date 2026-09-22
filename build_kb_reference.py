# ==========================================
# Gutenberg SSM Multi-Domain KB Builder
# Copyright (C) 2026 Juan Carlo R. Sagre
# ==========================================

import json
import re
import urllib.request
import urllib.parse
from pathlib import Path

KB_REF_FILE = Path("knowledgebasereference.json")

AUTO_FETCH_TOPICS = [
    "Machine learning", "Transformer (machine learning architecture)",
    "Neuroscience", "Immune system", "Thermodynamics",
    "Macroeconomics", "International Monetary Fund", "SpaceX Starship"
]

def clean_wiki_summary(text: str) -> str:
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\([^)]*see\s+map[^)]*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+°?\s*\d*\'?\s*(?:north|south|east|west|N|S|E|W)?', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.findall(r'[^.!?]+[.!?]', text)
    return " ".join([s.strip() for s in sentences[:2]]) if sentences else text

def fetch_wiki_topic(topic: str) -> str:
    try:
        url = (
            f"https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts"
            f"&exintro=1&explaintext=1&generator=search&gsrsearch={urllib.parse.quote(topic)}&gsrlimit=1"
        )
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            pages = data.get("query", {}).get("pages", {})
            for p_id, p_info in pages.items():
                extract = p_info.get("extract", "").strip()
                if extract:
                    return clean_wiki_summary(extract)
    except Exception as e:
        print(f"[FETCH WARNING] Skip '{topic}': {str(e)}")
    return ""

def build_kb():
    if not KB_REF_FILE.exists():
        print("[BUILDER] KB Reference file missing, please ensure knowledgebasereference.json exists.")
        return

    data = json.loads(KB_REF_FILE.read_text(encoding="utf-8"))
    data.setdefault("facts", {})
    data.setdefault("art_styles", {})

    print("[BUILDER] Fetching external topics from Wikipedia to expand knowledge base...")
    for topic in AUTO_FETCH_TOPICS:
        key = topic.lower().strip()
        if key not in data["facts"]:
            summary = fetch_wiki_topic(topic)
            if summary:
                data["facts"][key] = summary
                print(f"  + Added Topic: '{key}'")

    KB_REF_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[BUILDER SUCCESS] Consolidated {len(data['facts'])} facts and {len(data['art_styles'])} art styles into {KB_REF_FILE.name}")

if __name__ == "__main__":
    build_kb()