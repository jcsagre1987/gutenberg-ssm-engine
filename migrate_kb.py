# ==========================================
# Gutenberg SSM KB to SQLite & Vector DB Migrator
# Copyright (C) 2026 Juan Carlo R. Sagre
# ==========================================

import json
import sqlite3
import os
from pathlib import Path

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("[MIGRATOR NOTE] ChromaDB not installed. Run 'pip install chromadb sentence-transformers' for vector search.")

DB_FILE = Path("gutenberg_kb.db")
KB_REF_FILE = Path("knowledgebasereference.json")
KB_LEARNED_FILE = Path("knowledgebaselearned.json")

def init_sqlite_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Enable WAL mode for safe concurrent reading/writing
    cursor.execute("PRAGMA journal_mode=WAL;")
    
    # Drop existing tables to ensure a completely fresh rebuild
    cursor.execute("DROP TABLE IF EXISTS knowledge_fts")
    cursor.execute("DROP TABLE IF EXISTS knowledge")

    # Create main knowledge table
    cursor.execute("""
        CREATE TABLE knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_key TEXT UNIQUE,
            response_text TEXT,
            category TEXT,
            source TEXT
        )
    """)
    
    # Create FTS5 Full-Text Search Virtual Table
    cursor.execute("""
        CREATE VIRTUAL TABLE knowledge_fts USING fts5(
            prompt_key, response_text, content='knowledge', content_rowid='id'
        )
    """)
    conn.commit()
    return conn

def migrate_kb_data():
    conn = init_sqlite_db()
    cursor = conn.cursor()

    total_inserted = 0

    # Load Reference KB
    if KB_REF_FILE.exists():
        ref_data = json.loads(KB_REF_FILE.read_text(encoding="utf-8"))
        for category, entries in ref_data.items():
            for key, value in entries.items():
                try:
                    cursor.execute(
                        "INSERT OR REPLACE INTO knowledge (prompt_key, response_text, category, source) VALUES (?, ?, ?, ?)",
                        (key.lower().strip(), value.strip(), category, "Reference KB")
                    )
                    total_inserted += 1
                except Exception as e:
                    print(f"[SQL ERROR]: {str(e)}")

    # Load Learned KB
    if KB_LEARNED_FILE.exists():
        learned_data = json.loads(KB_LEARNED_FILE.read_text(encoding="utf-8"))
        for category, entries in learned_data.items():
            for key, value in entries.items():
                try:
                    cursor.execute(
                        "INSERT OR REPLACE INTO knowledge (prompt_key, response_text, category, source) VALUES (?, ?, ?, ?)",
                        (key.lower().strip(), value.strip(), category, "Learned KB")
                    )
                    total_inserted += 1
                except Exception as e:
                    print(f"[SQL ERROR]: {str(e)}")

    # Populate FTS Table
    cursor.execute("INSERT INTO knowledge_fts(rowid, prompt_key, response_text) SELECT id, prompt_key, response_text FROM knowledge")
    conn.commit()
    conn.close()

    print(f"[SQLITE MIGRATION COMPLETE] Indexed {total_inserted} entries in {DB_FILE.name}.")

    # Migrate to Vector Store
    if CHROMA_AVAILABLE:
        print("[VECTOR MIGRATOR] Building ChromaDB semantic embeddings...")
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        emb_fn = embedding_functions.DefaultEmbeddingFunction()
        collection = chroma_client.get_or_create_collection(name="gutenberg_kb", embedding_function=emb_fn)

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT prompt_key, response_text, category, source FROM knowledge")
        rows = cursor.fetchall()

        if rows:
            documents = [r[1] for r in rows]
            metadatas = [{"prompt": r[0], "category": r[2], "source": r[3]} for r in rows]
            ids = [f"doc_{i}" for i in range(len(rows))]

            collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
            print(f"[VECTOR MIGRATION COMPLETE] Indexed {len(rows)} embeddings into Vector DB.")
        conn.close()

if __name__ == "__main__":
    migrate_kb_data()