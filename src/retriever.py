import json
import re
import sqlite3
from typing import List, Dict, Any, Optional
from src.database import DB_PATH, cosine_similarity

STOP_WORDS = {
    "neler", "nedir", "nelerdir", "nasil", "nasildir", "hangi", "icin", "ile",
    "bir", "bu", "su", "ne", "mi", "mu", "ve", "veya", "gibi", "ama",
    "the", "and", "what", "are", "how",
}

def normalize_text(text: str) -> str:
    repl = str.maketrans({
        "İ": "i", "I": "i", "ı": "i",
        "Ş": "s", "ş": "s",
        "Ğ": "g", "ğ": "g",
        "Ü": "u", "ü": "u",
        "Ö": "o", "ö": "o",
        "Ç": "c", "ç": "c",
    })
    text = text.translate(repl).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def query_terms(query_text: str) -> List[str]:
    words = [w for w in normalize_text(query_text).split() if len(w) > 2]
    core = [w for w in words if w not in STOP_WORDS]
    return core or words


def lexical_score(query_text: str, content: str) -> float:
    terms = query_terms(query_text)
    if not terms:
        return 0.0
    blob = normalize_text(content)
    hits = 0.0
    for term in terms:
        if term in blob:
            hits += 1.0
        elif any(
            token.startswith(term[:4]) or term.startswith(token[:4])
            for token in blob.split()
            if len(token) > 3 and len(term) > 3
        ):
            hits += 0.7
    return hits / len(terms)


def title_phrase_boost(query_text: str, content: str) -> float:
    terms = query_terms(query_text)
    if len(terms) < 2:
        return 0.0
    blob = normalize_text(content)
    head = blob[:160]
    phrase = " ".join(terms[:2])
    if phrase in blob:
        return 0.5 if phrase in head else 0.28
    if all(t in head for t in terms[:2]):
        return 0.4
    return 0.0


def retrieve_smart_chunks(
    query_text: str,
    query_embedding: List[float],
    top_k: int = 5,
    filter_source: Optional[str] = None,
    db_path: str = DB_PATH
) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if filter_source and filter_source != "Tüm Belgeler":
        cursor.execute(
            "SELECT id, source_file, page_number, chunk_index, content, embedding FROM document_chunks WHERE source_file = ?",
            (filter_source,)
        )
    else:
        cursor.execute("SELECT id, source_file, page_number, chunk_index, content, embedding FROM document_chunks")

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return []

    query_words = set(normalize_text(query_text).split())
    results = []

    for row in rows:
        chunk_id, source_file, page_number, chunk_index, content, embedding_json = row
        chunk_vector = json.loads(embedding_json)

        cosine = 0.0
        if len(query_embedding) == len(chunk_vector):
            cosine = cosine_similarity(query_embedding, chunk_vector)

        lex = lexical_score(query_text, content)
        boost = title_phrase_boost(query_text, content)
        score = (0.2 * cosine) + (0.8 * lex) + boost

        file_norm = normalize_text(source_file.replace("_", " ").replace(".", " "))
        if query_words.intersection(set(file_norm.split())):
            score += 0.15

        qn = normalize_text(query_text)
        if page_number == 1 and ("nedir" in qn or "ne demek" in qn):
            score += 0.10

        results.append({
            "id": chunk_id,
            "source_file": source_file,
            "page_number": page_number,
            "chunk_index": chunk_index,
            "content": content,
            "similarity_score": min(score, 1.0),
            "is_relevant": lex >= 0.25 or boost >= 0.28 or cosine >= 0.45,
        })

    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return results[:top_k]
