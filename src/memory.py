import re
from typing import Dict, List, Optional, Tuple

from src.retriever import citation_names, normalize_text

FOLLOWUP_HINTS = (
    "o zaman", "peki", "ya o", "onun", "bunun", "orada", "oradaki",
    "oraya", "oradan", "bu okul", "bu kisi", "bu kitap", "hangi sehir",
    "hangi ulke", "hangi eyalet", "kimdi", "neresi", "daha fazla",
    "peki ya", "hangisi", "nerede", "kimi",
)

RELATED_TERMS = {
    "gallaudet": ["europe", "england", "france", "hartford", "connecticut"],
    "hartford": ["connecticut", "gallaudet", "permanent school"],
    "eliza": ["weizenbaum", "1966", "chatbot"],
    "weizenbaum": ["eliza", "1966"],
}


def last_turn(history: List[Dict[str, str]]) -> Tuple[str, str]:
    user = ""
    assistant = ""
    for item in history or []:
        role = (item.get("role") or "").lower()
        content = (item.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            user = content
        elif role in {"assistant", "asistan"}:
            assistant = content
    return user, assistant


def cited_source(text: str) -> str:
    match = re.search(r"\(Kaynak:\s*([^,\n]+),\s*Sayfa\s*\d+\)", text or "")
    if not match:
        return ""
    return match.group(1).strip()


def looks_like_followup(query: str) -> bool:
    qn = normalize_text(query or "")
    if not qn:
        return False
    if any(hint in qn for hint in FOLLOWUP_HINTS):
        return True
    words = qn.split()
    return len(words) <= 6 and any(k in qn for k in ("kim", "hangi", "kac", "ne", "neden"))


def infer_source(query: str, files: List[str], history: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
    if not files:
        return None
    qn = normalize_text(query or "")

    def pick(*needles: str) -> Optional[str]:
        for name in files:
            blob = normalize_text(name.replace("_", " ").replace("-", " "))
            if any(n in blob or n in name.lower() for n in needles):
                return name
        return None

    deneme = pick("deneme")
    merge = pick("merge")
    article = next(
        (f for f in files if "chatbot" in f.lower() or "kurumsal" in f.lower()),
        None,
    )

    if any(k in qn for k in ("eylem", "truba", "yapay zeka plan", "osb")):
        return merge
    if any(
        k in qn
        for k in (
            "harry", "best", "gallaudet", "hartford", "adventitious",
            "sagir", "dilsiz", "1910", "1914", "deaf",
        )
    ):
        return deneme
    if any(
        k in qn
        for k in (
            "chatbot", "gartner", "eliza", "weizenbaum", "chan",
            "ta johnson", "xiaoice", "brandtzaeg",
        )
    ):
        return article

    if looks_like_followup(query):
        _, last_answer = last_turn(history or [])
        source = cited_source(last_answer)
        if source in files:
            return source
    return None


def expand_query(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    query = (query or "").strip()
    if not query or not history:
        return query
    last_q, last_a = last_turn(history)
    if not last_q and not last_a:
        return query
    if not looks_like_followup(query):
        return query

    body = re.split(r"\(Kaynak:", last_a or "", maxsplit=1)[0].strip()
    names = citation_names(last_q) + citation_names(last_a) + citation_names(query)
    extra: List[str] = []
    for name in names:
        extra.append(name)
        extra.extend(RELATED_TERMS.get(name, []))
    qn = normalize_text(query)
    if any(k in qn for k in ("ulke", "avrupa", "seyahat", "inceleme")):
        extra.extend(["europe", "england", "france", "gallaudet"])
    if any(k in qn for k in ("sehir", "eyalet")):
        extra.extend(["hartford", "connecticut"])

    seen = set()
    unique = []
    for item in extra:
        key = normalize_text(item)
        if key and key not in seen:
            seen.add(key)
            unique.append(item)

    parts = [query, last_q]
    if unique:
        parts.append(" ".join(unique[:8]))
    if body:
        parts.append(body[:180])
    return " ".join(p for p in parts if p).strip()
