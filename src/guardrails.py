from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.llm import NOT_FOUND
from src.memory import CLARIFY_PROMPT, is_bare_prompt
from src.retriever import (
    chunks_cover_focus,
    content_focus_tokens,
    lexical_score,
    normalize_text,
    unknown_proper_names,
)


def is_abstention_text(text: str) -> bool:
    blob = (text or "").lower()
    return any(
        m in blob
        for m in (
            "bulunmamaktadır",
            "geçmemektedir",
            "gecmemektedir",
            "yeterli bilgi",
            "net bir soru yaz",
            "birlikte geçmemektedir",
            "özel isim geçmemektedir",
        )
    )


def strip_source_line(text: str) -> str:
    return re.split(r"\n*\(Kaynak:", text or "", maxsplit=1)[0].strip()


def reject_payload(answer: str, *, status: str, engine_used: str) -> Dict[str, Any]:
    return {
        "answer": strip_source_line(answer) or NOT_FOUND,
        "not_found": True,
        "verification": {
            "verified_citations": [],
            "details": [],
            "confidence_score": 0.0,
            "verification_status": status,
            "kind": "reject",
            "query_relevance": 0.0,
            "faithfulness": 0.0,
        },
        "engine_used": engine_used,
        "chunks": {"foundry": [], "fallback": []},
    }


def preflight_query(query: str, engine_used: str) -> Optional[Dict[str, Any]]:
    if is_bare_prompt(query):
        return reject_payload(CLARIFY_PROMPT, status="Net soru gerekli", engine_used=engine_used)
    missing = unknown_proper_names(query)
    if missing:
        shown = " / ".join(missing[:3])
        return reject_payload(
            f'Yüklenen belgelerde "{shown}" adlı bir proje veya özel isim geçmemektedir.',
            status="Belgede yok",
            engine_used=engine_used,
        )
    return None


def select_context_chunks(
    query: str,
    answer_chunks: List[Dict[str, Any]],
    *,
    min_score: float,
) -> List[Dict[str, Any]]:
    relevant = [c for c in answer_chunks if c.get("is_relevant")]
    usable = relevant or [
        c for c in answer_chunks if float(c.get("similarity_score") or 0) >= min_score
    ]
    if usable and not chunks_cover_focus(query, usable):
        return []
    return usable


def query_chunk_relevance(query: str, chunks: List[Dict[str, Any]]) -> float:
    if not chunks:
        return 0.0
    return round(max(lexical_score(query, c.get("content") or "") for c in chunks), 4)


def last_was_abstention(history: Optional[List[Dict[str, str]]]) -> bool:
    if not history:
        return False
    for item in reversed(history):
        if (item.get("role") or "").lower() in {"assistant", "asistan"}:
            return is_abstention_text(item.get("content") or "")
    return False


def finalize_response(
    *,
    query: str,
    cleaned: str,
    usable: List[Dict[str, Any]],
    verification: Dict[str, Any],
    engine_used: str,
    public_chunk_fn,
) -> Dict[str, Any]:
    q_rel = query_chunk_relevance(query, usable)
    verification = dict(verification or {})
    faith = float(verification.get("confidence_score") or 0)
    verification["query_relevance"] = q_rel
    verification["faithfulness"] = faith
    kind = verification.get("kind") or "low"

    if is_abstention_text(cleaned) or kind in {"reject", "empty"}:
        cleaned = strip_source_line(cleaned) or NOT_FOUND
        status = verification.get("verification_status") or "Belgede yok"
        if "net bir soru" in cleaned.lower():
            status = "Net soru gerekli"
        elif "özel isim" in cleaned.lower():
            status = "Belgede yok"
        elif "geçmemektedir" not in cleaned.lower() and "özel isim" not in cleaned.lower():
            if is_abstention_text(cleaned):
                cleaned = NOT_FOUND
                status = "Bilgi Belgelerde Bulunamadı"
        return reject_payload(cleaned, status=status, engine_used=engine_used)

    combined = round(0.6 * faith + 0.4 * min(100.0, q_rel * 200), 1)
    combined = max(0.0, min(100.0, combined))
    verification["confidence_score"] = combined
    if combined >= 60:
        verification["kind"] = "ok"
        verification["verification_status"] = f"%{int(round(combined))} Doğrulanmış"
    else:
        verification["kind"] = "low"
        verification["verification_status"] = f"%{int(round(combined))} Düşük örtüşme"

    return {
        "answer": cleaned,
        "not_found": False,
        "verification": verification,
        "engine_used": engine_used,
        "chunks": {
            "foundry": [],
            "fallback": [public_chunk_fn(c) for c in usable],
        },
    }
