import re
from typing import List, Dict, Any

from src.retriever import normalize_text


REJECT_HINTS = (
    "yeterli bilgi bulunmamaktadır",
    "geçmemektedir",
    "gecmemektedir",
    "birlikte geçmemektedir",
    "birlikte gecmemektedir",
)


def _tokens(text: str) -> set:
    return {t for t in normalize_text(text or "").split() if len(t) > 1}


def _numbers(text: str) -> set:
    compact = (text or "").replace(" ", "")
    found = re.findall(r"\d+(?:[.,]\d+)?", compact)
    out = set()
    for item in found:
        out.add(item.replace(",", "."))
        out.add(re.sub(r"[^\d]", "", item))
    return {n for n in out if n}


def verify_citations(
    response_text: str,
    retrieved_chunks: List[Dict[str, Any]],
    query_text: str = "",
) -> Dict[str, Any]:
    """
    Yanıtın kaynaklarla örtüşmesini ölçer.
    Türkçe cevap / İngilizce kaynakta sayı ve özel isim eşleşmesini güçlendirir.
    """
    empty = {
        "verified_citations": [],
        "details": [],
        "confidence_score": 0.0,
        "verification_status": "Kaynak Bulunamadı / Yetersiz",
        "kind": "empty",
    }
    if not retrieved_chunks or not response_text:
        return empty

    body = re.split(r"\(Kaynak:", response_text or "", maxsplit=1)[0]
    blob = normalize_text(body)
    if any(hint in blob or hint in (response_text or "").lower() for hint in REJECT_HINTS):
        status = (
            "Çapraz belge reddi"
            if "iki ayri belgedeki" in blob or "iki ayrı belgedeki" in (response_text or "").lower()
            else "Belgede yok"
        )
        return {
            "verified_citations": [],
            "details": [],
            "confidence_score": 0.0,
            "verification_status": status,
            "kind": "reject",
        }

    sentences = [s.strip() for s in re.split(r"[.!?]+", body) if len(s.strip()) > 10]
    query_names = {
        normalize_text(n.replace("-", " "))
        for n in re.findall(
            r"[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü]{2,}(?:\s+[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü]{2,})*",
            query_text or "",
        )
    }
    query_names |= {
        t
        for t in normalize_text(query_text or "").split()
        if len(t) >= 5
    }

    matched_sources = set()
    verified_details = []
    total_matches = 0

    for sentence in sentences:
        sent_tokens = _tokens(sentence)
        sent_nums = _numbers(sentence)
        if not sent_tokens and not sent_nums:
            continue
        best_match_chunk = None
        best_overlap_ratio = 0.0

        for chunk in retrieved_chunks:
            chunk_text = chunk.get("content") or ""
            chunk_tokens = _tokens(chunk_text)
            if not chunk_tokens and not _numbers(chunk_text):
                continue
            common = sent_tokens.intersection(chunk_tokens)
            overlap_ratio = len(common) / len(sent_tokens) if sent_tokens else 0.0
            chunk_nums = _numbers(chunk_text)
            if sent_nums and sent_nums.intersection(chunk_nums):
                overlap_ratio = max(overlap_ratio, 0.55)
            # Özel isim / odak kelime ortaklığı (TR↔EN çeviride kelime Jaccard düşük kalır)
            name_hits = sum(
                1
                for name in query_names
                if name
                and len(name) >= 4
                and (name in normalize_text(sentence) or name in normalize_text(chunk_text))
                and name in normalize_text(chunk_text)
                and name in normalize_text(sentence)
            )
            if name_hits:
                overlap_ratio = max(overlap_ratio, min(0.75, 0.4 + 0.15 * name_hits))
            if normalize_text(sentence) in normalize_text(chunk_text):
                overlap_ratio = max(overlap_ratio, 0.8)
            if overlap_ratio > best_overlap_ratio:
                best_overlap_ratio = overlap_ratio
                best_match_chunk = chunk

        if best_overlap_ratio >= 0.22 and best_match_chunk:
            total_matches += 1
            source_info = f"{best_match_chunk['source_file']} (Sayfa {best_match_chunk['page_number']})"
            matched_sources.add(source_info)
            verified_details.append({
                "sentence": sentence,
                "source": source_info,
                "confidence": round(best_overlap_ratio * 100, 1),
            })

    usable = [s for s in sentences if _tokens(s) or _numbers(s)]
    confidence_score = round((total_matches / len(usable)) * 100, 1) if usable else 0.0
    if confidence_score >= 60.0:
        status = "Yüksek Doğruluk"
        kind = "ok"
    else:
        status = "Düşük / Şüpheli Doğruluk"
        kind = "low"

    return {
        "verified_citations": sorted(list(matched_sources)),
        "details": verified_details,
        "confidence_score": confidence_score,
        "verification_status": status,
        "kind": kind,
    }
