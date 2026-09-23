"""Regresyon: görülen hata sınıfları (uydurma eşleşme, dolgu, takip sızıntısı)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.guardrails import finalize_response, preflight_query, select_context_chunks
from src.memory import expand_query, is_bare_prompt, looks_like_followup
from src.retriever import chunks_cover_focus, content_focus_tokens, rerank_chunks
from src.verifier import verify_citations


def run_tests():
    print("--- Guardrail regression ---")

    assert is_bare_prompt("pek")
    assert is_bare_prompt("peki")
    assert not is_bare_prompt("Gallaudet kimdir?")
    print("[OK] bare prompt")

    assert "fenerbahce" in content_focus_tokens("peki fenerbahce")
    assert "galatasaray" in content_focus_tokens("galatasaray nasil bir takim")
    chatbot = [{"content": "Chatbotlar isletmeler icin destek saglar.", "is_relevant": True, "similarity_score": 0.9}]
    assert chunks_cover_focus("peki fenerbahce", chatbot) is False
    assert select_context_chunks("peki fenerbahce", chatbot, min_score=0.2) == []
    print("[OK] focus grounding blocks OOD names")

    early = preflight_query("pek", "fallback")
    assert early and early["not_found"] and "Net soru" in early["verification"]["verification_status"]
    early2 = preflight_query("fenerbahce nedir", "fallback")
    assert early2 and early2["not_found"]
    print("[OK] preflight")

    hist = [
        {"role": "user", "content": "galatasaray nasil bir takim"},
        {"role": "assistant", "content": "Yuklenen belgelerde bu soruyla ilgili yeterli bilgi bulunmamaktadir."},
    ]
    assert looks_like_followup("peki fenerbahce", hist) is False
    expanded = expand_query("peki fenerbahce", hist)
    assert "galatasaray" not in expanded.lower() or expanded == "peki fenerbahce"
    # After abstention, expand returns query only
    assert expand_query("peki daha ne var", hist) == "peki daha ne var" or "bulunmamakta" not in expand_query("peki daha ne var", hist)
    print("[OK] followup after abstention")

    # Rerank: focus-missing chunk should rank below focus-hit
    cands = [
        {"content": "Chatbotlar egitim ve pazarlamada kullanilir.", "similarity_score": 2.0},
        {"content": "Thomas Hopkins Gallaudet Hartford Connecticut permanent school.", "similarity_score": 0.5},
    ]
    ranked = rerank_chunks("Gallaudet kimdir?", cands, 2)
    assert "Gallaudet" in ranked[0]["content"]
    print("[OK] rerank prefers focus")

    # Verifier: TR answer + EN chunk with shared name/number
    chunks = [{
        "source_file": "deneme.pdf",
        "page_number": 72,
        "content": "Thomas Hopkins Gallaudet founded the first permanent school in Hartford.",
        "is_relevant": True,
    }]
    ans = "Thomas Hopkins Gallaudet, Hartford'ta ilk kalici okulu kurmustur. (Kaynak: deneme.pdf, Sayfa 72)"
    v = verify_citations(ans, chunks, query_text="Gallaudet kimdir?")
    assert v["kind"] in {"ok", "low"}
    assert v["confidence_score"] > 0
    print("[OK] verifier name bridge")

    # finalize strips source on abstention
    out = finalize_response(
        query="x",
        cleaned="Yeterli bilgi bulunmamaktadir.\n\n(Kaynak: a.pdf, Sayfa 1)",
        usable=chunks,
        verification={"kind": "reject", "confidence_score": 0, "verification_status": "Belgede yok"},
        engine_used="fallback",
        public_chunk_fn=lambda c: c,
    )
    assert "(Kaynak:" not in out["answer"]
    assert out["chunks"]["fallback"] == []
    print("[OK] abstention strips source")

    print("--- REGRESSION PASSED ---")


if __name__ == "__main__":
    run_tests()
