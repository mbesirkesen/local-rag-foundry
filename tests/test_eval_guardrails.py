import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.database import list_source_files
from src.llm import LLMEngine
from src.memory import expand_query, looks_like_followup
from src.retriever import retrieve_smart_chunks


def _engine():
    engine = LLMEngine.__new__(LLMEngine)
    engine.model_id = ""
    engine.client = None
    engine.foundry_model = None
    engine.is_foundry_active = False
    engine._page_cache = {}
    return engine


def _answer(engine, query, history=None):
    files = list_source_files()
    if not files:
        return None
    search = expand_query(query, history or [])
    chunks = retrieve_smart_chunks(search, [], top_k=5, use_vector=False)
    prompt = query if not looks_like_followup(query, history or []) else search
    return engine.generate_answer(prompt, chunks, original_query=query)


def run_tests():
    print("--- Guardrail / altın soru eval ---")
    engine = _engine()
    dummy = [{
        "source_file": "ornek.pdf",
        "page_number": 1,
        "chunk_index": 0,
        "content": "2016 yılında 190,8 milyon $ olan küresel chatbot pazarı 2025 yılında yaklaşık 1,25 milyar $ büyüklüğüne ulaşacağı tahmin edilmektedir.",
    }]

    anadolu = engine.generate_answer(
        "Anadolu-Bot projesinin 2024 başarı oranları nedir?",
        dummy,
        original_query="Anadolu-Bot projesinin 2024 başarı oranları nedir?",
    )
    assert "Anadolu-Bot" in anadolu and "geçmemektedir" in anadolu
    assert "1,25" not in anadolu and "1.25" not in anadolu
    print("[OK] Anadolu-Bot reddi")

    mixed = engine.generate_answer(
        "Alan Turing'in 1950 Turing Testi, Harry Best kitabındaki 1900 nüfus sayımını nasıl etkiledi?",
        dummy,
        original_query="Alan Turing'in 1950 Turing Testi, Harry Best kitabındaki 1900 nüfus sayımını nasıl etkiledi?",
    )
    assert "iki ayrı belgedeki" in mixed
    print("[OK] Turing / Best çapraz belge reddi")

    hist = [
        {"role": "user", "content": "Anadolu-Bot başarı oranları nedir?"},
        {"role": "assistant", "content": 'Yüklenen belgelerde "Anadolu-Bot" adlı bir proje veya özel isim geçmemektedir.'},
    ]
    assert looks_like_followup("Gallaudet kimdir?", hist) is False
    leaked = expand_query("Gallaudet kimdir?", hist)
    assert "Anadolu-Bot" not in leaked
    print("[OK] Gallaudet takip sızıntısı yok")

    files = list_source_files()
    has_deneme = any("deneme" in (f or "").lower() for f in files)
    has_article = any("chatbot" in (f or "").lower() or "kurumsal" in (f or "").lower() for f in files)
    if not has_deneme and not has_article:
        print("[SKIP] Yerel belgeler yok, korpus eval atlandı")
        print("--- EVAL PASSED ---")
        return

    if has_deneme:
        mn = _answer(
            engine,
            "Harry Best'in kitabında, 1913 yılında hangi eyalette division for the deaf kurulmuştur?",
        )
        assert mn and "Minnesota" in mn
        assert "50.1" not in mn and "%50.1" not in mn
        print("[OK] Minnesota 1913")

        emp = _answer(engine, "20 yaş ve üzeri sağır bireylerin istihdam oranı nedir?")
        assert emp and "50.1" in emp
        print("[OK] Ulusal istihdam %50.1")

        who = _answer(engine, "Gallaudet kimdir?")
        assert who and "Gallaudet" in who and "Anadolu-Bot" not in who
        print("[OK] Gallaudet kimdir")

    if has_article:
        juniper = _answer(
            engine,
            "Kurumsal iletişim makalesine göre 2026 yılına kadar küresel chatbot mesajlaşma uygulamalarına erişim sayısı kaç olacak?",
        )
        assert juniper and "9,5" in juniper and "2026" in juniper
        print("[OK] Juniper 9,5 milyar")

        market = _answer(engine, "2016 ve 2025 yıllarında küresel chatbot pazar büyüklüğü nedir?")
        assert market and "190,8" in market and "1,25" in market
        print("[OK] Pazar 2016/2025")

    print("--- EVAL PASSED ---")


if __name__ == "__main__":
    run_tests()
