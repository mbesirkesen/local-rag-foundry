import os
import sys

# Ana klasörü sys.path'e ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.llm import LLMEngine

def run_tests():
    print("--- LLM Engine Unit Tests Starting ---")
    
    engine = LLMEngine()
    
    # 1. Embedding Üretim Testi
    sample_text = "Yapay zeka ve yerel RAG mimarisi"
    vector = engine.generate_embedding(sample_text)
    
    assert isinstance(vector, list), "Vektör bir liste olmalı!"
    assert len(vector) == 384, f"Vektör boyutu 384 olmalı, alınan: {len(vector)}"
    print("[OK] Embedding Generation Test Passed (384 dimensions)")

    # 2. Bağlamlı Yanıt Üretim Testi (Context Provided)
    mock_chunks = [
        {
            "source_file": "staj_rehberi.pdf",
            "page_number": 2,
            "content": "Microsoft staj süresi 4 haftadır."
        }
    ]
    query = "Staj ne kadar sürüyor?"
    answer = engine.generate_answer(query, mock_chunks)
    
    assert "staj_rehberi.pdf" in answer or "Microsoft staj" in answer, "Yanıt kaynak metni veya kaynak adını içermeli!"
    print("[OK] Context-Based Answer Generation Test Passed")

    # 3. Boş Bağlam Testi (No Context)
    empty_answer = engine.generate_answer("Soru", [])
    assert "bulunamadı" in empty_answer.lower(), "Boş bağlamda bulunamadı uyarısı verilmeli!"
    print("[OK] Empty Context Handling Test Passed")

    print("--- ALL LLM ENGINE TESTS PASSED SUCCESSFULLY! ---")

if __name__ == "__main__":
    run_tests()
