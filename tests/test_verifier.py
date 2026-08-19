import os
import sys

# Ana klasörü sys.path'e ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.verifier import verify_citations

def run_tests():
    print("--- Citation Verifier Unit Tests Starting ---")
    
    mock_chunks = [
        {
            "source_file": "rehber.pdf",
            "page_number": 3,
            "content": "Microsoft staj programı uzaktan yürütülmektedir ve 4 hafta sürer."
        }
    ]
    
    # 1. Doğru Alıntı Testi
    correct_response = "Microsoft staj programı uzaktan yürütülmektedir ve 4 hafta sürer."
    res1 = verify_citations(correct_response, mock_chunks)
    
    assert res1["confidence_score"] == 100.0, "Tam eşleşmede skor %100 olmalı!"
    assert "rehber.pdf (Sayfa 3)" in res1["verified_citations"]
    print("[OK] Verified Citation Test Passed (%100 Confidence)")

    # 2. Uydurma / Yanlış Bilgi Testi (Hallucinated Sentence)
    fake_response = "Uzay mekikleri roket yakıtı ile hidrojen kullanır."
    res2 = verify_citations(fake_response, mock_chunks)
    
    assert res2["confidence_score"] == 0.0, "Uydurma cümlede skor %0 olmalı!"
    assert res2["verification_status"] == "Düşük / Şüpheli Doğruluk"
    print("[OK] Hallucination Detection Test Passed (%0 Confidence)")

    print("--- ALL VERIFIER TESTS PASSED SUCCESSFULLY! ---")

if __name__ == "__main__":
    run_tests()
