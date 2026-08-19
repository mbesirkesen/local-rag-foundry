import os
import sys
import math

# Projenin gerçek kök dizinini sys.path'e açıkça ekle
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.database import init_db, save_chunks
from src.retriever import retrieve_smart_chunks

TEST_DB = os.path.join(os.path.dirname(__file__), "test_retriever.db")

def run_tests():
    print("--- Smart Multi-Doc Retriever Unit Tests Starting ---")
    
    init_db(TEST_DB)
    
    sample_chunks = [
        {
            "source_file": "baro_raporu.pdf",
            "page_number": 1,
            "chunk_index": 1,
            "content": "Barolar çalıştay raporudur.",
            "embedding": [1.0, 0.0, 0.0]
        },
        {
            "source_file": "eylem_plani.pdf",
            "page_number": 1,
            "chunk_index": 1,
            "content": "Ulusal yapay zeka eylem planıdır.",
            "embedding": [0.9, 0.1, 0.0]
        }
    ]
    
    save_chunks(sample_chunks, db_path=TEST_DB)
    
    # 1. Filtreleme Testi (Source Filter)
    filtered = retrieve_smart_chunks(
        query_text="Çalıştay",
        query_embedding=[1.0, 0.0, 0.0],
        top_k=2,
        filter_source="baro_raporu.pdf",
        db_path=TEST_DB
    )
    
    assert len(filtered) == 1, "Sadece 1 dosya filtrelenmeliydi!"
    assert filtered[0]["source_file"] == "baro_raporu.pdf"
    print("[OK] Source File Filtering Test Passed")

    # 2. Document Score Boosting Testi
    # "baro" sorgusu sorulduğunda baro_raporu.pdf bonus almalı
    boosted = retrieve_smart_chunks(
        query_text="baro raporu detayları",
        query_embedding=[0.9, 0.1, 0.0],
        top_k=2,
        filter_source=None,
        db_path=TEST_DB
    )
    
    assert boosted[0]["source_file"] == "baro_raporu.pdf", "Boosting algoritması baro raporunu en üste çıkarmalıydı!"
    print("[OK] Score Boosting Algorithm Test Passed")

    # Temizlik
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
        
    print("--- ALL RETRIEVER TESTS PASSED SUCCESSFULLY! ---")

if __name__ == "__main__":
    run_tests()
