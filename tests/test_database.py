import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.database import init_db, save_chunks, search_similar_chunks, clear_db, cosine_similarity, delete_source, list_documents

TEST_DB = os.path.join(os.path.dirname(__file__), "test_vector.db")

def run_tests():
    print("--- Database Module Unit Tests Starting ---")
    
    init_db(TEST_DB)
    
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]
    
    assert math.isclose(cosine_similarity(v1, v2), 1.0), "Kosinüs benzerliği aynı vektörler için 1.0 olmalı!"
    assert math.isclose(cosine_similarity(v1, v3), 0.0), "Kosinüs benzerliği dik vektörler için 0.0 olmalı!"
    print("[OK] Cosine Similarity Test Passed")

    sample_data = [
        {
            "source_file": "doc1.txt",
            "page_number": 1,
            "chunk_index": 1,
            "content": "Yapay zeka ve RAG mimarisi",
            "embedding": [1.0, 0.0, 0.0]
        },
        {
            "source_file": "doc1.txt",
            "page_number": 1,
            "chunk_index": 2,
            "content": "Aşçılık ve yemek tarifleri",
            "embedding": [0.0, 1.0, 0.0]
        }
    ]
    
    save_chunks(sample_data, db_path=TEST_DB)
    
    query_vec = [1.0, 0.0, 0.0]
    results = search_similar_chunks(query_vec, top_k=1, db_path=TEST_DB)
    
    assert len(results) == 1
    assert results[0]["content"] == "Yapay zeka ve RAG mimarisi"
    assert math.isclose(results[0]["similarity_score"], 1.0)
    print("[OK] SQLite JSON Vector Storage and Retrieval Test Passed")

    delete_source("doc1.txt", db_path=TEST_DB)
    assert list_documents(TEST_DB) == []
    print("[OK] Delete Source Test Passed")

    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
        
    print("--- ALL DATABASE TESTS PASSED SUCCESSFULLY! ---")

if __name__ == "__main__":
    run_tests()
