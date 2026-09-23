import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.ingest import chunk_text, process_document

def run_tests():
    print("--- Ingestion Module Unit Tests Starting ---")
    
    sample_text = " ".join([f"kelime{i}" for i in range(1, 101)])
    chunks = chunk_text(sample_text, chunk_size=40, overlap=10)
    
    assert len(chunks) > 1, "Metin birden fazla parçaya bölünmeliydi!"
    first_chunk_words = chunks[0].split()
    second_chunk_words = chunks[1].split()
    
    overlap_words = first_chunk_words[-10:]
    assert overlap_words == second_chunk_words[:10], "Overlap (Çakışma) mekanizması hatalı!"
    print("[OK] Chunking & Overlap Logic Test Passed")

    from src.ingest import semantic_chunk_text
    long_text = " ".join(
        f"Thomas Hopkins Gallaudet went to France in year {i}." for i in range(1, 40)
    )
    semantic = semantic_chunk_text(long_text, target_words=80, max_words=140, min_words=40)
    assert len(semantic) > 1, "Uzun metin semantik olarak bölünmeliydi!"
    for piece in semantic:
        assert piece.strip().endswith("."), f"Cümle ortasından kesildi: {piece[-40:]}"
    print("[OK] Semantic Sentence Boundary Test Passed")

    test_txt_path = os.path.join(os.path.dirname(__file__), "test_doc.txt")
    with open(test_txt_path, "w", encoding="utf-8") as f:
        f.write("Bu birinci test paragrafıdır. Yapay zeka ve RAG mimarisi üzerine çalışıyoruz.")
        
    results = process_document(test_txt_path)
    
    assert len(results) > 0
    assert results[0]["source_file"] == "test_doc.txt"
    assert results[0]["page_number"] == 1
    assert "Yapay zeka" in results[0]["content"]
    print("[OK] Document Metadata Extraction Test Passed")

    if os.path.exists(test_txt_path):
        os.remove(test_txt_path)
        
    print("--- ALL INGESTION TESTS PASSED SUCCESSFULLY! ---")

if __name__ == "__main__":
    run_tests()
