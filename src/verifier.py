import re
from typing import List, Dict, Any

def verify_citations(response_text: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Modelin ürettiği yanıtın, veritabanından çekilen kaynak metinlerle ne kadar örtüştüğünü doğrular.
    
    Args:
        response_text: LLM tarafından üretilen yanıt metni
        retrieved_chunks: Veritabanından çekilen kaynak parçalar listesi
        
    Returns:
        Dict: {
            "verified_citations": ["staj_rehberi.pdf (Sayfa 1)", ...],
            "details": [{"sentence": "...", "source": "...", "confidence": 85.0}],
            "confidence_score": 85.0,
            "verification_status": "Yüksek Doğruluk"
        }
    """
    if not retrieved_chunks or not response_text:
        return {
            "verified_citations": [],
            "details": [],
            "confidence_score": 0.0,
            "verification_status": "Kaynak Bulunamadı / Yetersiz"
        }
        
    # Yanıtı noktalama işaretlerine göre cümlelere böl (en az 10 karakterlik anlamlı cümleler)
    sentences = [s.strip() for s in re.split(r'[.!?]+', response_text) if len(s.strip()) > 10]
    
    matched_sources = set()
    verified_details = []
    total_matches = 0
    
    # Eğer getirilen parçalar alakasız olarak işaretlenmişse (Relevance Check)
    has_relevant_chunk = any(chunk.get("is_relevant", True) for chunk in retrieved_chunks)
    if not has_relevant_chunk or "yeterli bilgi bulunmamaktadır" in response_text.lower():
        return {
            "verified_citations": [],
            "details": [],
            "confidence_score": 0.0,
            "verification_status": "Bilgi Belgelerde Bulunamadı"
        }
        
    for sentence in sentences:
        words = set(sentence.lower().split())
        best_match_chunk = None
        best_overlap_ratio = 0.0
        
        for chunk in retrieved_chunks:
            chunk_content_lower = chunk["content"].lower()
            chunk_words = set(chunk_content_lower.split())
            if not chunk_words:
                continue
                
            # 1. Kelime kesişim oranı (Jaccard)
            common_words = words.intersection(chunk_words)
            overlap_ratio = len(common_words) / len(words) if words else 0.0
            
            # 2. Alt string eşleşmesi
            if sentence.lower() in chunk_content_lower:
                overlap_ratio = max(overlap_ratio, 0.8)
                
            if overlap_ratio > best_overlap_ratio:
                best_overlap_ratio = overlap_ratio
                best_match_chunk = chunk
                
        # Eğer cümle kelimelerinin en az %25'i kaynak metinde varsa ve parça alakalıysa doğrula
        if best_overlap_ratio >= 0.25 and best_match_chunk and best_match_chunk.get("is_relevant", True):
            total_matches += 1
            source_info = f"{best_match_chunk['source_file']} (Sayfa {best_match_chunk['page_number']})"
            matched_sources.add(source_info)
            verified_details.append({
                "sentence": sentence,
                "source": source_info,
                "confidence": round(best_overlap_ratio * 100, 1)
            })
            
    # Toplam Doğruluk Skoru Hesaplama
    confidence_score = round((total_matches / len(sentences)) * 100, 1) if sentences else 0.0
    
    status = "Yüksek Doğruluk" if confidence_score >= 60.0 else "Düşük / Şüpheli Doğruluk"
    
    return {
        "verified_citations": sorted(list(matched_sources)),
        "details": verified_details,
        "confidence_score": confidence_score,
        "verification_status": status
    }
