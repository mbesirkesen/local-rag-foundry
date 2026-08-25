import sqlite3
import json
import numpy as np
import os
from typing import List, Dict, Any

# Veritabanı dosya yolu: data/vector_store.db
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vector_store.db")

def init_db(db_path: str = DB_PATH):
    """
    SQLite veritabanını ve 'document_chunks' tablosunu ilklendirir.
    Vektörler 'embedding' alanında JSON String (TEXT) formatında saklanır.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS document_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_file TEXT NOT NULL,
        page_number INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        embedding TEXT NOT NULL
    );
    """)
    
    conn.commit()
    conn.close()

def save_chunks(chunks_data: List[Dict[str, Any]], db_path: str = DB_PATH):
    """
    Parçalanmış metin ve vektör listesini veritabanına kaydeder.
    
    chunks_data nesne yapısı:
    {
        "source_file": "ders_notu.pdf",
        "page_number": 1,
        "chunk_index": 1,
        "content": "Metin içeriği...",
        "embedding": [0.12, -0.45, ...] # list veya np.ndarray
    }
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for item in chunks_data:
        # Vektörü JSON string'e çeviriyoruz (Örn: "[0.12, -0.45, 0.89]")
        embedding_val = item["embedding"]
        if isinstance(embedding_val, np.ndarray):
            embedding_val = embedding_val.tolist()
            
        embedding_json = json.dumps(embedding_val)
        
        cursor.execute("""
        INSERT INTO document_chunks (source_file, page_number, chunk_index, content, embedding)
        VALUES (?, ?, ?, ?, ?)
        """, (item["source_file"], item["page_number"], item["chunk_index"], item["content"], embedding_json))
        
    conn.commit()
    conn.close()

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    İki liste/vektör arasındaki Kosinüs Benzerliğini (Cosine Similarity) hesaplar.
    Sonuç 0.0 (tamamen farklı) ile 1.0 (birebir aynı anlama gelen) arasındadır.
    """
    if len(vec1) != len(vec2):
        return 0.0

    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    return float(dot_product / (norm_v1 * norm_v2))

def search_similar_chunks(query_embedding: List[float], top_k: int = 3, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """
    Sorgu vektörüne (query_embedding) kosinüs benzerliği en yüksek olan top_k sayıda kaydı getirir.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, source_file, page_number, chunk_index, content, embedding FROM document_chunks")
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    
    for row in rows:
        chunk_id, source_file, page_number, chunk_index, content, embedding_json = row
        # JSON string'i tekrar Python listesine çeviriyoruz
        chunk_vector = json.loads(embedding_json)
        
        if len(query_embedding) != len(chunk_vector):
            continue
            
        score = cosine_similarity(query_embedding, chunk_vector)
        
        results.append({
            "id": chunk_id,
            "source_file": source_file,
            "page_number": page_number,
            "chunk_index": chunk_index,
            "content": content,
            "similarity_score": score
        })
        
    # Skorlara göre büyükten küçüğe sırala
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return results[:top_k]

def get_page_chunks(
    source_file: str,
    page_number: int,
    db_path: str = DB_PATH,
) -> List[Dict[str, Any]]:
    """Aynı belgenin aynı sayfasındaki parçaları sırayla döndürür."""
    if not os.path.exists(db_path) or not source_file:
        return []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, source_file, page_number, chunk_index, content
        FROM document_chunks
        WHERE source_file = ? AND page_number = ?
        ORDER BY chunk_index
        """,
        (source_file, page_number),
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id": row[0],
            "source_file": row[1],
            "page_number": row[2],
            "chunk_index": row[3],
            "content": row[4],
        }
        for row in rows
    ]


def list_source_files(db_path: str = DB_PATH) -> List[str]:
    """Kayıtlı benzersiz kaynak dosya adlarını döndürür."""
    return [row["source_file"] for row in list_documents(db_path)]


def chunk_count(db_path: str = DB_PATH) -> int:
    if not os.path.exists(db_path):
        return 0
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM document_chunks")
    total = cursor.fetchone()[0]
    conn.close()
    return int(total)


def list_documents(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Dosya bazında parça sayısı döndürür."""
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT source_file, COUNT(*) as chunks, MAX(id) as last_id
        FROM document_chunks
        GROUP BY source_file
        ORDER BY source_file
        """
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {"source_file": row[0], "chunks": row[1], "last_id": row[2]}
        for row in rows
    ]


def delete_source(source_file: str, db_path: str = DB_PATH) -> None:
    if not os.path.exists(db_path):
        return
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM document_chunks WHERE source_file = ?", (source_file,))
    conn.commit()
    conn.close()


def clear_db(db_path: str = DB_PATH):
    """Veritabanındaki tüm kayıtları siler."""
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM document_chunks")
        conn.commit()
        conn.close()
