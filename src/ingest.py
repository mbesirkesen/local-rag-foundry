import os
from typing import List, Dict, Any
from pypdf import PdfReader

# pdfplumber kütüphanesi kontrolü
HAS_PDFPLUMBER = False
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

def extract_tables_as_markdown(page_obj) -> str:
    """
    pdfplumber sayfa nesnesinden tabloları çıkarır ve Markdown tablo formatına çevirir.
    """
    if not HAS_PDFPLUMBER:
        return ""
        
    markdown_tables = []
    try:
        tables = page_obj.extract_tables()
        for table in tables:
            if not table or len(table) < 2:  # En az 1 başlık + 1 veri satırı olmalı
                continue
                
            md_lines = []
            # Başlık Satırı (Header)
            header = [str(cell).strip() if cell else "" for cell in table[0]]
            md_lines.append("| " + " | ".join(header) + " |")
            md_lines.append("|" + "|".join(["---"] * len(header)) + "|")
            
            # Veri Satırları
            for row in table[1:]:
                row_cells = [str(cell).strip() if cell else "" for cell in row]
                md_lines.append("| " + " | ".join(row_cells) + " |")
                
            markdown_tables.append("\n".join(md_lines))
    except Exception:
        pass
        
    return "\n\n".join(markdown_tables)

def extract_text_by_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """
    PDF dosyasını sayfa sayfa okur. Hem düz metni hem de TABLOLARI (Markdown olarak) çıkarır.
    """
    pages_data = []
    
    # 1. Öncelik: pdfplumber ile Tablo ve Metin Çıkarma
    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for idx, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    tables_md = extract_tables_as_markdown(page)
                    
                    full_page_content = text.strip()
                    if tables_md:
                        full_page_content += f"\n\n--- TABLO VERİLERİ ---\n{tables_md}"
                        
                    if full_page_content.strip():
                        pages_data.append({
                            "page_number": idx + 1,
                            "text": full_page_content
                        })
            return pages_data
        except Exception as e:
            print(f"pdfplumber ayrıştırma hatası, pypdf fallback moduna geçiliyor: {e}")

    # 2. Fallback: pypdf kütüphanesi
    reader = PdfReader(pdf_path)
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages_data.append({
                "page_number": idx + 1,
                "text": text
            })
            
    return pages_data

def chunk_text(text: str, chunk_size: int = 250, overlap: int = 30) -> List[str]:
    """
    Verilen metni kelime bazlı, çakışmalı (overlapping) parçalara böler.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
        
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        
        start += (chunk_size - overlap)
        
    return chunks

def process_document(file_path: str, chunk_size: int = 250, overlap: int = 30) -> List[Dict[str, Any]]:
    """
    Ana İşleme Fonksiyonu: PDF (Metin + Tablolar) veya TXT dosyasını alır ve parçalar.
    """
    filename = os.path.basename(file_path)
    chunks_with_metadata = []
    
    if file_path.lower().endswith(".pdf"):
        pages = extract_text_by_pages(file_path)
        for page in pages:
            page_chunks = chunk_text(page["text"], chunk_size=chunk_size, overlap=overlap)
            for idx, chunk_content in enumerate(page_chunks):
                chunks_with_metadata.append({
                    "source_file": filename,
                    "page_number": page["page_number"],
                    "chunk_index": idx + 1,
                    "content": chunk_content
                })
                
    elif file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        text_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for idx, chunk_content in enumerate(text_chunks):
            chunks_with_metadata.append({
                "source_file": filename,
                "page_number": 1,
                "chunk_index": idx + 1,
                "content": chunk_content
            })
            
    return chunks_with_metadata
