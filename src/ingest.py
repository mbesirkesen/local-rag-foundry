import os
import re
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

_HYPHEN_STOP = {
    "ve", "ile", "veya", "çok", "bir", "bu", "en", "da", "de", "için", "the", "and",
}


def _join_hyphen_break(match: re.Match) -> str:
    left, right = match.group(1), match.group(2)
    if left.isdigit() or right[:1].isupper():
        return match.group(0)
    if right.lower() in _HYPHEN_STOP or len(right) <= 2:
        return match.group(0)
    return left + right


def repair_pdf_text(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(\w+)-\s+(\w+)", _join_hyphen_break, text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _join_wrapped_column(lines: List[str]) -> str:
    out: List[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if not out:
            out.append(line)
            continue
        prev = out[-1]
        if re.search(r"[.!?:]$", prev):
            out.append(line)
        elif re.match(r"^[a-zçğıöşü]", line):
            last = prev.split()[-1] if prev.split() else ""
            out[-1] = prev + line if len(last) <= 2 else prev + " " + line
        else:
            out.append(line)
    return " ".join(out)


def extract_page_plain_text(page) -> str:
    raw = page.extract_text(layout=True) or ""
    if raw.strip():
        left_lines: List[str] = []
        right_lines: List[str] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in re.split(r" {5,}", line.strip()) if p.strip()]
            if len(parts) >= 2:
                left_lines.append(parts[0])
                right_lines.append(" ".join(parts[1:]))
            elif parts:
                left_lines.append(parts[0])
        body = _join_wrapped_column(left_lines)
        side = _join_wrapped_column(right_lines)
        text = body + "\n\n" + side if len(side) > 200 else body
        if len(text) > 80:
            return repair_pdf_text(text)

    mid = float(page.width) / 2.0
    gutter = 18
    left_box = (0, 0, max(mid - gutter, 1), float(page.height))
    right_box = (min(mid + gutter, float(page.width) - 1), 0, float(page.width), float(page.height))
    left = (page.crop(left_box).extract_text() or "").strip()
    right = (page.crop(right_box).extract_text() or "").strip()
    if len(left) > 120 and len(right) > 120:
        text = left + "\n\n" + right
    else:
        text = (page.extract_text() or "").strip()
    return repair_pdf_text(text)


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
                    text = extract_page_plain_text(page)
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
