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


def _row_cells(words: List[Dict[str, Any]], gap: float = 16.0) -> List[str]:
    cells: List[str] = []
    buf: List[str] = []
    prev_x1 = None
    for word in sorted(words, key=lambda item: item["x0"]):
        if prev_x1 is not None and (word["x0"] - prev_x1) > gap:
            cells.append(" ".join(buf).strip())
            buf = [word["text"]]
        else:
            buf.append(word["text"])
        prev_x1 = word["x1"]
    if buf:
        cells.append(" ".join(buf).strip())
    return [cell for cell in cells if cell]


def extract_aligned_markdown_tables(page_obj) -> str:
    """Çizgisiz istatistik tablolarını kelime hizasından Markdown'a çevirir."""
    try:
        words = page_obj.extract_words() or []
    except Exception:
        return ""
    if not words:
        return ""

    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for word in words:
        key = int(round(float(word["top"]) / 6.0))
        buckets.setdefault(key, []).append(word)

    rows: List[Any] = []
    for key in sorted(buckets):
        line_words = buckets[key]
        texts = [w["text"] for w in line_words]
        n_num = sum(1 for t in texts if re.search(r"\d", t))
        if n_num < 2 or len(texts) > 16:
            rows.append(None)
            continue
        if n_num / max(len(texts), 1) < 0.22 and not re.match(
            r"^(17|18|19|20)\d{2}\b", texts[0]
        ):
            rows.append(None)
            continue
        cells = _row_cells(line_words)
        if len(cells) < 2:
            rows.append(None)
            continue
        rows.append(cells)

    tables: List[List[List[str]]] = []
    current: List[List[str]] = []
    for row in rows + [None]:
        if row is None:
            if len(current) >= 2:
                tables.append(current)
            current = []
        else:
            current.append(row)
    if len(current) >= 2:
        tables.append(current)

    markdown = []
    for table in tables:
        width = max(len(row) for row in table)
        padded = [row + [""] * (width - len(row)) for row in table]
        header, body = padded[0], padded[1:]
        lines = ["| " + " | ".join(header) + " |"]
        lines.append("|" + "|".join(["---"] * width) + "|")
        for row in body:
            lines.append("| " + " | ".join(row) + " |")
        markdown.append("\n".join(lines))
    return "\n\n".join(markdown)


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
                    aligned_md = extract_aligned_markdown_tables(page)
                    if aligned_md:
                        tables_md = "\n\n".join(part for part in (tables_md, aligned_md) if part)

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


def _token_set(text: str) -> set:
    return set(re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9]{3,}", (text or "").lower()))


def _is_heading(sent: str) -> bool:
    s = (sent or "").strip()
    if len(s) < 6 or len(s) > 90:
        return False
    letters = re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü]", "", s)
    if len(letters) < 6:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return (upper / len(letters)) >= 0.72


def _split_sentences(text: str) -> List[str]:
    placeholders: List[str] = []

    def keep(match: re.Match) -> str:
        placeholders.append(match.group(0))
        return f"«ABBR{len(placeholders) - 1}»"

    protected = re.sub(
        r"\b(?:vd|vs|örn|ör|vb|dr|mr|mrs|prof|say|vol|pp|ed|eds|nr|no|cf|etc)\.",
        keep,
        text or "",
        flags=re.I,
    )
    protected = re.sub(r"\bet al\.", keep, protected, flags=re.I)
    parts = re.split(r"(?<!\d)(?<=[.!?])\s+(?!\d)", protected)
    out: List[str] = []
    for part in parts:
        for i, orig in enumerate(placeholders):
            part = part.replace(f"«ABBR{i}»", orig)
        part = re.sub(r"\s+", " ", part).strip()
        if part:
            out.append(part)
    return out


def _content_blocks(text: str) -> List[str]:
    if "--- TABLO VERİLERİ ---" not in (text or ""):
        return [text] if (text or "").strip() else []
    blocks: List[str] = []
    pieces = re.split(r"(--- TABLO VERİLERİ ---\n?)", text)
    buf = ""
    i = 0
    while i < len(pieces):
        piece = pieces[i]
        if piece.startswith("--- TABLO VERİLERİ ---"):
            table = piece
            if i + 1 < len(pieces):
                table += pieces[i + 1]
                i += 2
            else:
                i += 1
            if buf.strip():
                blocks.append(buf.strip())
                buf = ""
            blocks.append(table.strip())
        else:
            buf += piece
            i += 1
    if buf.strip():
        blocks.append(buf.strip())
    return blocks


def semantic_chunk_text(
    text: str,
    target_words: int = 160,
    max_words: int = 280,
    min_words: int = 70,
) -> List[str]:
    """Cümle sınırlarını koruyarak konu kaymasında yeni parça açar."""
    text = (text or "").strip()
    if not text:
        return []
    chunks: List[str] = []
    for block in _content_blocks(text):
        if block.startswith("--- TABLO VERİLERİ ---") or block.lstrip().startswith("|"):
            chunks.append(block)
            continue
        sentences = _split_sentences(block)
        if not sentences:
            continue
        if sum(len(s.split()) for s in sentences) <= max_words:
            chunks.append(" ".join(sentences))
            continue
        buf: List[str] = []
        buf_words = 0
        prev_tail: set = set()
        for sent in sentences:
            words = len(sent.split())
            nxt = _token_set(sent)
            cohesion = 1.0
            if prev_tail and nxt:
                cohesion = len(prev_tail & nxt) / max(1, min(len(prev_tail), len(nxt)))
            should_break = bool(buf) and (
                buf_words >= max_words
                or (_is_heading(sent) and buf_words >= min_words)
                or (buf_words >= target_words and cohesion < 0.12)
            )
            if should_break:
                chunks.append(" ".join(buf))
                overlap = buf[-1:] if buf else []
                buf = overlap + [sent]
                buf_words = sum(len(x.split()) for x in buf)
            else:
                buf.append(sent)
                buf_words += words
            prev_tail = nxt or prev_tail
        if buf:
            body = " ".join(buf)
            if chunks and len(body.split()) < 40:
                chunks[-1] = (chunks[-1] + " " + body).strip()
            else:
                chunks.append(body)
    cleaned = [c.strip() for c in chunks if c and c.strip()]
    return cleaned or chunk_text(text)


def process_document(file_path: str, chunk_size: int = 250, overlap: int = 30) -> List[Dict[str, Any]]:
    """
    Ana İşleme Fonksiyonu: PDF (Metin + Tablolar) veya TXT dosyasını alır ve parçalar.
    """
    filename = os.path.basename(file_path)
    chunks_with_metadata = []
    
    if file_path.lower().endswith(".pdf"):
        pages = extract_text_by_pages(file_path)
        for page in pages:
            page_chunks = semantic_chunk_text(
                page["text"],
                target_words=max(90, chunk_size - 90),
                max_words=chunk_size + 40,
            )
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
            
        text_chunks = semantic_chunk_text(
            text,
            target_words=max(90, chunk_size - 90),
            max_words=chunk_size + 40,
        )
        for idx, chunk_content in enumerate(text_chunks):
            chunks_with_metadata.append({
                "source_file": filename,
                "page_number": 1,
                "chunk_index": idx + 1,
                "content": chunk_content
            })
            
    return chunks_with_metadata
