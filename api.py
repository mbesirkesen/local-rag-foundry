from contextlib import asynccontextmanager
from datetime import datetime
import os
import re
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from urllib.parse import unquote
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))

from src.database import (
    chunk_count,
    delete_source,
    init_db,
    list_documents,
    list_source_files,
    save_chunks,
)
from src.ingest import process_document
from src.llm import HAS_FOUNDRY_LOCAL, LLMEngine, NOT_FOUND
from src.guardrails import finalize_response, preflight_query, select_context_chunks
from src.memory import (
    expand_query,
    infer_source,
    is_compare_query,
    is_presence_query,
    looks_like_followup,
    mentioned_source,
)
from src.retriever import retrieve_smart_chunks
from src.verifier import verify_citations

ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "data")
STATIC_DIR = os.path.join(ROOT, "static")

# Alakasız / zayıf chunk'larla cevap üretme eşiği (similarity_score 0–1+ ölçeğinde).
MIN_ANSWER_SCORE = 0.35

_engine: Optional[LLMEngine] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    os.makedirs(DATA_DIR, exist_ok=True)
    init_db()
    yield


app = FastAPI(title="Local RAG Foundry", docs_url=None, redoc_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def get_engine() -> LLMEngine:
    global _engine
    if _engine is None:
        _engine = LLMEngine()
    return _engine


def short_model_name(model_id: str) -> str:
    return (model_id or "local-model").split(":")[0]


def public_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    content = chunk.get("content") or ""
    return {
        "id": chunk.get("id"),
        "source_file": chunk.get("source_file"),
        "page_number": chunk.get("page_number"),
        "chunk_index": chunk.get("chunk_index"),
        "similarity_score": round(float(chunk.get("similarity_score") or 0), 4),
        "is_relevant": chunk.get("is_relevant", True),
        "content": content,
        "snippet": content[:280],
        "vector_score": chunk.get("vector_score"),
    }


def resolve_data_file(filename: str) -> str:
    name = os.path.basename(unquote(filename or ""))
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Geçersiz dosya adı.")
    path = os.path.abspath(os.path.join(DATA_DIR, name))
    root = os.path.abspath(DATA_DIR)
    try:
        if os.path.commonpath([root, path]) != root:
            raise HTTPException(status_code=400, detail="Geçersiz yol.")
    except ValueError:
        raise HTTPException(status_code=400, detail="Geçersiz yol.")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Dosya bulunamadı.")
    return path


def file_meta(name: str) -> Dict[str, Any]:
    path = os.path.join(DATA_DIR, name)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    mtime = os.path.getmtime(path) if os.path.exists(path) else None
    ingested = (
        datetime.fromtimestamp(mtime).isoformat(timespec="seconds") if mtime else None
    )
    return {"size": size, "last_ingested": ingested}


def knowledge_rows() -> List[Dict[str, Any]]:
    rows = []
    for doc in list_documents():
        meta = file_meta(doc["source_file"])
        rows.append(
            {
                "filename": doc["source_file"],
                "status": "Indexed",
                "chunks": doc["chunks"],
                "size": meta["size"],
                "last_ingested": meta["last_ingested"],
            }
        )
    return rows


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    source: Optional[str] = None
    history: List[ChatTurn] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=10)
    temperature: float = Field(default=0.1, ge=0, le=1)


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/status")
def status():
    engine = get_engine()
    docs = knowledge_rows()
    foundry = bool(getattr(engine, "is_foundry_active", False))
    return {
        "foundry": foundry,
        "sdk_available": bool(HAS_FOUNDRY_LOCAL),
        "model_id": engine.model_id if foundry else "fallback-hash-384",
        "model_name": short_model_name(engine.model_id) if foundry else "Hash-384",
        "vector_db": "SQLite",
        "documents_indexed": len(docs),
        "chunks_indexed": chunk_count(),
        "files": [row["filename"] for row in docs],
        "runtime": "Foundry Local" if foundry else "Yedek motor",
    }


@app.get("/api/documents")
def documents():
    return {"documents": knowledge_rows()}


@app.delete("/api/documents/{filename:path}")
def delete_document(filename: str):
    name = os.path.basename(unquote(filename or ""))
    if not name or name in {".", "..", ".gitkeep"}:
        raise HTTPException(status_code=400, detail="Geçersiz dosya adı.")
    path = os.path.abspath(os.path.join(DATA_DIR, name))
    root = os.path.abspath(DATA_DIR)
    try:
        if os.path.commonpath([root, path]) != root:
            raise HTTPException(status_code=400, detail="Geçersiz yol.")
    except ValueError:
        raise HTTPException(status_code=400, detail="Geçersiz yol.")
    indexed = name in list_source_files()
    exists = os.path.isfile(path)
    if not indexed and not exists:
        raise HTTPException(status_code=404, detail="Belge bulunamadı.")
    delete_source(name)
    if exists:
        os.remove(path)
    return {"deleted": name, "documents": knowledge_rows()}


@app.get("/api/files/{filename:path}")
def open_file(filename: str):
    path = resolve_data_file(filename)
    ext = os.path.splitext(path)[1].lower()
    media = "application/pdf" if ext == ".pdf" else "text/plain; charset=utf-8"
    return FileResponse(
        path,
        media_type=media,
        headers={
            "Content-Disposition": f'inline; filename="{os.path.basename(path)}"'
        },
    )


@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Dosya yok.")

    engine = get_engine()
    saved = []

    for file in files:
        name = os.path.basename(file.filename or "")
        ext = os.path.splitext(name)[1].lower()
        if ext not in {".pdf", ".txt"}:
            raise HTTPException(status_code=400, detail=f"Desteklenmeyen tür: {name}")

        path = os.path.join(DATA_DIR, name)
        content = await file.read()
        with open(path, "wb") as handle:
            handle.write(content)

        chunks = process_document(path)
        if not chunks:
            delete_source(name)
            if os.path.isfile(path):
                os.remove(path)
            raise HTTPException(
                status_code=400,
                detail=f"Belgeden metin çıkarılamadı: {name}",
            )

        for chunk in chunks:
            chunk["embedding"] = engine.generate_embedding(chunk["content"])
        delete_source(name)
        save_chunks(chunks)
        saved.append({"name": name, "chunks": len(chunks)})

    return {"uploaded": saved, "documents": knowledge_rows()}


def retrieve_for(engine: LLMEngine, query: str, source: Optional[str], top_k: int, mode: str):
    embed_engine = "fallback" if mode == "fallback" else "auto"
    try:
        embedding = engine.generate_embedding(query, engine=embed_engine)
    except Exception:
        embedding = []
    return retrieve_smart_chunks(
        query_text=query,
        query_embedding=embedding or [],
        top_k=top_k,
        filter_source=source,
        use_vector=bool(embedding),
    )


@app.post("/api/chat")
def chat(payload: ChatRequest):
    query = (payload.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Soru boş.")

    engine = get_engine()
    engine_used = "foundry" if engine.is_foundry_active else "fallback"
    early = preflight_query(query, engine_used)
    if early:
        return early

    history = [
        {"role": turn.role, "content": turn.content}
        for turn in (payload.history or [])
    ][-8:]
    search_query = expand_query(query, history)
    files = list_source_files()
    source = payload.source or None
    if source in {None, "", "Tüm Belgeler"}:
        source = infer_source(query, files, history)
    embed_mode = "auto"
    if is_compare_query(query):
        merged = []
        seen = set()
        for fname in files:
            for chunk in retrieve_for(engine, search_query, fname, min(3, payload.top_k), embed_mode):
                cid = chunk.get("id")
                if cid in seen:
                    continue
                seen.add(cid)
                merged.append(chunk)
        merged.sort(key=lambda item: float(item.get("similarity_score") or 0), reverse=True)
        answer_chunks = merged[: max(payload.top_k * 2, 6)]
    else:
        answer_chunks = retrieve_for(engine, search_query, source, payload.top_k, embed_mode)

    usable = select_context_chunks(query, answer_chunks, min_score=MIN_ANSWER_SCORE)
    if not usable:
        return finalize_response(
            query=query,
            cleaned=NOT_FOUND,
            usable=[],
            verification={"kind": "reject", "verification_status": "Bilgi Belgelerde Bulunamadı", "confidence_score": 0},
            engine_used=engine_used,
            public_chunk_fn=public_chunk,
        )

    prompt_query = query if (
        is_presence_query(query)
        or is_compare_query(query)
        or mentioned_source(query, files)
        or not looks_like_followup(query, history)
    ) else search_query
    response_text = engine.generate_answer(
        prompt_query,
        usable,
        temperature=payload.temperature,
        original_query=query,
    )
    cleaned = engine._sanitize_model_text(
        response_text.replace("|||---|---|---|---|", "")
        .replace("|||", "")
        .replace("||", "")
    )
    verification = verify_citations(cleaned, usable, query_text=query)
    return finalize_response(
        query=query,
        cleaned=cleaned,
        usable=usable,
        verification=verification,
        engine_used=engine_used,
        public_chunk_fn=public_chunk,
    )
