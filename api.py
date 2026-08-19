from contextlib import asynccontextmanager
from datetime import datetime
import os
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))

from src.database import (
    chunk_count,
    clear_db,
    init_db,
    list_documents,
    list_source_files,
    save_chunks,
)
from src.ingest import process_document
from src.llm import LLMEngine
from src.retriever import retrieve_smart_chunks
from src.verifier import verify_citations

ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "data")
STATIC_DIR = os.path.join(ROOT, "static")

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
    }


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


class ChatRequest(BaseModel):
    query: str
    source: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=10)
    temperature: float = Field(default=0.1, ge=0, le=1)


class SearchRequest(BaseModel):
    query: str
    source: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=10)


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/status")
def status():
    engine = get_engine()
    docs = knowledge_rows()
    return {
        "foundry": bool(getattr(engine, "is_foundry_active", False)),
        "sdk_available": bool(getattr(engine, "is_foundry_active", False)),
        "model_id": engine.model_id,
        "model_name": short_model_name(engine.model_id),
        "vector_db": "SQLite",
        "documents_indexed": len(docs),
        "chunks_indexed": chunk_count(),
        "files": [row["filename"] for row in docs],
        "runtime": "Foundry Local SDK" if engine.is_foundry_active else "Fallback Hash Engine",
    }


@app.get("/api/documents")
def documents():
    return {"documents": knowledge_rows()}


@app.get("/api/models")
def models():
    engine = get_engine()
    active = short_model_name(engine.model_id)
    catalog = [
        {
            "id": engine.model_id,
            "name": active,
            "provider": "Foundry Local",
            "status": "loaded" if engine.is_foundry_active else "fallback",
            "active": True,
        },
        {
            "id": "phi-4-mini-instruct",
            "name": "Phi-4-mini",
            "provider": "Foundry catalog",
            "status": "available",
            "active": False,
        },
        {
            "id": "fallback-hash-384",
            "name": "Hash-384",
            "provider": "Deterministic fallback",
            "status": "ready",
            "active": not engine.is_foundry_active,
        },
    ]
    return {"active": engine.model_id, "foundry": engine.is_foundry_active, "models": catalog}


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
        for chunk in chunks:
            chunk["embedding"] = engine.generate_embedding(chunk["content"])
        if chunks:
            save_chunks(chunks)
        saved.append({"name": name, "chunks": len(chunks)})

    return {"uploaded": saved, "documents": knowledge_rows()}


def retrieve_for(engine: LLMEngine, query: str, source: Optional[str], top_k: int, mode: str):
    query_vec = engine.generate_embedding(query, engine=mode)
    return retrieve_smart_chunks(
        query_text=query,
        query_embedding=query_vec,
        top_k=top_k,
        filter_source=source,
    )


@app.post("/api/chat")
def chat(payload: ChatRequest):
    query = (payload.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Soru boş.")

    engine = get_engine()
    source = payload.source or None
    # Saklanan vektörler hash; Foundry chat modeli embedding için kullanılmaz.
    answer_chunks = retrieve_for(engine, query, source, payload.top_k, "fallback")
    engine_used = "foundry" if engine.is_foundry_active else "fallback"
    usable = [c for c in answer_chunks if c.get("is_relevant") or (c.get("similarity_score") or 0) > 0.2]
    if not usable:
        usable = answer_chunks[: payload.top_k]

    response_text = engine.generate_answer(
        query,
        usable,
        temperature=payload.temperature,
    )
    cleaned = (
        response_text.replace("|||---|---|---|---|", "")
        .replace("|||", "")
        .replace("||", "")
    )
    verification = verify_citations(response_text, usable)
    not_found = (not usable) or "bulunmamaktadır" in response_text.lower()
    return {
        "answer": cleaned,
        "not_found": not_found,
        "verification": verification,
        "engine_used": engine_used,
        "chunks": {
            "foundry": [],
            "fallback": [public_chunk(c) for c in usable],
        },
    }


@app.post("/api/search")
def search(payload: SearchRequest):
    query = (payload.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Arama boş.")

    engine = get_engine()
    source = payload.source or None
    fallback_chunks = retrieve_for(engine, query, source, payload.top_k, "fallback")
    foundry_chunks = (
        retrieve_for(engine, query, source, payload.top_k, "foundry")
        if engine.is_foundry_active
        else []
    )
    return {
        "chunks": {
            "foundry": [public_chunk(c) for c in foundry_chunks],
            "fallback": [public_chunk(c) for c in fallback_chunks],
        }
    }


@app.post("/api/reset")
def reset():
    clear_db()
    return {"ok": True, "documents": []}
