from __future__ import annotations
import os, sys
from typing import Optional, List
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Path, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
import requests
from dotenv import load_dotenv

# Para importar rag_core desde este mismo directorio
sys.path.insert(0, os.path.dirname(__file__))

from rag_core import (
    upsert_document, generate_answer,
    delete_document, doc_stats, list_docs,
    # Importamos estas para healthcheck y logs útiles
    OLLAMA_HOST, _ollama, LLM_MODEL as RAG_LLM_MODEL, EMBED_MODEL as RAG_EMBED_MODEL
)

load_dotenv()

# opcionalmente puedes seguir exponiendo estas
LLM_MODEL = os.getenv("LLM_MODEL", RAG_LLM_MODEL)
EMBED_MODEL = os.getenv("EMBED_MODEL", RAG_EMBED_MODEL)

app = FastAPI(
    title="Rare Diseases Agent (stateless RAG)",
    docs_url="/swagger",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Schemas =====
class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None
    topic: Optional[str] = None
    min_year: Optional[int] = None
    types: Optional[List[str]] = None
    lang: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    citations: Optional[List[dict]] = None
    citations_apa: Optional[List[str]] = None

# ===== Health + Chat =====
@app.get("/health")
def health():
    """Healthcheck que también comprueba conectividad con Ollama."""
    embeddings_ready = False
    try:
        # Ping ligero: lista de modelos (equivale a GET /api/tags)
        _ = _ollama.list()
        embeddings_ready = True
    except Exception:
        embeddings_ready = False

    return {
        "status": "ok",
        "llm": LLM_MODEL,
        "embed": EMBED_MODEL,
        "ollama_host": OLLAMA_HOST,
        "embeddings_ready": embeddings_ready,
    }

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        reply, metas, citations_apa = generate_answer(
            req.message,
            screen_context=req.context or "",
            topic=req.topic,
            min_year=req.min_year,
            types=req.types,
            lang=req.lang,
        )
        return ChatResponse(reply=reply, citations=metas, citations_apa=citations_apa)
    except RuntimeError as e:
        # p.ej. cuando no hay conexión con Ollama -> 503 en vez de 500
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

# ===== Ingesta =====
@app.post("/ingest")
def ingest_file(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
    topic: Optional[str] = Form(None),
    year: Optional[int] = Form(None),
    type: Optional[str] = Form(None),
    lang: Optional[str] = Form(None),
    country: Optional[str] = Form(None),
    doi: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    quality_score: Optional[float] = Form(None),
):
    try:
        name = file.filename or "unknown"
        doc_id = doc_id or name
        source = source or f"upload:{name}"
        content_type = (file.content_type or "").lower()
        raw: bytes = file.file.read()
        text = ""

        is_pdf = ("pdf" in content_type) or name.lower().endswith(".pdf")
        is_text = (
            content_type.startswith("text/")
            or name.lower().endswith(".txt")
            or name.lower().endswith(".md")
        )

        if is_pdf:
            reader = PdfReader(BytesIO(raw))
            if getattr(reader, "is_encrypted", False):
                reader.decrypt("")
            pages = [(p.extract_text() or "") for p in reader.pages]
            text = "\n".join(pages).strip()
        elif is_text:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1", errors="ignore")
            text = text.strip()
        else:
            raise HTTPException(status_code=415, detail="Formato no soportado (usa PDF, TXT o MD).")

        if not text:
            raise HTTPException(status_code=422, detail="No se extrajo texto. ¿PDF escaneado sin OCR?")

        extra = {
            "year": year, "type": type, "lang": lang, "country": country,
            "doi": doi, "url": url, "quality_score": quality_score
        }
        extra = {k: v for k, v in extra.items() if v is not None}

        count = upsert_document(doc_id=doc_id, source=source, full_text=text, topic=topic, extra_meta=extra)
        return {"ok": True, "chunks_indexed": count, "doc_id": doc_id, "source": source, "topic": topic, **extra}
    finally:
        try:
            file.file.close()
        except Exception:
            pass

@app.post("/ingest_url")
def ingest_url(req: dict):
    r = requests.get(req["url"], timeout=45)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"No se pudo descargar: {r.status_code}")
    name = req.get("doc_id") or req["url"].split("/")[-1] or "remote_doc"
    content_type = r.headers.get("content-type", "") or ""

    text = ""
    if "pdf" in content_type or name.lower().endswith(".pdf"):
        reader = PdfReader(BytesIO(r.content))
        text = "\n".join([(p.extract_text() or "") for p in reader.pages])
    elif "text" in content_type or name.lower().endswith((".txt", ".md")):
        text = r.content.decode("utf-8", errors="ignore")
    else:
        raise HTTPException(status_code=415, detail="Formato no soportado (usa PDF, TXT o MD).")

    extra = {k: v for k, v in req.items() if k not in ["url", "doc_id", "source"]}
    count = upsert_document(
        doc_id=name,
        source=req.get("source") or f"url:{req['url']}",
        full_text=text,
        topic=req.get("topic"),
        extra_meta=extra
    )
    return {"ok": True, "chunks_indexed": count, "doc_id": name, "source": req.get("source") or f"url:{req['url']}", "topic": req.get("topic"), **extra}

# ===== Admin =====
@app.get("/docs")
def list_all_docs_route():
    return {"items": list_docs()}

@app.get("/docs/{doc_id}")
def get_doc_stats_route(doc_id: str = Path(..., description="ID lógico del documento (doc_id)")):
    stats = doc_stats(doc_id)
    if stats.get("count", 0) == 0:
        raise HTTPException(status_code=404, detail=f"No se encontraron chunks para doc_id '{doc_id}'.")
    return stats

@app.delete("/docs/{doc_id}")
def delete_doc_route(
    doc_id: str,
    dry_run: bool = Query(False, description="Si true, no borra; solo indica cuántos borraría."),
):
    stats = doc_stats(doc_id)
    if dry_run:
        return {"ok": True, "deleted": 0, "would_delete": stats["count"], "doc_id": doc_id}
    deleted = delete_document(doc_id)
    return {"ok": True, "deleted": deleted, "doc_id": doc_id}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))  # Usa el puerto inyectado por Railway
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
