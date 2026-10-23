from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.rag_engine import ProcurementRAGEngine

rag_engine: ProcurementRAGEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine
    rag_engine = ProcurementRAGEngine()
    yield


app = FastAPI(
    title="Procurement Assistant RAG API",
    description="FastAPI service running fine-tuned Qwen3-0.6B with ChromaDB RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000, examples=["What approval do I need for a $3,500 laptop?"])
    top_k: int = Field(default=3, ge=1, le=10)
    max_new_tokens: int = Field(default=250, ge=50, le=512)


class ChatResponse(BaseModel):
    question: str
    answer: str
    retrieved_context: list[str]


class IngestRequest(BaseModel):
    documents: list[str] = Field(..., description="List of policy text chunks to index")
    ids: list[str] = Field(..., description="Unique IDs for each chunk (e.g., ['policy_1', 'policy_2'])")


@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "indexed_documents": rag_engine.collection.count() if rag_engine else 0,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    if not rag_engine:
        raise HTTPException(status_code=503, detail="Model is still initializing.")
    return rag_engine.ask(
        question=request.question,
        top_k=request.top_k,
        max_new_tokens=request.max_new_tokens,
    )


@app.post("/api/ingest")
async def ingest_documents(request: IngestRequest):
    if len(request.documents) != len(request.ids):
        raise HTTPException(status_code=400, detail="'documents' and 'ids' must have the exact same length.")
    
    rag_engine.add_documents(documents=request.documents, ids=request.ids)
    return {
        "message": f"Successfully indexed {len(request.documents)} policy chunks.",
        "total_documents": rag_engine.collection.count(),
    }