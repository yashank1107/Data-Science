from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum

class LLMProvider(str, Enum):
    GEMINI = "gemini"
    GROQ = "groq"
    COHERE = "cohere"

class RAGVariant(str, Enum):
    BASIC = "basic"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    HYBRID = "hybrid"

class DocumentType(str, Enum):
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    IMAGE = "image"
    PPTX = "pptx"
    XLSX = "xlsx"

class DocumentInfo(BaseModel):
    id: str
    name: str
    type: DocumentType
    upload_time: str
    size: int
    content_summary: Optional[str] = None

class ChatMessage(BaseModel):
    message: str
    images: Optional[List[str]] = None  # Base64 encoded images
    session_id: str
    document_ids: Optional[List[str]] = None

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[Dict[str, Any]]] = None
    session_id: str
    tokens_used: Optional[int] = None
    is_relevant: bool = True
    rejection_reason: Optional[str] = None

class ConfigUpdate(BaseModel):
    selected_llm: str
    selected_rag_variant: RAGVariant
    selected_documents: List[str]
    enable_internet_search: bool = False

class MemoryMessage(BaseModel):
    role: str  # user or assistant
    content: str
    timestamp: str
    document_context: Optional[List[str]] = None

class UploadResponse(BaseModel):
    success: bool
    document_id: str
    filename: str
    message: str
    document_type: DocumentType