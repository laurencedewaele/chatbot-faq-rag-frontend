"""Pydantic models mirroring the faq-rag-backend-api contract (src/adapters/inbound/api/schemas.py)."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------
# POST /api/v1/ask
# ---------------------------------------------------------
class AskQuestionRequest(BaseModel):
    question: str = Field(..., min_length=1)
    # FAQ step
    faq_top_k: Optional[int] = Field(None, ge=1, le=20)
    faq_similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    faq_accept_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    faq_judge_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    # RAG step
    top_k: int = Field(3, ge=1, le=20)
    use_mmr: bool = True
    lambda_mult: Optional[float] = Field(None, ge=0.0, le=1.0)
    similarity_threshold: float = Field(0.4, ge=0.0, le=1.0)
    # Tracing (Langfuse is handled by the backend)
    trace_name: str = Field("chatbot-frontend", min_length=1)
    user_id: str = Field("anonymous", min_length=1)


class Chunk(BaseModel):
    id: str
    content: str
    score: Optional[float] = None
    title: Optional[str] = None
    author: Optional[str] = None


class FaqCandidate(BaseModel):
    formulation: str
    theme: str
    response: str
    similarity_score: float
    cross_encoder_score: Optional[float] = None


class Judge(BaseModel):
    is_answering: bool
    reason: Optional[str] = None


class AskQuestionResponse(BaseModel):
    answer: str
    source: Literal["faq", "faq_judged", "rag"]
    chunks: list[Chunk] = Field(default_factory=list)
    faq_best_score: Optional[float] = None
    faq_candidates: list[FaqCandidate] = Field(default_factory=list)
    judge: Optional[Judge] = None
    trace_id: Optional[str] = None


# ---------------------------------------------------------
# POST /api/v1/feedback
# ---------------------------------------------------------
class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., min_length=1)
    score_value: str = Field(..., min_length=1)
    feedback_type: str = Field(..., min_length=1)
    comment: Optional[str] = None


# ---------------------------------------------------------
# GET /api/v1/faq
# ---------------------------------------------------------
class FaqIntent(BaseModel):
    intent_key: str
    category: Optional[str] = None
    canonical_question: Optional[str] = None
    utterances: list[str] = Field(default_factory=list)
    response: str


class FaqListResponse(BaseModel):
    count: int
    intents: list[FaqIntent]
