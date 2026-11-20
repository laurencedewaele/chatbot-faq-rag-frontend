import os

from dotenv import load_dotenv

# Load .env before reading any value below (config is imported before app code runs)
load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------
# Default values of the /api/v1/ask parameters
# ---------------------------------------------------------
# FAQ step
FAQ_TOP_K: int = int(os.getenv("FAQ_TOP_K", "3"))
FAQ_SIMILARITY_THRESHOLD: float = float(os.getenv("FAQ_SIMILARITY_THRESHOLD", "0.85"))
# score cross-encoder >= FAQ_ACCEPT_THRESHOLD -> réponse FAQ directe
FAQ_ACCEPT_THRESHOLD: float = float(os.getenv("FAQ_ACCEPT_THRESHOLD", "0.96"))
# FAQ_JUDGE_THRESHOLD <= score < FAQ_ACCEPT_THRESHOLD -> LLM juge ; en dessous -> RAG
FAQ_JUDGE_THRESHOLD: float = float(os.getenv("FAQ_JUDGE_THRESHOLD", "0.60"))
# RAG step
TOP_K: int = int(os.getenv("TOP_K", "3"))
USE_MMR: bool = _env_bool("USE_MMR", True)
LAMBDA_MULT: float = float(os.getenv("LAMBDA_MULT", "0.55"))
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.4"))

# Bounds accepted by the backend (AskQuestionRequest)
MAX_TOP_K: int = 20

# ---------------------------------------------------------
# Backend API
# ---------------------------------------------------------
API_URL: str = os.getenv("API_URL", "https://loren-got-chatbot-backend.hf.space").rstrip("/")
ASK_URL: str = f"{API_URL}/api/v1/ask"
FEEDBACK_URL: str = f"{API_URL}/api/v1/feedback"
FAQ_URL: str = f"{API_URL}/api/v1/faq"
HEALTH_URL: str = f"{API_URL}/health"
# RAG + LLM judge + Gemini generation can be slow
REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "90"))
# Name of the Langfuse trace created by the backend for requests sent by this app
TRACE_NAME: str = os.getenv("TRACE_NAME", "chatbot-frontend")

# ---------------------------------------------------------
# Gradio / misc
# ---------------------------------------------------------
# 7860 is the backend port when both run locally; HF Spaces sets GRADIO_SERVER_PORT=7860 itself
GRADIO_PORT: int = int(os.getenv("GRADIO_SERVER_PORT", "7861"))
LANGFUSE_LINK = "https://cloud.langfuse.com/project/cmwnkw8rg00bgad0gzz2w1bax/traces?filter=name%3BstringOptions%3B%3Bnone+of%3Bfaq-rerank%257Cfaq-retrieval%257Cfaq-judge%257Canswer-generation%257Cretrieval&dateRange=30m"
