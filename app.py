"""Gradio frontend of the FAQ + RAG chatbot.

Calls faq-rag-backend-api (POST /api/v1/ask, POST /api/v1/feedback, GET /api/v1/faq).
Tracing and feedback scores are handled by the backend (Langfuse).
"""

# =========================================================
# IMPORTS
# =========================================================
import logging
import os
import time
import uuid
from itertools import groupby
from typing import Optional

import gradio as gr
import requests
from pydantic import ValidationError

from config import (
    API_URL, ASK_URL, FAQ_URL, FEEDBACK_URL, GRADIO_PORT, LANGFUSE_LINK, REQUEST_TIMEOUT, TRACE_NAME,
    FAQ_TOP_K, FAQ_SIMILARITY_THRESHOLD, FAQ_ACCEPT_THRESHOLD, FAQ_JUDGE_THRESHOLD,
    TOP_K, USE_MMR, LAMBDA_MULT, SIMILARITY_THRESHOLD, MAX_TOP_K,
)
from models import AskQuestionRequest, AskQuestionResponse, FaqListResponse, FeedbackRequest

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SOURCE_LABELS = {
    "faq": "FAQ",
    "faq_judged": "FAQ (validée par le juge LLM)",
    "rag": "RAG",
}
CANDIDATE_HEADERS = ["Formulation", "Thème", "Similarité", "Cross-encoder"]
CHUNK_PREVIEW_CHARS = 500

# Default values of the 8 /ask parameters, in the order of the Gradio components
DEFAULT_PARAMS = (
    FAQ_TOP_K, FAQ_SIMILARITY_THRESHOLD, FAQ_ACCEPT_THRESHOLD, FAQ_JUDGE_THRESHOLD,
    TOP_K, SIMILARITY_THRESHOLD, USE_MMR, LAMBDA_MULT,
)


# =========================================================
# HTTP HELPERS
# =========================================================
# [WAKEUP-RETRY] réveil du Space HF endormi + retry sur 503
def _restart_space() -> None:
    """Réveille le Space via l'API HF (nécessite HF_SPACE_ID et HF_TOKEN avec droit d'écriture)."""
    space_id = os.getenv("HF_SPACE_ID")  # ex. "<owner>/<nom-du-space>"
    token = os.getenv("HF_TOKEN")
    if not space_id or not token:
        logger.warning("⚠️ HF_SPACE_ID / HF_TOKEN non définis : impossible de réveiller le Space")
        return
    # Accepte aussi l'URL complète de la page du Space
    space_id = space_id.strip().rstrip("/").removeprefix("https://huggingface.co/spaces/")
    try:
        r = requests.post(f"https://huggingface.co/api/spaces/{space_id}/restart",
                          headers={"Authorization": f"Bearer {token}"}, timeout=30)
        logger.info(f"🔄 Restart du Space {space_id} : HTTP {r.status_code}")
        if not r.ok:
            logger.warning(f"⚠️ Échec du restart : {r.text[:200]}")
    except requests.RequestException as e:
        logger.warning(f"⚠️ Échec du restart : {e}")


def _with_wakeup(method: str, url: str, retry_on_timeout: bool = True,
                 max_wait_s: int = 300, interval_s: int = 10, **kwargs):
    """HTTP request with HF Space wake-up and retry while the service answers 503.

    Generator: yields ("waiting", elapsed_s) before each retry, then ("done", response).
    retry_on_timeout=False for non-idempotent calls (e.g. POST /ask: a slow answer must not be resent).
    """
    retry_errors = (requests.ConnectionError, requests.Timeout) if retry_on_timeout else (requests.ConnectionError,)
    start = time.monotonic()
    woke = False
    while True:
        try:
            resp = requests.request(method, url, **kwargs)
            if resp.status_code != 503:
                yield "done", resp
                return
        except retry_errors as e:
            logger.warning(f"⏳ API injoignable ({e})")
        elapsed = int(time.monotonic() - start)
        if elapsed >= max_wait_s:
            raise TimeoutError(f"API toujours indisponible après {max_wait_s}s : {url}")
        if not woke:
            _restart_space()
            woke = True
        logger.info(f"⏳ Space en cours de démarrage, nouvel essai dans {interval_s}s")
        yield "waiting", elapsed
        time.sleep(interval_s)


# =========================================================
# WAKE-UP UI (status banner + query box lock)
# =========================================================
QUERY_PLACEHOLDER = "Pose ta question sur l'association Guild Open Tech puis appuie sur Entrée"


def _waking_ui(elapsed_s: int) -> tuple:
    """(status_banner, sel_query) updates while the backend is waking up."""
    return (
        gr.update(value=f"⏳ **Le backend est en veille, démarrage en cours…** ({elapsed_s}s écoulées, "
                        f"cela peut prendre 1 à 3 minutes)", visible=True),
        gr.update(interactive=False, placeholder="Backend en cours de démarrage…"),
    )


def _ready_ui(query_value: Optional[str] = None) -> tuple:
    """(status_banner, sel_query) updates once the backend answers (or gave up).

    query_value=None keeps the current text of the query box.
    """
    query_update = {"interactive": True, "placeholder": QUERY_PLACEHOLDER}
    if query_value is not None:
        query_update["value"] = query_value
    return gr.update(value="", visible=False), gr.update(**query_update)


def _api_error_message(resp: requests.Response) -> str:
    """Readable message from a backend error: {"error": ...} (400/500) or {"detail": ...} (422)."""
    try:
        body = resp.json()
    except ValueError:
        return f"HTTP {resp.status_code} : {resp.text[:300]}"
    if isinstance(body, dict):
        if "error" in body:
            return f"HTTP {resp.status_code} : {body['error']}"
        if "detail" in body:
            detail = body["detail"]
            if isinstance(detail, list):
                detail = " ; ".join(f"{'.'.join(map(str, d.get('loc', [])))} : {d.get('msg')}" for d in detail)
            return f"HTTP {resp.status_code} : {detail}"
    return f"HTTP {resp.status_code} : {body}"


# =========================================================
# FAQ CONTENT
# =========================================================
def faq_to_markdown(faq: FaqListResponse) -> str:
    """FAQ grouped by category (file order kept), one block per theme."""
    lines = [f"*{faq.count} thèmes*"]
    for category, intents in groupby(faq.intents, key=lambda i: i.category or "Sans catégorie"):
        lines.append(f"\n### {category}")
        for intent in intents:
            title = intent.canonical_question or (intent.utterances[0] if intent.utterances else intent.intent_key)
            lines.append(f"\n**{title}**  \n`{intent.intent_key}`")
            lines.extend(f"- {u}" for u in intent.utterances if u != title)
            lines.append("\n" + "\n".join(f"> {line}" for line in intent.response.splitlines()))
    return "\n".join(lines)


def load_faq():
    """Generator. Outputs: status banner, query box, FAQ content."""
    logger.info("📃 Retrieving FAQ content")
    try:
        for state, value in _with_wakeup("GET", FAQ_URL, timeout=30):
            if state == "waiting":
                yield (*_waking_ui(value), gr.skip())
        resp = value
        if not resp.ok:
            raise RuntimeError(_api_error_message(resp))
        content = faq_to_markdown(FaqListResponse(**resp.json()))
    except Exception as e:
        logger.error(f"❌ FAQ content error: {e}")
        content = f"⚠️ Impossible de charger le contenu de la FAQ : {e}"
    yield (*_ready_ui(), content)


# =========================================================
# ASK
# =========================================================
def format_answer(result: AskQuestionResponse) -> str:
    details = [f"Source : {SOURCE_LABELS.get(result.source, result.source)}"]
    if result.faq_best_score is not None:
        details.append(f"meilleur score FAQ : {result.faq_best_score:.3f}")
    footer = " · ".join(details)
    if result.judge is not None:
        verdict = "oui" if result.judge.is_answering else "non"
        footer += f"  \nJuge LLM : {verdict}" + (f" — {result.judge.reason}" if result.judge.reason else "")
    return f"{result.answer}\n\n---\n*{footer}*"


def candidates_rows(result: AskQuestionResponse) -> list[list]:
    ranked = sorted(
        result.faq_candidates,
        key=lambda c: c.cross_encoder_score if c.cross_encoder_score is not None else float("-inf"),
        reverse=True,
    )
    return [
        [c.formulation, c.theme, round(c.similarity_score, 3),
         round(c.cross_encoder_score, 3) if c.cross_encoder_score is not None else None]
        for c in ranked
    ]


def chunks_markdown(result: AskQuestionResponse) -> str:
    if not result.chunks:
        return "*Aucun extrait RAG (réponse issue de la FAQ ou aucun extrait trouvé).*"
    blocks = []
    for i, chunk in enumerate(result.chunks, start=1):
        meta = [chunk.title or chunk.id]
        if chunk.author:
            meta.append(chunk.author)
        if chunk.score is not None:
            meta.append(f"score {chunk.score:.3f}")
        content = chunk.content.strip()
        if len(content) > CHUNK_PREVIEW_CHARS:
            content = content[:CHUNK_PREVIEW_CHARS].rstrip() + "…"
        quoted = "\n".join(f"> {line}" for line in content.splitlines())
        blocks.append(f"**{i}. {' · '.join(meta)}**\n\n{quoted}")
    return "\n\n".join(blocks)


def ask_backend(request: AskQuestionRequest):
    """Generator: yields ("waiting", elapsed_s) while the Space wakes up, then ("done", AskQuestionResponse).

    Retries only on 503 / connection error (request not processed); a timeout is not retried.
    """
    for state, value in _with_wakeup("POST", ASK_URL, retry_on_timeout=False,
                                     json=request.model_dump(), timeout=REQUEST_TIMEOUT):
        if state == "waiting":
            yield state, value
    resp = value
    if not resp.ok:
        raise RuntimeError(_api_error_message(resp))
    yield "done", AskQuestionResponse(**resp.json())


def submit_query(
    query: str,
    history: Optional[list[dict]],
    session_id: Optional[str],
    faq_top_k: int,
    faq_similarity_threshold: float,
    faq_accept_threshold: float,
    faq_judge_threshold: float,
    top_k: int,
    similarity_threshold: float,
    use_mmr: bool,
    lambda_mult: float,
):
    """Send the question to the backend and update chat, trace id and answer details.

    Generator (status banner shown while the backend Space wakes up).
    Outputs: chatbot, trace_id, query box, candidates, chunks, feedback status, status banner.
    """
    skip_all = (gr.skip(),) * 7
    if not query or not query.strip():
        gr.Warning("Saisis une question.")
        yield skip_all
        return
    if faq_judge_threshold > faq_accept_threshold:
        gr.Warning(f"Le seuil du juge LLM ({faq_judge_threshold}) doit être inférieur ou égal "
                   f"au seuil d'acceptation FAQ ({faq_accept_threshold}).")
        yield skip_all
        return

    try:
        request = AskQuestionRequest(
            question=query.strip(),
            faq_top_k=int(faq_top_k),
            faq_similarity_threshold=faq_similarity_threshold,
            faq_accept_threshold=faq_accept_threshold,
            faq_judge_threshold=faq_judge_threshold,
            top_k=int(top_k),
            similarity_threshold=similarity_threshold,
            use_mmr=use_mmr,
            lambda_mult=lambda_mult if use_mmr else None,
            trace_name=TRACE_NAME,
            user_id=session_id or "anonymous",
        )
    except ValidationError as e:
        raise gr.Error(f"Paramètres invalides : {e}")

    logger.info(f"🔍 Ask: question='{request.question[:50]}', params="
                f"{request.model_dump(exclude={'question', 'trace_name', 'user_id'})}")
    error_msg = None
    try:
        for state, value in ask_backend(request):
            if state == "waiting":
                banner, query_update = _waking_ui(value)
                yield (gr.skip(), gr.skip(), query_update, gr.skip(), gr.skip(), gr.skip(), banner)
        result = value
    except requests.Timeout:
        logger.error("❌ Ask timeout")
        error_msg = f"Le backend n'a pas répondu en {REQUEST_TIMEOUT}s."
    except Exception as e:
        logger.error(f"❌ Ask error: {e}")
        error_msg = str(e)
    if error_msg:
        # Unlock the query box (question kept so the user can resend it), then show the error
        banner, query_update = _ready_ui()
        yield (gr.skip(), gr.skip(), query_update, gr.skip(), gr.skip(), gr.skip(), banner)
        raise gr.Error(error_msg)
    logger.info(f"✅ Answer: source={result.source}, faq_best_score={result.faq_best_score}, "
                f"trace_id={result.trace_id}")

    updated_history = list(history or []) + [
        {"role": "user", "content": request.question},
        {"role": "assistant", "content": format_answer(result)},
    ]
    banner, query_update = _ready_ui(query_value="")
    yield (updated_history, result.trace_id, query_update, candidates_rows(result),
           chunks_markdown(result), "", banner)


# =========================================================
# FEEDBACK (stored as a Langfuse score by the backend)
# =========================================================
def send_feedback(trace_id: Optional[str], score_value: str, feedback_type: str,
                  comment: Optional[str]) -> tuple[str, str]:
    if not trace_id:
        gr.Warning("Aucune réponse tracée à évaluer.")
        return gr.skip(), gr.skip()
    payload = FeedbackRequest(trace_id=trace_id, score_value=score_value,
                              feedback_type=feedback_type, comment=(comment or "").strip() or None)
    try:
        resp = requests.post(FEEDBACK_URL, json=payload.model_dump(), timeout=30)
        if not resp.ok:
            raise RuntimeError(_api_error_message(resp))
    except Exception as e:
        logger.error(f"❌ Feedback error: {e}")
        raise gr.Error(f"Échec de l'envoi du feedback : {e}")
    logger.info(f"✅ Feedback recorded: {feedback_type}={score_value}")
    label = "✅ Feedback positif enregistré" if score_value == "OK" else "❌ Feedback négatif enregistré"
    return label, ""


def positive_feedback(trace_id, comment):
    return send_feedback(trace_id, "OK", "helpful", comment)


def negative_feedback(trace_id, comment):
    return send_feedback(trace_id, "KO", "not_helpful", comment)


# =========================================================
# UI HELPERS
# =========================================================
def new_session_id() -> str:
    return str(uuid.uuid4())


def default_params():
    """Default values of the 8 parameters + interactive state of the lambda slider."""
    *values, lambda_mult = DEFAULT_PARAMS
    return (*values, gr.update(value=lambda_mult, interactive=USE_MMR))


def on_clear():
    # query, chatbot, trace_id, session_id, feedback_status, feedback_comment, candidates, chunks
    return ("", [], None, new_session_id(), "", "", None, "") + default_params()


# =========================================================
# GRADIO WEB INTERFACE
# =========================================================
custom_css = """
/* Responsive chatbot height (Gradio wrapper + inner containers) */
#chatbot,
#chatbot > .wrap,
#chatbot .wrap,
#chatbot .bubble-wrap {
    height: clamp(200px, 40vh, 420px) !important;
    max-height: clamp(200px, 40vh, 420px) !important;
}

@media (max-width: 768px) {
    #chatbot,
    #chatbot > .wrap,
    #chatbot .wrap,
    #chatbot .bubble-wrap {
        height: clamp(160px, 35vh, 300px) !important;
        max-height: clamp(160px, 35vh, 300px) !important;
    }
}

#faq_content { max-height: 60vh; overflow-y: auto; }

#status_banner {
    padding: 0.6em 1em;
    border-left: 4px solid #f0a020;
    background: rgba(240, 160, 32, 0.12);
    border-radius: 4px;
}
"""

logger.info("=" * 60)
logger.info("🤖 FAQ + RAG CHATBOT FRONTEND STARTUP")
logger.info(f"API URL: {API_URL}")
logger.info("=" * 60)

with gr.Blocks(title="Chatbot FAQ + RAG", css=custom_css) as demo:
    session_id = gr.State()
    current_trace_id = gr.State()

    # Header with logo and title
    with gr.Row():
        with gr.Column(scale=1, min_width=50):
            gr.Image(value="logo.png", type="filepath",
                    width=90, min_width=90,interactive=False, show_fullscreen_button=False,
                    show_share_button=False, show_download_button=False, show_label=False)
        with gr.Column(scale=5):
            gr.HTML("<h1 style='text-align: left; font-weight: bold;'>   Test et évaluation du chatbot FAQ + RAG</h1>")
            gr.HTML(f"<a href='{LANGFUSE_LINK}' target='_blank'>Langfuse traces</a>")

    status_banner = gr.Markdown(visible=False, elem_id="status_banner")

    with gr.Accordion("📃 Contenu de la FAQ", open=False):
        faq_content = gr.Markdown("*Chargement de la FAQ…*", elem_id="faq_content")

    sel_query = gr.Textbox(label="Question", placeholder=QUERY_PLACEHOLDER)
    chatbot = gr.Chatbot(type="messages", elem_id="chatbot",
                         label="Réponse (pas de mode conversation : seule la dernière question est considérée)")

    # Feedback section
    with gr.Row():
        feedback_comment = gr.Textbox(label="Commentaire (optionnel)", lines=1, scale=8)
        btn_like = gr.Button("👍 Correct", scale=1)
        btn_dislike = gr.Button("👎 Incorrect", scale=1)
    feedback_status = gr.Markdown()

    clear = gr.Button("Clear", variant="primary")

    with gr.Accordion("🔎 Candidats FAQ", open=False):
        view_candidates = gr.Dataframe(headers=CANDIDATE_HEADERS, datatype=["str", "str", "number", "number"],
                                       interactive=False, wrap=True)
    with gr.Accordion("📚 Extraits RAG", open=False):
        view_chunks = gr.Markdown()

    with gr.Accordion("⚙️ Paramètres avancés", open=False):
        gr.Markdown("**Étape FAQ** : recherche bi-encoder puis rerank cross-encoder")
        with gr.Row():
            faq_top_k = gr.Slider(1, MAX_TOP_K, value=FAQ_TOP_K, step=1, label="FAQ Top K",
                                  info="Nombre de candidats FAQ (1 par thème).")
            faq_similarity_threshold = gr.Slider(0, 1, value=FAQ_SIMILARITY_THRESHOLD, step=0.01,
                                                 label="Seuil de similarité FAQ",
                                                 info="Similarité cosinus minimale du bi-encoder.")
        with gr.Row():
            faq_accept_threshold = gr.Slider(0, 1, value=FAQ_ACCEPT_THRESHOLD, step=0.01,
                                             label="Seuil d'acceptation FAQ",
                                             info="Score cross-encoder à partir duquel la réponse FAQ "
                                                  "est renvoyée directement.")
            faq_judge_threshold = gr.Slider(0, 1, value=FAQ_JUDGE_THRESHOLD, step=0.01,
                                            label="Seuil du juge LLM",
                                            info="Score à partir duquel le juge LLM est consulté "
                                                 "(≤ seuil d'acceptation). En dessous : RAG.")
        gr.Markdown("**Étape RAG** : recherche dans la base documentaire puis génération")
        with gr.Row():
            top_k = gr.Slider(1, MAX_TOP_K, value=TOP_K, step=1, label="Top K",
                              info="Nombre d'extraits documentaires récupérés.")
            similarity_threshold = gr.Slider(0, 1, value=SIMILARITY_THRESHOLD, step=0.01,
                                             label="Seuil de similarité",
                                             info="Score minimal d'un extrait documentaire.")
        with gr.Row():
            use_mmr = gr.Checkbox(value=USE_MMR, label="Use MMR",
                                  info="Diversifie les extraits (Maximal Marginal Relevance).")
            lambda_mult = gr.Slider(0, 1, value=LAMBDA_MULT, step=0.05, label="Lambda MMR",
                                    interactive=USE_MMR,
                                    info="0 = diversité maximale, 1 = pertinence maximale.")
        btn_reset_params = gr.Button("↺ Valeurs par défaut", size="sm")


    params = [faq_top_k, faq_similarity_threshold, faq_accept_threshold, faq_judge_threshold,
              top_k, similarity_threshold, use_mmr, lambda_mult]

    # Page load: new session id + FAQ content (wakes the backend Space up if needed)
    demo.load(new_session_id, inputs=None, outputs=session_id, show_progress="hidden")
    demo.load(load_faq, inputs=None, outputs=[status_banner, sel_query, faq_content], show_progress="hidden")

    sel_query.submit(
        submit_query,
        inputs=[sel_query, chatbot, session_id, *params],
        outputs=[chatbot, current_trace_id, sel_query, view_candidates, view_chunks, feedback_status,
                 status_banner],
    )

    use_mmr.change(lambda enabled: gr.update(interactive=enabled), inputs=use_mmr, outputs=lambda_mult)
    btn_reset_params.click(default_params, inputs=None, outputs=params)

    btn_like.click(positive_feedback, inputs=[current_trace_id, feedback_comment],
                   outputs=[feedback_status, feedback_comment])
    btn_dislike.click(negative_feedback, inputs=[current_trace_id, feedback_comment],
                      outputs=[feedback_status, feedback_comment])

    clear.click(
        on_clear,
        inputs=None,
        outputs=[sel_query, chatbot, current_trace_id, session_id, feedback_status, feedback_comment,
                 view_candidates, view_chunks, *params],
    )

if __name__ == "__main__":
    # Disable experimental SSR to avoid 405 POST issues on callback routes.
    demo.queue().launch(debug=True, ssr_mode=False, server_port=GRADIO_PORT)
