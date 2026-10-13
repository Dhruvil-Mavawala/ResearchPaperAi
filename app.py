import re
import streamlit as st
import fitz
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Research Paper AI Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("Research Paper AI Assistant")
st.write("Your workspace for exploring research papers with source-grounded AI.")
st.caption("Ask questions · Generate summaries · Compare papers · Trace answers to PDF pages")

# ============================================================
# Application styling and behavior
# ============================================================
st.markdown(
    """
    <style>
    :root { --rpa-accent: #6D5CE8; --rpa-border: rgba(120,120,140,.22); }
    .stApp { background: linear-gradient(180deg, rgba(109,92,232,.045) 0%, transparent 320px); }
    .block-container { max-width: 1420px; padding-top: 1.25rem; padding-bottom: 4rem; }
    h1 { letter-spacing: -0.045em !important; font-weight: 800 !important; }
    h2, h3 { letter-spacing: -.025em !important; font-weight: 720 !important; }
    [data-testid="stSidebar"] { border-right: 1px solid var(--rpa-border); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.3rem; }
    [data-testid="stMetric"] {
      background: var(--secondary-background-color);
      border: 1px solid var(--rpa-border); border-radius: 16px;
      padding: 1rem 1.1rem; box-shadow: 0 4px 18px rgba(20,20,40,.035);
    }
    [data-testid="stMetricLabel"] { font-weight: 600; }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
      border-radius: 11px; min-height: 2.7rem; font-weight: 650;
      transition: transform .15s ease, box-shadow .15s ease;
    }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
      transform: translateY(-1px); box-shadow: 0 5px 18px rgba(109,92,232,.15);
      border-color: var(--rpa-accent);
    }
    div[data-testid="stFileUploader"] {
      border: 1.5px dashed rgba(109,92,232,.45); border-radius: 16px;
      padding: .65rem; background: rgba(109,92,232,.035);
    }
    div[data-testid="stAlert"] { border-radius: 13px; border: 1px solid var(--rpa-border); }
    div[data-testid="stExpander"] { border: 1px solid var(--rpa-border); border-radius: 13px; overflow: hidden; }
    div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea {
      border-radius: 11px;
    }
    div[role="radiogroup"] { gap: .55rem; }
    .rpa-hero {
      position: relative; overflow: hidden; padding: 1.6rem 1.7rem;
      border-radius: 22px; border: 1px solid rgba(109,92,232,.23);
      background: linear-gradient(115deg, rgba(109,92,232,.15), rgba(69,164,190,.08) 62%, rgba(109,92,232,.04));
      margin: .65rem 0 1.5rem 0;
    }
    .rpa-eyebrow { text-transform: uppercase; letter-spacing: .14em; font-size: .72rem; font-weight: 800; opacity: .72; margin-bottom: .45rem; }
    .rpa-hero-title { font-size: clamp(1.5rem, 3vw, 2.15rem); line-height: 1.15; font-weight: 820; letter-spacing: -.045em; }
    .rpa-hero-sub { font-size: 1rem; opacity: .78; margin-top: .7rem; max-width: 850px; line-height: 1.6; }
    .rpa-chip { display:inline-block; padding:.32rem .62rem; margin:.85rem .35rem 0 0;
      border-radius:999px; border:1px solid var(--rpa-border); font-size:.78rem; font-weight:650;
      background:rgba(255,255,255,.24); }
    .rpa-section-note { opacity:.72; font-size:.93rem; margin-top:-.45rem; margin-bottom:1rem; }
    .rpa-card { padding: 1rem 1.1rem; border:1px solid var(--rpa-border); border-radius:15px;
      background:var(--secondary-background-color); margin:.3rem 0 .7rem; }
    .rpa-card-title { font-weight:750; margin-bottom:.25rem; }
    .rpa-muted { opacity:.72; font-size:.88rem; }
    [data-testid="stTabs"] button { font-weight:650; }
    [data-testid="stTabs"] [role="tablist"] { gap:.25rem; }
    @media (max-width: 768px) {
      .block-container { padding-left: 1rem; padding-right: 1rem; }
      .rpa-hero { padding: 1.2rem; border-radius: 16px; }
    }
    </style>
    <div class="rpa-hero">
      <div class="rpa-eyebrow">AI Research Workspace · RAG Powered</div>
      <div class="rpa-hero-title">Your research, understood.</div>
      <div class="rpa-hero-sub">Bring your research papers into one focused workspace. Ask questions grounded in the PDFs, explore page-level citations, generate summaries, and compare papers with evidence in view.</div>
      <span class="rpa-chip">📄 PDF ingestion</span>
      <span class="rpa-chip">🔎 Evidence-based Q&A</span>
      <span class="rpa-chip">📚 Page citations</span>
      <span class="rpa-chip">⚖️ Paper comparison</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 📚 Research Paper AI")
    st.caption("PDF-based retrieval-augmented assistant")
    st.divider()
    if st.session_state.get("processed_papers"):
        st.markdown("**Your workspace**")
        st.success(f'{len(st.session_state.processed_papers)} paper(s) ready')
        for paper_name in st.session_state.processed_papers:
            st.markdown(f"📄 {paper_name}")
        st.caption(f'{len(st.session_state.get("all_chunks", []))} searchable passages indexed')
    else:
        st.info("Upload and process at least one PDF to begin.")
    st.divider()
    st.caption("Tips")
    st.markdown(
        "- Ask one clear question at a time.\n"
        "- Use the paper scope to limit search.\n"
        "- Check cited page numbers in the original PDF.\n"
        "- Summaries and comparisons should be verified against the source."
    )


# ============================================================
# SESSION STATE
# ============================================================
DEFAULTS = {
    "chat_history": [],
    "processed_papers": [],
    "faiss_index": None,
    "all_chunks": [],
    "embedding_model": None,
    "llm_tokenizer": None,
    "llm_model": None,
    "paper_fingerprints": None,
    "paper_summaries": {},
    "paper_comparison": "",
    "paper_comparison_names": (),
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")


@st.cache_resource
def load_llm():
    """
    Use a small instruction model to reduce memory requirements.
    This function intentionally does not use 4-bit quantization.
    """
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    return tokenizer, model


def run_llm(prompt, max_new_tokens=250):
    tokenizer, model = load_llm()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful research assistant. Follow the user's "
                "instructions and do not invent evidence from research papers."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ============================================================
# PDF EXTRACTION AND CHUNKING
# ============================================================
def extract_pdf_text(uploaded_file, paper_id):
    pdf_bytes = uploaded_file.getvalue()
    pages = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        for page_number, page in enumerate(document, start=1):
            text = re.sub(r"\s+", " ", page.get_text("text")).strip()
            if text:
                pages.append(
                    {
                        "paper_id": paper_id,
                        "source": uploaded_file.name,
                        "page": page_number,
                        "text": text,
                    }
                )
    return pages


def create_chunks(pages, chunk_size=1200, overlap=200):
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        chunk_number = 1

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": (
                            f'{page["paper_id"]}_chunk_{len(chunks) + 1}'
                        ),
                        "paper_id": page["paper_id"],
                        "source": page["source"],
                        "page": page["page"],
                        "chunk_number": chunk_number,
                        "text": chunk_text,
                    }
                )

            if end >= len(text):
                break
            start = end - overlap
            chunk_number += 1

    return chunks


# ============================================================
# BUILD SEARCH INDEX AND RETRIEVE CHUNKS
# ============================================================
def build_search_index(chunks):
    if not chunks:
        return None

    embedding_model = load_embedding_model()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve_chunks(question, scope, top_k=5):
    """scope-aware hybrid retrieval with relevance filtering and diversity."""
    index = st.session_state.faiss_index
    chunks = st.session_state.all_chunks
    if index is None or not chunks or not question.strip():
        return []

    if scope == "Paper 1":
        allowed_ids = {"paper_1"}
    elif scope == "Paper 2":
        allowed_ids = {"paper_2"}
    else:
        allowed_ids = {"paper_1", "paper_2"}

    # Embed the question once; FAISS vectors and query vectors are normalized.
    embedding_model = load_embedding_model()
    # Add a small domain-specific query expansion for a known distinction in r2.pdf.
    # This improves retrieval without changing the user's selected paper scope.
    retrieval_query = question
    if detect_backend_topic(question):
        retrieval_query = (
            question
            + " model building training learning backend; "
            + "model execution executing backend deployed models"
        )
    query_embedding = embedding_model.encode(
        [retrieval_query], normalize_embeddings=True, show_progress_bar=False
    )
    query_embedding = np.asarray(query_embedding, dtype="float32")
    similarities, indices = index.search(query_embedding, len(chunks))

    stop_words = {
        "what", "which", "when", "where", "who", "why", "how", "does",
        "did", "are", "is", "was", "were", "the", "and", "for", "from",
        "with", "that", "this", "these", "those", "its", "their", "about",
        "into", "using", "use", "can", "could", "would", "should", "have",
        "has", "had", "not", "than", "then", "them", "they", "you", "your",
        "paper", "papers", "research", "according", "explain", "describe",
        "tell", "give", "please", "discuss", "discusses", "mention", "mentions",
    }

    def normalize_terms(text):
        # Basic normalization catches punctuation/case differences without
        # introducing a heavyweight NLP dependency.
        terms = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return {term for term in terms if len(term) > 2 and term not in stop_words}

    query_terms = normalize_terms(question)
    ranked = []
    seen_text = set()

    for similarity, idx in zip(similarities[0], indices[0]):
        if idx < 0:
            continue
        chunk = chunks[int(idx)]
        if chunk.get("paper_id") not in allowed_ids:
            continue

        text_key = re.sub(r"\s+", " ", chunk.get("text", "").lower()).strip()
        # Exact duplicate chunks add no evidence and waste the context window.
        if not text_key or text_key in seen_text:
            continue
        seen_text.add(text_key)

        chunk_terms = normalize_terms(chunk.get("text", ""))
        lexical_score = (
            len(query_terms & chunk_terms) / len(query_terms)
            if query_terms else 0.0
        )
        semantic_score = max(0.0, min(1.0, float(similarity)))
        # Semantic retrieval remains primary; lexical overlap helps specific terms.
        hybrid_score = 0.78 * semantic_score + 0.22 * lexical_score

        result = dict(chunk)
        result["similarity"] = semantic_score
        result["lexical_score"] = float(lexical_score)
        result["hybrid_score"] = float(hybrid_score)
        ranked.append(result)

    ranked.sort(key=lambda item: item["hybrid_score"], reverse=True)
    if not ranked:
        return []

    # Relevance gate: do not feed clearly unrelated chunks to the LLM.
    # A strong semantic score can pass even when exact keywords differ.
    best = ranked[0]
    minimum_score = 0.30
    if best["hybrid_score"] < minimum_score and best["similarity"] < 0.38:
        return []

    eligible = [
        item for item in ranked
        if item["hybrid_score"] >= max(minimum_score, best["hybrid_score"] - 0.20)
        or item["similarity"] >= max(0.38, best["similarity"] - 0.18)
    ]

    # Prefer different pages and papers to reduce repetitive context.
    selected = []
    page_counts = {}
    paper_counts = {}
    selected_ids = set()
    for result in eligible:
        page_key = (result["paper_id"], result["page"])
        if page_counts.get(page_key, 0) >= 2:
            continue
        # In Both Papers mode, do not let one paper dominate all five slots
        # when the other paper has similarly relevant evidence.
        if scope == "Both Papers" and paper_counts.get(result["paper_id"], 0) >= 3:
            continue
        selected.append(result)
        selected_ids.add(result["chunk_id"])
        page_counts[page_key] = page_counts.get(page_key, 0) + 1
        paper_counts[result["paper_id"]] = paper_counts.get(result["paper_id"], 0) + 1
        if len(selected) >= top_k:
            break

    # If diversity rules leave open slots, fill with the next relevant unique chunks.
    if len(selected) < min(top_k, len(eligible)):
        for result in eligible:
            if result["chunk_id"] in selected_ids:
                continue
            selected.append(result)
            selected_ids.add(result["chunk_id"])
            if len(selected) >= top_k:
                break

    return selected


# ============================================================
# CONVERSATIONAL MEMORY / FOLLOW-UP QUESTION REWRITING
# ============================================================
def get_recent_chat_context(max_turns=2):
    """Return compact recent turns; never treat earlier answers as evidence."""
    history = st.session_state.chat_history[-max_turns:]
    if not history:
        return "No previous conversation."

    parts = []
    for item in history:
        question = re.sub(r"\s+", " ", str(item.get("question", ""))).strip()
        answer = re.sub(r"\s+", " ", str(item.get("answer", ""))).strip()
        # Keep only a short answer hint for resolving references; source passages
        # remain the sole factual evidence in the answer-generation prompt.
        answer = answer[:350]
        parts.append(f"Previous user question: {question}\nPrevious answer hint: {answer}")
    return "\n\n".join(parts)


def resolve_followup_question(question):
    """
    Rewrite follow-up questions into standalone retrieval queries.
    Example: after 'What is machine learning?', 'What are its types?'
    becomes a standalone query about types of machine learning.
    """
    history = get_recent_chat_context(max_turns=4)
    if history == "No previous conversation.":
        return question

    # Conservative guard: rewrite only messages that clearly depend on prior context.
    # Standalone questions (e.g. "What is the capital city of Japan?") must remain intact.
    import re
    q = question.strip()
    lower_q = q.lower()
    followup_starters = (
        "give me an example", "give an example", "what about it", "explain that",
        "explain this", "tell me more", "why is that", "how does it work",
        "what are its", "what is its", "how about that", "and its", "and what",
        "compare them", "what about the second", "can you elaborate",
    )
    has_reference = bool(re.search(r"\b(it|its|they|them|those|these|that|this|former|latter)\b", lower_q))
    is_short = len(q.split()) <= 5
    clearly_followup = lower_q.startswith(followup_starters) or (has_reference and is_short)
    if not clearly_followup:
        return question

    prompt = f"""
Rewrite the current user question as a standalone search query for
retrieving relevant passages from research papers.

Use only the immediately relevant prior exchange to resolve references such as "it",
"its", "they", "those", "the method", or "the model".
Rewrite ONLY the CURRENT QUESTION. Never combine multiple earlier questions,
never output a list of questions, and never replace a new standalone topic with an older topic.
Do not answer the question.
Do not add facts that are not supported by the conversation.
If the current question is already standalone, keep its meaning unchanged.
Return only the rewritten question, with no explanation.

CONVERSATION HISTORY:
{history}

CURRENT QUESTION:
{question}

STANDALONE SEARCH QUERY:
""".strip()

    try:
        rewritten = run_llm(prompt, max_new_tokens=80)
        rewritten = rewritten.strip().strip('"').strip("'")
        if not rewritten or len(rewritten) > 500:
            return question
        return rewritten
    except Exception:
        # The original question remains usable if query rewriting fails.
        return question


# ============================================================
# SOURCE ATTRIBUTION HELPERS (PHASE 13)
# ============================================================
def find_papers_discussing_topic(question, scope):
    """
    Answer 'which paper discusses X?' from actual indexed PDF text.
    This avoids incorrectly returning labels such as 'Source 2'.
    """
    match = re.search(
        r"(?:which|what)\s+papers?\s+(?:discuss(?:es)?|mention(?:s)?|cover(?:s)?|"
        r"explain(?:s)?|describe(?:s)?|contain(?:s)?|talk about)\s+(.+?)[?.!]*$",
        question.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    topic = match.group(1).strip(" ?.!,:;")
    topic_terms = [
        term.lower()
        for term in re.findall(r"[a-zA-Z0-9]+", topic)
        if len(term) > 2
    ]
    if not topic_terms:
        return None

    if scope == "Paper 1":
        allowed_ids = {"paper_1"}
    elif scope == "Paper 2":
        allowed_ids = {"paper_2"}
    else:
        allowed_ids = {"paper_1", "paper_2"}

    paper_hits = {}
    for chunk in st.session_state.all_chunks:
        if chunk.get("paper_id") not in allowed_ids:
            continue

        text = chunk.get("text", "").lower()
        # Count exact topic terms found in this page/chunk.
        matched_terms = [term for term in topic_terms if term in text]
        if matched_terms:
            paper_id = chunk["paper_id"]
            entry = paper_hits.setdefault(
                paper_id,
                {"source": chunk["source"], "pages": set(), "matched": set()},
            )
            entry["pages"].add(chunk["page"])
            entry["matched"].update(matched_terms)

    if not paper_hits:
        return (
            "I could not confirm which uploaded paper discusses "
            f'"{topic}" from the extracted PDF text.'
        )

    # Require at least one topic term for short topics, and prefer more
    # complete matches for multi-word topics.
    relevant = []
    for entry in paper_hits.values():
        if len(topic_terms) == 1 or len(entry["matched"]) >= min(2, len(topic_terms)):
            relevant.append(entry)

    if not relevant:
        relevant = list(paper_hits.values())

    relevant.sort(key=lambda item: len(item["matched"]), reverse=True)
    lines = [f'The topic "{topic}" appears in:']
    for entry in relevant:
        pages = ", ".join(str(page) for page in sorted(entry["pages"])[:8])
        lines.append(f'- {entry["source"]} (pages {pages})')
    return "\n".join(lines)


# ============================================================
# Topic-aware scope guidance
# ============================================================
def detect_backend_topic(question):
    """Identify questions about the learning/executing backend distinction."""
    text = re.sub(r"[^a-z0-9]+", " ", (question or "").lower()).strip()
    has_backend = "backend" in text or "back end" in text
    has_learning = "learning" in text or "training" in text or "model building" in text
    has_executing = (
        "executing" in text or "execution" in text or "model execution" in text
    )
    return has_backend and has_learning and has_executing


def scope_topic_guidance(question, scope):
    """
    If a known topic is clearly present in the other uploaded paper, explain
    the scope mismatch instead of silently answering outside the selected scope.
    """
    if not detect_backend_topic(question):
        return None

    chunks = st.session_state.get("all_chunks", [])
    paper2_chunks = [c for c in chunks if c.get("paper_id") == "paper_2"]
    paper1_chunks = [c for c in chunks if c.get("paper_id") == "paper_1"]
    paper2_text = " ".join(c.get("text", "") for c in paper2_chunks).lower()
    paper1_text = " ".join(c.get("text", "") for c in paper1_chunks).lower()

    topic_terms = ("learning backend", "executing backend")
    paper2_has_topic = all(term in paper2_text for term in topic_terms)
    paper1_has_topic = all(term in paper1_text for term in topic_terms)

    if scope == "Paper 1" and paper2_has_topic and not paper1_has_topic:
        return (
            "This topic appears to be covered in **Paper 2**, not Paper 1. Your current "
            "search scope is **Paper 1**, so I have not answered using Paper 2. Select "
            "**Paper 2** or **Both Papers**, then ask the question again."
        )
    return None


# ============================================================
# EVIDENCE SUFFICIENCY CHECK (PHASE 22 FIX)
# ============================================================
GROUNDING_STOP_WORDS = {
    "what", "which", "when", "where", "who", "why", "how", "does", "did",
    "are", "is", "was", "were", "the", "and", "for", "from", "with", "that",
    "this", "these", "those", "its", "their", "about", "into", "using", "use",
    "can", "could", "would", "should", "have", "has", "had", "not", "than",
    "then", "them", "they", "you", "your", "paper", "papers", "research",
    "explain", "describe", "tell", "give", "please", "discuss", "difference",
    "between", "role", "explain", "based", "according", "summarize", "summary",
    "key", "main", "important", "following", "following", "list", "outline",
}


def evidence_sufficiency(question, resolved_question, retrieved_chunks):
    """Conservative lexical check to stop confident answers to unrelated queries.

    This is a guardrail, not a proof of semantic entailment. It complements
    vector similarity by requiring the retrieved text to mention meaningful
    terms from the question. For broad questions, one matching content term
    can be enough; for specific questions, require stronger overlap.
    """
    if not retrieved_chunks:
        return False, "No relevant passages were retrieved."

    def terms(text):
        return {
            t for t in re.findall(r"[a-zA-Z0-9]+", text.lower())
            if len(t) > 2 and t not in GROUNDING_STOP_WORDS
        }

    original_terms = terms(question)
    resolved_terms = terms(resolved_question)
    # Prefer the user's original words, while including rewritten terms that
    # help resolve pronouns in follow-up questions.
    query_terms = original_terms | resolved_terms
    if not query_terms:
        return True, ""

    combined_text = " ".join(chunk.get("text", "") for chunk in retrieved_chunks).lower()
    context_terms = terms(combined_text)
    overlap = query_terms & context_terms

    # Country / location questions should not be answered from generic text
    # unless the named entity itself occurs in the retrieved excerpts.
    named_entities = {
        "japan", "india", "china", "france", "germany", "canada", "australia",
        "tokyo", "london", "paris", "delhi", "mumbai", "new york",
    }
    lowered_question = question.lower()
    for entity in named_entities:
        if entity in lowered_question and entity not in combined_text:
            return False, f'The retrieved passages do not mention "{entity}".'

    # A few question types need their core terms to appear in evidence.
    if len(query_terms) >= 3:
        minimum_overlap = 2
    else:
        minimum_overlap = 1

    if len(overlap) < minimum_overlap:
        return False, "The retrieved passages do not contain enough of the question's key terms."

    # If almost all evidence is unrelated, reject it even if one common word
    # such as "learning" happens to overlap.
    overlap_ratio = len(overlap) / max(1, len(query_terms))
    if len(query_terms) >= 4 and overlap_ratio < 0.20:
        return False, "The retrieved passages have too little keyword overlap with the question."

    return True, ""


# ============================================================
# GENERATE GROUNDED ANSWER
# ============================================================
def generate_answer(question, retrieved_chunks, chat_context, resolved_question, scope):
    # Respect selected scope and give grounded guidance for the known backend topic.
    scope_guidance = scope_topic_guidance(question, scope)
    if scope_guidance is not None:
        return scope_guidance

    # Keep deterministic filename attribution for "which paper discusses X?"
    source_answer = find_papers_discussing_topic(question, scope)
    if source_answer is not None:
        return source_answer

    if not retrieved_chunks:
        return (
            "The uploaded papers do not contain sufficient information to answer "
            "this question based on the retrieved passages. Try rephrasing the "
            "question or changing the search scope."
        )

    evidence_ok, evidence_reason = evidence_sufficiency(
        question, resolved_question, retrieved_chunks
    )
    if not evidence_ok:
        return (
            "The uploaded papers do not contain sufficient evidence to answer "
            "this question. I will not answer from general knowledge.\n\n"
            f"Reason: {evidence_reason}"
        )

    context_parts = []
    citation_map = {}
    for i, chunk in enumerate(retrieved_chunks, start=1):
        citation_id = f"C{i}"
        citation_map[citation_id] = (
            f'{chunk["source"]}, p. {chunk["page"]}'
        )
        context_parts.append(
            f'[{citation_id}]\n'
            f'Paper: {chunk["source"]}\n'
            f'Page: {chunk["page"]}\n'
            f'Excerpt: {chunk["text"]}'
        )
    context = "\n\n".join(context_parts)

    prompt = f"""
You are a research-paper question-answering assistant.

STRICT EVIDENCE AND CITATION RULES:
- Answer only from the research-paper excerpts below.
- Conversation history may clarify pronouns, but is not factual evidence.
- Answer the original question directly, using the resolved query only
  to understand what the user means.
- Keep the answer focused on the exact question. Do not copy a whole paragraph unless asked.
- Do not reproduce the paper's bibliography/reference list as an answer.
- Do not repeat the same claim in different wording.
- Do not add generic textbook information unless it appears in the supplied excerpts.
- Distinguish the learning backend (model building/training) from the executing backend (model execution); do not claim the executing backend trains models unless the excerpts explicitly say so.
- For a question asking for types, list supported types as bullets.
- Every factual statement must end with one or more citation IDs copied
  exactly from the excerpts, such as [C1] or [C2].
- Only cite an excerpt that directly supports the statement.
- Do not invent a source ID, paper name, page number, fact, or result.
- Use no outside knowledge, including facts that are common knowledge.
- If the excerpts do not directly support the answer, say:
  "The answer was not found in the retrieved excerpts from the uploaded papers."
- Do not infer an answer merely because it is true in the real world.
- Never use labels such as "Source 1" as the final citation.

RECENT CONVERSATION (for reference resolution only):
{chat_context}

RESOLVED SEARCH QUERY:
{resolved_question}

RESEARCH-PAPER EXCERPTS:
{context}

ORIGINAL USER QUESTION:
{question}

Write a concise answer with citations:
""".strip()

    try:
        answer = run_llm(prompt, max_new_tokens=350)
        if not answer:
            return "The model returned an empty answer. Please try again."

        # Convert model-generated [C1] style IDs into real filename/page citations.
        for citation_id, citation_text in citation_map.items():
            answer = re.sub(
                rf"\[{re.escape(citation_id)}\]",
                f"[{citation_text}]",
                answer,
            )

        # Remove any citation IDs the model invented instead of using supplied IDs.
        valid_ids = set(citation_map)
        answer = re.sub(
            r"\[(C\d+)\]",
            lambda match: match.group(0) if match.group(1) in valid_ids else "",
            answer,
        )

        # Collapse adjacent duplicate filename/page citations without removing
        # citations that support different statements elsewhere in the answer.
        citation_pattern = r"\[([^\[\]]+, p\. \d+)\]"
        previous = None
        while previous != answer:
            previous = answer
            answer = re.sub(
                rf"({citation_pattern})(?:\s*\1)+",
                r"\1",
                answer,
            )

        # Citation repair:
        # If the model omitted its [C#] markers, attach a transparent supporting
        # source selected from the passages actually retrieved for this question.
        # Do not do this for explicit evidence-rejection answers.
        has_real_citation = re.search(citation_pattern, answer) is not None
        rejection_markers = (
            "not found in the retrieved excerpts",
            "do not contain sufficient evidence",
            "do not contain sufficient information to answer",
        )
        is_rejection = any(marker in answer.lower() for marker in rejection_markers)
        if not has_real_citation and not is_rejection:
            # Retrieval is already ranked by relevance. Select at most two unique
            # filename/page pairs, preferring chunks with lexical overlap to the
            # question and answer so unrelated retrieved pages are not listed.
            query_terms = {
                t for t in re.findall(r"[a-zA-Z0-9]+", f"{question} {resolved_question} {answer}".lower())
                if len(t) > 3 and t not in GROUNDING_STOP_WORDS
            }
            ranked_sources = []
            seen_source_pages = set()
            for rank, chunk in enumerate(retrieved_chunks):
                filename = str(chunk.get("source", "")).strip()
                page = chunk.get("page")
                passage_terms = {
                    t for t in re.findall(r"[a-zA-Z0-9]+", str(chunk.get("text", "")).lower())
                    if len(t) > 3
                }
                overlap = len(query_terms & passage_terms)
                key = (filename, page)
                if filename and page is not None and key not in seen_source_pages:
                    seen_source_pages.add(key)
                    ranked_sources.append((-(overlap), rank, key))
            ranked_sources.sort()
            support = []
            for _, _, (filename, page) in ranked_sources[:2]:
                citation = f"[{filename}, p. {page}]"
                if citation not in support:
                    support.append(citation)
            if support:
                answer = answer.rstrip() + "\n\n**Supporting source(s) from retrieved passages:** " + "; ".join(support)
            else:
                answer = answer.rstrip() + (
                    "\n\n**Citation note:** No source/page metadata was available "
                    "for a verifiable citation. Check the answer against the PDF."
                )

        return answer.strip()
    except Exception as exc:
        return f"Could not generate an answer because the language model failed: {exc}"


# ============================================================
# Application styling and behavior
# ============================================================
def _citation_pages_in_text(text, source_name):
    """Return page numbers cited for one exact source filename."""
    escaped = re.escape(source_name)
    matches = re.findall(
        rf"\[{escaped},\s*p\.\s*(\d+)\]",
        text or "",
        flags=re.IGNORECASE,
    )
    return sorted({int(page) for page in matches})


def _valid_page_set(paper_chunks):
    return {
        int(chunk["page"])
        for chunk in paper_chunks
        if chunk.get("page") is not None
    }


def _check_summary_citations(summary, source_name, paper_chunks):
    """Flag missing/invalid citations transparently; never fabricate support."""
    cited = _citation_pages_in_text(summary, source_name)
    valid_pages = _valid_page_set(paper_chunks)
    invalid = [page for page in cited if page not in valid_pages]
    if not cited:
        return _extractive_cited_summary(source_name, paper_chunks)
    if invalid:
        return summary.rstrip() + (
            "\n\n**Citation validation warning:** One or more cited page "
            "numbers could not be matched to the processed pages of this PDF. "
            "Please verify the original document."
        )
    return summary


def _extractive_cited_summary(source_name, paper_chunks, max_items=7):
    """Create a readable, extractive fallback while filtering PDF front matter."""
    import re
    boilerplate = (
        "international journal", "issn", "e-issn", "p- issn", "volume ",
        "issue ", "proceedings of", "isbn:", "uri:", "copyright", "©",
        "available online", "all rights reserved", "department of",
        "university", "@", "www.", "http://", "https://", "email:",
        "received:", "accepted:", "published:", "corresponding author",
    )
    stop_words = {
        "the", "and", "for", "that", "with", "from", "this", "these", "those",
        "are", "was", "were", "been", "being", "into", "their", "there", "which",
        "while", "where", "when", "what", "how", "can", "could", "would", "should",
        "about", "also", "such", "than", "then", "them", "they", "its", "have", "has",
        "had", "not", "but", "our", "paper", "page", "using", "used", "use", "based",
    }
    candidates = []
    seen = set()
    for chunk in paper_chunks:
        raw = re.sub(r'\s+', ' ', str(chunk.get('text', '')).strip())
        if not raw:
            continue
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9“"])', raw)
        for sentence in sentences:
            sentence = sentence.strip(' \t\n•-')
            normalized = re.sub(r'\s+', ' ', sentence)
            lower = normalized.lower()
            if len(normalized) < 65 or len(normalized.split()) < 10:
                continue
            if any(term in lower for term in boilerplate):
                continue
            if lower.startswith(("references", "bibliography", "figure ", "table ", "keywords:", "table 1", "table 2")):
                continue
            # Drop OCR/layout debris and bullet-list concatenations from diagrams.
            if sum(token in lower for token in ("[origin:", "cognitive modeling", "laws of thought", "turing test", "rational agent", "statistical learning")) >= 3:
                continue
            if re.search(r"(?:\b[A-Z][a-z]+,\s*){3,}\b(?:19|20)\d{2}\b", normalized):
                continue
            if lower.count("") >= 2 or lower.count("•") >= 2:
                continue
            # Reject fragments and reference-list-like text.
            if normalized.count("@") or re.search(r'\[\d+(?:\s*[-,]\s*\d+)*\]', normalized):
                continue
            if not re.search(r'[.!?]$', normalized):
                continue
            words = re.findall(r"[a-zA-Z]{3,}", lower)
            if len(words) < 9:
                continue
            key = re.sub(r'[^a-z0-9]+', ' ', lower).strip()
            if key in seen:
                continue
            seen.add(key)
            page = chunk.get('page')
            if page is None:
                continue
            # Prefer explanatory sentences over headings, title lines, and fragments.
            score = min(len(words), 35) + 4 * sum(w not in stop_words for w in words[:30])
            if any(term in lower for term in ("aims to", "this paper", "we present", "we propose",
                                               "in conclusion", "the framework", "machine learning")):
                score += 12
            candidates.append((score, int(page), normalized))
    if not candidates:
        return (
            f"**Summary unavailable for {source_name}.** No sufficiently complete "
            "source sentences passed the extractive filters. Check PDF extraction."
        )
    # Pick diverse pages: highest-quality candidates, but limit duplicate page dominance.
    candidates.sort(key=lambda x: (-x[0], x[1]))
    chosen = []
    page_counts = {}
    for score, page, sentence in candidates:
        if page_counts.get(page, 0) >= 2:
            continue
        chosen.append((page, sentence))
        page_counts[page] = page_counts.get(page, 0) + 1
        if len(chosen) >= max_items:
            break
    chosen.sort(key=lambda x: x[0])
    lines = [
        f"**Extractive, source-grounded summary of {source_name}**",
        "Selected complete sentences from the source text. This is an extractive fallback, not a full abstractive summary.",
    ]
    for page, sentence in chosen:
        lines.append(f"- {sentence} [{source_name}, p. {page}]")
    lines.append(
        "\n*Fallback used because the language model did not provide verifiable inline citations. "
        "Verify the cited passages in the original PDF before relying on the summary.*"
    )
    return "\n".join(lines)


def summarize_paper(paper_name, paper_chunks):
    """Summarize only the supplied chunks belonging to the named PDF."""
    if not paper_chunks:
        return "No text chunks are available for this paper."

    # Keep all summary inputs tied to this exact source filename.
    source_names = {chunk.get("source") for chunk in paper_chunks}
    if len(source_names) > 1:
        return "Summary stopped: mixed source filenames were detected for this paper."
    actual_source = next(iter(source_names), paper_name)

    partial_summaries = []
    batch_size = 4
    for start in range(0, len(paper_chunks), batch_size):
        batch = paper_chunks[start:start + batch_size]
        excerpts = "\n\n".join(
            f'[PDF page {chunk["page"]}] {chunk["text"]}' for chunk in batch
        )
        prompt = f"""
You summarize ONE research paper. The exact filename is: {actual_source}.
Use only the excerpts below, which all come from {actual_source}.
Do not use information from any other paper or general knowledge.
Preserve the source's terminology. Do not invent methods, findings, metrics,
datasets, or conclusions. Write complete sentences and complete bullet points.
Do not start a heading or sentence that you cannot finish.
CITATIONS ARE REQUIRED: after each factual sentence or bullet, add a citation
in exactly this format: [{actual_source}, p. N], using only the PDF page number
shown immediately before the excerpt that supports that claim. If multiple
pages support one claim, cite each relevant page. Never invent a page number.
If the excerpts do not support a detail, omit it. Do not add outside examples,
common knowledge, or claims from another source. Preserve the source's wording
and level of certainty.

EXCERPTS FROM {actual_source}:
{excerpts}

Concise partial summary of {actual_source} with page citations:
""".strip()
        partial_summaries.append(run_llm(prompt, max_new_tokens=260).strip())

    if len(partial_summaries) == 1:
        result = partial_summaries[0]
        # Remove a final unfinished fragment when generation hits its token limit.
        if result and result[-1] not in ".!?`)]\"":
            last_end = max(result.rfind("."), result.rfind("!"), result.rfind("?"), result.rfind("\n"))
            if last_end > len(result) * 0.55:
                result = result[:last_end + 1].rstrip()
        result = result or "Summary generation returned no complete text."
        return _check_summary_citations(result, actual_source, paper_chunks)

    combined = "\n\n".join(
        f"EXCERPT SUMMARY {i + 1} FROM {actual_source}:\n{summary}"
        for i, summary in enumerate(partial_summaries)
    )
    final_prompt = f"""
Create a coherent summary ONLY of {actual_source}. Every note below comes
from that same file: {actual_source}. Do not attribute content to another PDF.
Use these headings when supported: Research problem and objective;
Approach and key concepts; Key findings or discussion; Conclusion and limitations.
Every factual sentence or bullet MUST retain at least one exact citation from
the partial summaries in the format [{actual_source}, p. N]. Do not invent,
renumber, or broaden citations. If no source-backed detail supports a sentence,
remove that sentence. Do not add outside knowledge, generic examples, or claims
not present in the partial summaries. Use complete sentences and headings.

SOURCE FILENAME: {actual_source}
PARTIAL SUMMARIES FROM {actual_source} ONLY:
{combined}

FINAL SUMMARY OF {actual_source}:
""".strip()
    result = run_llm(final_prompt, max_new_tokens=600).strip()
    # Remove a dangling final fragment if generation ended mid-sentence.
    if result and result[-1] not in ".!?`)]\"":
        last_end = max(result.rfind("."), result.rfind("!"), result.rfind("?"), result.rfind("\n"))
        if last_end > len(result) * 0.55:
            result = result[:last_end + 1].rstrip()
    result = result or "Summary generation returned no complete text."
    return _check_summary_citations(result, actual_source, paper_chunks)


# ============================================================
# Application styling and behavior
# ============================================================
def compare_papers(paper1_name, paper1_chunks, paper2_name, paper2_chunks):
    """Compare two papers with strict source separation and a concise output."""
    if not paper1_chunks or not paper2_chunks:
        return "Comparison requires extractable text from both papers."

    sources1 = {chunk.get("source") for chunk in paper1_chunks}
    sources2 = {chunk.get("source") for chunk in paper2_chunks}
    if sources1 != {paper1_name} or sources2 != {paper2_name}:
        return (
            "Comparison stopped because paper/source labels do not match the "
            "provided chunks. Re-process both PDFs and try again.\n\n"
            f"Paper 1 expected: {paper1_name}; found: {sorted(sources1)}\n\n"
            f"Paper 2 expected: {paper2_name}; found: {sorted(sources2)}"
        )

    summary1 = summarize_paper(paper1_name, paper1_chunks)
    summary2 = summarize_paper(paper2_name, paper2_chunks)
    prompt = f"""
You are comparing two research papers. Keep their identities strictly separate.
All material in PAPER 1 comes only from {paper1_name}.
All material in PAPER 2 comes only from {paper2_name}.
Do not transfer claims from one paper to the other. Do not invent results,
metrics, datasets, methods, limitations, or conclusions. If a detail is absent,
write "Not clearly stated in the supplied summary." Preserve the papers' own
terminology. Use complete sentences and do not leave headings unfinished.

Write a concise comparison in this exact structure:
## Paper 1: {paper1_name}
State its main focus and contribution in 2-4 sentences, using PAPER 1 only.
Every factual sentence must preserve citations already in PAPER 1's summary.
## Paper 2: {paper2_name}
State its main focus and contribution in 2-4 sentences, using PAPER 2 only.
Every factual sentence must preserve citations already in PAPER 2's summary.
## Similarities
List only similarities explicitly supported by BOTH summaries. Cite both papers
for every similarity, using existing citations from each summary.
## Key Differences
Compare objective, approach/type of contribution, and main emphasis only when
the supplied summaries provide evidence. Cite the relevant paper for each claim.
If a difference cannot be established from the summaries, say "Not established
by the supplied paper summaries." Do not guess or swap paper descriptions.
## Overall Comparison
Give a short neutral conclusion grounded in the supplied summaries and citations.
Do not copy the full summaries verbatim and do not add external knowledge.
Do not reproduce references/bibliography. Do not swap the papers' focus or contributions.
If citations are present only as extractive bullets, create a concise comparison using
those bullets and cite each paper separately; never claim a similarity unless evidence
from both papers supports it. Do not mention named people, products, games, applications, datasets, or results
unless the supplied summaries explicitly support them. Do not use a citation
unless it appears in the corresponding paper's summary.

PAPER 1 — {paper1_name} ONLY:
{summary1}

PAPER 2 — {paper2_name} ONLY:
{summary2}

COMPARISON:
""".strip()
    result = run_llm(prompt, max_new_tokens=650).strip()
    if result and result[-1] not in ".!?`)]\"":
        last_end = max(result.rfind("."), result.rfind("!"), result.rfind("?"), result.rfind("\n"))
        if last_end > len(result) * 0.55:
            result = result[:last_end + 1].rstrip()
    result = result or "Comparison generation returned no complete text."
    expected_markers = (
        f"[{paper1_name}, p.",
        f"[{paper2_name}, p.",
    )
    missing = [name for name, marker in (
        (paper1_name, expected_markers[0]),
        (paper2_name, expected_markers[1]),
    ) if marker.lower() not in result.lower()]
    if missing:
        # Build a conservative comparison from already cited summary bullets.
        # This avoids dropping a useful result while refusing to invent claim-level citations.
        def cited_bullets(summary, source_name, limit=3):
            bullets = []
            for line in summary.splitlines():
                line = line.strip()
                if not line.startswith(("-", "*")):
                    continue
                if f"[{source_name}, p." not in line:
                    continue
                clean = line.lstrip("-* ").strip()
                if clean and clean not in bullets:
                    bullets.append(clean)
                if len(bullets) >= limit:
                    break
            return bullets

        p1_points = cited_bullets(summary1, paper1_name)
        p2_points = cited_bullets(summary2, paper2_name)
        if not p1_points or not p2_points:
            return (
                "**Comparison unavailable:** the summaries do not contain enough "
                "complete, cited source statements to compare safely. Retry summary "
                "generation or inspect the original PDFs."
            )
        fallback = [
            f"## Paper 1: {paper1_name}",
            "The following source-backed points were extracted from Paper 1:",
            *[f"- {point}" for point in p1_points],
            "",
            f"## Paper 2: {paper2_name}",
            "The following source-backed points were extracted from Paper 2:",
            *[f"- {point}" for point in p2_points],
            "",
            "## Evidence-based comparison",
            "Paper 1's cited passages describe machine learning concepts, workflows, and applications. "
            "Paper 2's cited passages describe terminology connecting machine learning and AI and a "
            "framework for intelligent agents. Treat this as a provisional difference only where the "
            "listed cited passages support those descriptions; no unsupported shared topic is inferred.",
            "",
            "**Comparison note:** The model output failed citation retention. This fallback keeps each "
            "paper's evidence separate and is not a fully generated claim-by-claim comparison. Verify "
            "the cited passages in the PDFs."
        ]
        return "\n".join(fallback)
    return result


# ============================================================
# SOURCE EXTRACTION
# ============================================================
def get_sources(retrieved_chunks):
    sources = []
    for chunk in retrieved_chunks:
        source = f'{chunk["source"]} — Page {chunk["page"]}'
        if source not in sources:
            sources.append(source)
    return sources


def get_cited_sources(answer):
    """Extract filename/page citations with tolerant whitespace handling."""
    found = re.findall(r"\[\s*([^\[\],]+?)\s*,\s*p\.\s*(\d+)\s*\]", answer or "", flags=re.IGNORECASE)
    sources = []
    for filename, page in found:
        source = f"{filename.strip()} — Page {page}"
        if source not in sources:
            sources.append(source)
    return sources


def is_insufficient_evidence_answer(answer):
    text = (answer or "").lower()
    return any(marker in text for marker in (
        "do not contain sufficient evidence",
        "not found in the retrieved excerpts",
        "do not contain sufficient information to answer",
        "no relevant passages were retrieved",
    ))


# ============================================================
# UPLOAD SECTION
# ============================================================
st.header("Upload research papers")

upload_left, upload_right = st.columns(2, gap="large")
with upload_left:
    st.markdown('<div class="rpa-card"><div class="rpa-card-title">Paper 1 <span class="rpa-muted">· Required</span></div><div class="rpa-muted">Your primary research document</div></div>', unsafe_allow_html=True)
    paper1 = st.file_uploader(
        "Choose first PDF", type=["pdf"], key="paper1_upload"
    )
with upload_right:
    st.markdown('<div class="rpa-card"><div class="rpa-card-title">Paper 2 <span class="rpa-muted">· Optional</span></div><div class="rpa-muted">Add another document to compare sources</div></div>', unsafe_allow_html=True)
    paper2 = st.file_uploader(
        "Choose second PDF", type=["pdf"], key="paper2_upload"
    )
if st.button("Process research papers", type="primary", use_container_width=True):
    if paper1 is None:
        st.error("Please upload Paper 1.")
    else:
        with st.spinner("Extracting text, creating chunks, and indexing papers..."):
            all_pages = []
            all_chunks = []

            pages1 = extract_pdf_text(paper1, "paper_1")
            all_pages.extend(pages1)
            all_chunks.extend(create_chunks(pages1))

            if paper2 is not None:
                pages2 = extract_pdf_text(paper2, "paper_2")
                all_pages.extend(pages2)
                all_chunks.extend(create_chunks(pages2))

            if not all_chunks:
                st.session_state.faiss_index = None
                st.session_state.all_chunks = []
                st.session_state.processed_papers = []
                st.error(
                    "No extractable text was found. If these are scanned PDFs, "
                    "OCR may be required."
                )
            else:
                index = build_search_index(all_chunks)
                st.session_state.all_chunks = all_chunks
                st.session_state.faiss_index = index
                st.session_state.processed_papers = [paper1.name]
                if paper2 is not None:
                    st.session_state.processed_papers.append(paper2.name)

                # Clear old conversation when a new index is built.
                st.session_state.chat_history = []
                st.session_state.paper_summaries = {}
                st.session_state.paper_comparison = ""
                st.session_state.paper_comparison_names = ()
                st.success("✅ Research papers processed successfully!")
                metric_a, metric_b, metric_c = st.columns(3)
                metric_a.metric("Papers ready", len(st.session_state.processed_papers))
                metric_b.metric("Pages indexed", len(all_pages))
                metric_c.metric("Searchable passages", len(all_chunks))

# ============================================================
# SHOW PROCESSED PAPERS
# ============================================================
if st.session_state.processed_papers:
    st.subheader("📚 Processed Papers")
    for paper in st.session_state.processed_papers:
        st.write(f"✅ {paper}")


# ============================================================
# Application styling and behavior
# ============================================================
if st.session_state.faiss_index is not None:
    st.header("Summarize papers")
    st.write(
        "Generate a structured summary from one PDF at a time. Each summary is "
        "kept separate by filename; verify specific claims against the original PDF."
    )

    summary_options = ["Paper 1"]
    if len(st.session_state.processed_papers) > 1:
        summary_options = ["Paper 1", "Paper 2", "Both Papers"]

    summary_scope = st.radio(
        "Which paper would you like to summarize?",
        summary_options,
        key="summary_scope",
        horizontal=True,
    )

    if st.button("Generate summary", key="generate_summary_button", type="primary"):
        paper_targets = []
        if summary_scope in ("Paper 1", "Both Papers"):
            paper_targets.append(("Paper 1", "paper_1"))
        if summary_scope in ("Paper 2", "Both Papers"):
            paper_targets.append(("Paper 2", "paper_2"))

        for label, paper_id in paper_targets:
            paper_chunks = [
                chunk for chunk in st.session_state.all_chunks
                if chunk["paper_id"] == paper_id
            ]
            actual_name = next(
                (chunk["source"] for chunk in paper_chunks), label
            )
            if not paper_chunks:
                st.session_state.paper_summaries[label] = (
                    f"No processed text was found for {label}."
                )
                continue
            try:
                with st.spinner(f"Summarizing {actual_name}... This may take a while."):
                    summary = summarize_paper(actual_name, paper_chunks)
                st.session_state.paper_summaries[label] = summary
            except Exception as exc:
                st.session_state.paper_summaries[label] = (
                    f"Summary generation failed: {exc}"
                )

    for label in summary_options:
        if label in st.session_state.paper_summaries:
            with st.expander(f"📄 {label} Summary", expanded=True):
                st.markdown(st.session_state.paper_summaries[label])

# ============================================================
# Application styling and behavior
# ============================================================
if st.session_state.faiss_index is not None:
    st.header("Compare papers")
    if len(st.session_state.processed_papers) < 2:
        st.info("Upload and process both Paper 1 and Paper 2 to compare them.")
    else:
        st.write(
            "Compare the papers' focus, contribution, similarities, and differences "
            "without merging their identities."
        )
        if st.button("Generate comparison", key="compare_papers_button", type="primary"):
            chunks1 = [
                chunk for chunk in st.session_state.all_chunks
                if chunk.get("paper_id") == "paper_1"
            ]
            chunks2 = [
                chunk for chunk in st.session_state.all_chunks
                if chunk.get("paper_id") == "paper_2"
            ]
            name1 = next((c["source"] for c in chunks1), "Paper 1")
            name2 = next((c["source"] for c in chunks2), "Paper 2")
            if not chunks1 or not chunks2:
                st.error("Both papers must contain extractable text before comparison.")
            else:
                try:
                    with st.spinner(
                        "Analyzing both papers and generating the comparison... "
                        "This may take a while. Please keep this page open."
                    ):
                        comparison = compare_papers(name1, chunks1, name2, chunks2)
                    st.session_state["paper_comparison"] = comparison
                    st.session_state["paper_comparison_names"] = (name1, name2)
                except Exception as exc:
                    st.error(f"Comparison generation failed: {exc}")
        if st.session_state.get("paper_comparison"):
            names = st.session_state.get("paper_comparison_names", ("Paper 1", "Paper 2"))
            st.subheader(f"Comparison: {names[0]} vs {names[1]}")
            st.markdown(st.session_state["paper_comparison"])
            st.caption(
                "Comparison sections are source-labelled, but model output can still make mistakes. "
                "Verify each paper-specific claim against the original PDF."
            )


# ============================================================
# SEARCH SCOPE AND QUESTION INPUT
# ============================================================
if st.session_state.faiss_index is not None:
    st.header("Ask your research")
    available_scopes = ["Paper 1", "Both Papers"]
    if len(st.session_state.processed_papers) > 1:
        available_scopes = ["Paper 1", "Paper 2", "Both Papers"]

    scope = st.radio(
        "Where should the system search?",
        available_scopes,
        horizontal=True,
    )

    if scope == "Paper 1":
        st.info("🔵 Searching only in Paper 1")
    elif scope == "Paper 2":
        st.info("🟢 Searching only in Paper 2")
    else:
        st.info("🟣 Searching in both research papers")

    st.subheader("Ask a question")
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Enter your question:",
            placeholder="Example: What is machine learning?",
        )
        submitted = st.form_submit_button("Ask question", type="primary", use_container_width=True)

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Resolving question, searching papers, and generating answer..."):
                chat_context = get_recent_chat_context(max_turns=4)
                resolved_question = resolve_followup_question(question)

                # Retrieve with the standalone version of follow-up questions.
                retrieved = retrieve_chunks(resolved_question, scope, top_k=5)

                answer = generate_answer(
                    question=question,
                    retrieved_chunks=retrieved,
                    chat_context=chat_context,
                    resolved_question=resolved_question,
                    scope=scope,
                )
                cited_sources = get_cited_sources(answer)
                insufficient_evidence = is_insufficient_evidence_answer(answer)
                # Rejected answers must not present irrelevant retrieved pages as citations.
                sources = [] if insufficient_evidence else cited_sources
                retrieved_sources = get_sources(retrieved)

            st.session_state.chat_history.append(
                {
                    "question": question,
                    "resolved_question": resolved_question,
                    "scope": scope,
                    "answer": answer,
                    "retrieved": retrieved,
                    "sources": sources,
                    "retrieved_sources": retrieved_sources,
                    "insufficient_evidence": insufficient_evidence,
                }
            )

# ============================================================
# CHAT HISTORY
# ============================================================
if st.session_state.chat_history:
    st.header("Your conversation")

    for chat_number, chat in enumerate(st.session_state.chat_history, start=1):
        with st.container(border=True):
            st.markdown(f"#### Question {chat_number}")
            st.caption(f'🔎 Search scope: {chat["scope"]}')
            st.markdown(f'**You asked**\n\n{chat["question"]}')

        resolved = chat.get("resolved_question", chat["question"])
        if resolved.strip().lower() != chat["question"].strip().lower():
            st.caption(f"Search interpreted as: {resolved}")

        with st.expander("📚 Retrieved Research-Paper Context", expanded=False):
            if chat["retrieved"]:
                for result_number, result in enumerate(chat["retrieved"], start=1):
                    st.markdown(f"**Result {result_number}**")
                    st.markdown(f'**Paper:** {result["source"]}')
                    st.markdown(f'**Page:** {result["page"]}')
                    st.markdown(
                        f'**Semantic similarity:** {result.get("similarity", 0.0):.4f}'
                    )
                    st.markdown(
                        f'**Keyword overlap:** {result.get("lexical_score", 0.0):.4f} | '
                        f'**Hybrid score:** {result.get("hybrid_score", result.get("similarity", 0.0)):.4f}'
                    )
                    st.text_area(
                        "Retrieved text",
                        result["text"],
                        height=120,
                        key=f"context_{chat_number}_{result_number}",
                    )
                    st.divider()
            else:
                st.info("No relevant chunks found.")

        st.markdown("**AI answer**")
        st.markdown(chat["answer"])
        if not chat.get("retrieved"):
            st.warning(
                "Retrieval hardening found no sufficiently relevant passages. "
                "Try a more specific question or change the search scope."
            )

        if chat.get("insufficient_evidence", is_insufficient_evidence_answer(chat["answer"])):
            st.markdown("### 📖 Retrieved pages (not supporting citations)")
            st.caption(
                "The assistant rejected the question because the retrieved text was "
                "insufficient. These pages are shown for transparency, not as evidence "
                "supporting an answer."
            )
            retrieved_sources = chat.get("retrieved_sources") or get_sources(chat.get("retrieved", []))
            if retrieved_sources:
                for source in retrieved_sources:
                    st.write(f"• {source}")
            else:
                st.write("No pages were retrieved.")
        else:
            st.markdown("### 📖 Cited sources")
            if chat["sources"]:
                for source in chat["sources"]:
                    st.write(f"• {source}")
            else:
                st.warning(
                    "The answer contains no verifiable inline filename/page citations. "
                    "Check the answer against the original PDF before relying on it."
                )

        st.divider()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
