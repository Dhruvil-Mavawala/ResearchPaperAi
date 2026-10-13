import faiss
import numpy as np
import torch
import streamlit as st

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# MODEL NAMES
# ============================================================

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


# ============================================================
# BGE MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(model, chunks):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    if not texts:
        return np.empty(
            (0, 384),
            dtype=np.float32
        )

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings.astype(
        np.float32
    )


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_faiss_index(embeddings):

    if len(embeddings) == 0:
        return None

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


# ============================================================
# CREATE QUERY EMBEDDING
# ============================================================

def create_query_embedding(model, query):

    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embedding.astype(
        np.float32
    )


# ============================================================
# SEARCH FAISS
# ============================================================

def search_faiss(
    index,
    query_embedding,
    chunks,
    top_k=5
):

    if index is None:
        return []

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    if query_embedding.ndim == 1:

        query_embedding = query_embedding.reshape(
            1,
            -1
        )

    # Make sure k does not exceed number of vectors
    actual_k = min(
        top_k,
        index.ntotal
    )

    scores, indices = index.search(
        query_embedding,
        actual_k
    )

    results = []

    for score, index_number in zip(
        scores[0],
        indices[0]
    ):

        if index_number < 0:
            continue

        chunk = chunks[
            index_number
        ].copy()

        chunk["similarity"] = float(
            score
        )

        results.append(
            chunk
        )

    return results


# ============================================================
# LOAD QWEN LLM
# ============================================================

@st.cache_resource
def load_llm():

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_NAME
    )

    if torch.cuda.is_available():

        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    else:

        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.float32
        )

    return tokenizer, model


# ============================================================
# BUILD RAG CONTEXT
# ============================================================

def build_rag_context(
    retrieved_chunks
):

    if not retrieved_chunks:
        return ""

    context_parts = []

    for number, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {number}

Paper ID: {chunk['paper_id']}
File: {chunk['source']}
Page: {chunk['page']}

TEXT:
{chunk['text']}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# GENERATE RAG ANSWER
# ============================================================

def generate_rag_answer(
    tokenizer,
    model,
    question,
    retrieved_chunks
):

    if not retrieved_chunks:

        return (
            "I could not find this information "
            "in the selected uploaded paper(s)."
        )

    context = build_rag_context(
        retrieved_chunks
    )

    prompt = f"""
You are a research paper question-answering assistant.

Answer the user's question using ONLY the information
provided in the research-paper context below.

Do not use outside knowledge.

If the answer cannot be found in the context, say:

"I could not find this information in the selected uploaded paper(s)."

Keep the answer clear, concise, and academically accurate.

Research-paper context:
-----------------------
{context}
-----------------------

User question:
{question}

Answer:
"""

    messages = [
        {
            "role": "system",
            "content": (
                "Answer questions using only the "
                "provided research-paper context."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    device = next(
        model.parameters()
    ).device

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = output[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return answer