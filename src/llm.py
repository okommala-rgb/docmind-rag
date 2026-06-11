"""
LLM integration using Google Gemini via LangChain.
Supports streaming, context-aware prompting, citation generation,
and hallucination-reduction techniques.
"""

from __future__ import annotations

import os
from typing import Generator, Iterator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.retriever import RetrievalResult
from src.utils import config, logger, truncate


# ── System prompt ─────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are a precise, helpful research assistant with access to a curated document collection.

INSTRUCTIONS:
1. Answer the user's question using ONLY the provided context passages.
2. For every factual claim, cite the source using [Source: filename, Page: X] format.
3. If the context does not contain enough information to answer confidently, say so explicitly — do NOT speculate or invent facts.
4. Structure longer answers with clear sections and bullet points where appropriate.
5. At the end of your answer, list all cited sources in a "References" block.
6. Be concise: avoid padding or repetition.

CITATION FORMAT:
- Inline: "The system processes data in real time [Source: report.pdf, Page: 3]."
- References block:
  **References**
  1. report.pdf, Page 3
  2. manual.docx, Section 2
"""

SUMMARY_SYSTEM_PROMPT = """You are a professional summariser. Produce a structured, accurate summary of the provided text.
Include: key topics, main findings, important details, and any conclusions.
Do not add information not present in the text."""

QUIZ_SYSTEM_PROMPT = """Generate {n} multiple-choice quiz questions based on the provided document content.
Format each question as:
Q[N]: <question>
A) <option>  B) <option>  C) <option>  D) <option>
Answer: <correct letter>
Explanation: <brief explanation>"""

INSIGHT_SYSTEM_PROMPT = """Analyse the following document content and provide:
1. **Main Theme** – What is this document primarily about?
2. **Key Entities** – Important people, organisations, systems, or concepts mentioned.
3. **Key Claims** – The 3–5 most important assertions made.
4. **Gaps & Caveats** – What the document does NOT address or acknowledges as uncertain.
5. **Suggested Follow-up Questions** – 3 questions a reader might want answered next."""

COMPARE_SYSTEM_PROMPT = """Compare the two documents provided below. Structure your comparison as:
1. **Common Themes** – What both documents agree on or share.
2. **Key Differences** – How they differ in content, scope, or conclusions.
3. **Unique Contributions** – What each document adds that the other does not.
4. **Overall Assessment** – Which is more comprehensive and why."""


# ── LLM factory ──────────────────────────────────────────────────────────────

def get_llm(
    model_name: str | None = None,
    temperature: float | None = None,
    streaming: bool = True,
) -> ChatGoogleGenerativeAI:
    api_key = config.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Add it to your .env file or environment variables."
        )
    return ChatGoogleGenerativeAI(
        model=model_name or config.DEFAULT_LLM_MODEL,
        google_api_key=api_key,
        temperature=temperature if temperature is not None else config.TEMPERATURE,
        streaming=streaming,
        max_output_tokens=config.MAX_TOKENS,
        convert_system_message_to_human=True,
    )


# ── Context builder ───────────────────────────────────────────────────────────

def _build_context(results: list[RetrievalResult], max_chars: int = 12_000) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    parts: list[str] = []
    total_chars = 0
    for i, result in enumerate(results, 1):
        doc = result.document
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        section = doc.metadata.get("section_heading", "")
        header = f"[{i}] Source: {source}, Page: {page}"
        if section:
            header += f", Section: {section}"
        header += f" (relevance: {result.score:.2f})"
        chunk_text = f"{header}\n{doc.page_content.strip()}"
        if total_chars + len(chunk_text) > max_chars:
            break
        parts.append(chunk_text)
        total_chars += len(chunk_text)
    return "\n\n---\n\n".join(parts)


# ── Main RAG answer function ──────────────────────────────────────────────────

def stream_rag_answer(
    query: str,
    results: list[RetrievalResult],
    conversation_history: list[dict] | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
) -> Iterator[str]:
    """
    Stream a RAG answer token-by-token.
    Yields string chunks as they arrive from the Gemini API.
    """
    llm = get_llm(model_name=model_name, temperature=temperature, streaming=True)
    context = _build_context(results)

    messages: list = [SystemMessage(content=RAG_SYSTEM_PROMPT)]

    # Add conversation history (last 6 turns to keep context window manageable)
    if conversation_history:
        for turn in conversation_history[-6:]:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))

    user_message = (
        f"CONTEXT PASSAGES:\n\n{context}\n\n"
        f"---\n\nQUESTION: {query}"
    )
    messages.append(HumanMessage(content=user_message))

    try:
        for chunk in llm.stream(messages):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content
    except Exception as exc:
        logger.error(f"LLM streaming error: {exc}")
        yield f"\n\n⚠️ Error generating response: {exc}"


def get_rag_answer(
    query: str,
    results: list[RetrievalResult],
    conversation_history: list[dict] | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
) -> str:
    """Non-streaming version — returns complete answer string."""
    return "".join(
        stream_rag_answer(query, results, conversation_history, model_name, temperature)
    )


# ── Utility LLM tasks ─────────────────────────────────────────────────────────

def summarise_text(
    text: str,
    model_name: str | None = None,
    streaming: bool = False,
) -> str | Iterator[str]:
    """Summarise arbitrary text."""
    llm = get_llm(model_name=model_name, streaming=streaming)
    messages = [
        SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
        HumanMessage(content=f"Please summarise the following:\n\n{text[:15000]}"),
    ]
    if streaming:
        def _stream():
            for chunk in llm.stream(messages):
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
        return _stream()
    response = llm.invoke(messages)
    return response.content


def generate_quiz(text: str, n: int = 5, model_name: str | None = None) -> str:
    """Generate n MCQ quiz questions from text."""
    llm = get_llm(model_name=model_name, streaming=False)
    prompt = QUIZ_SYSTEM_PROMPT.format(n=n)
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=text[:12000]),
    ]
    response = llm.invoke(messages)
    return response.content


def generate_insights(text: str, model_name: str | None = None) -> str:
    """Generate structured document insights."""
    llm = get_llm(model_name=model_name, streaming=False)
    messages = [
        SystemMessage(content=INSIGHT_SYSTEM_PROMPT),
        HumanMessage(content=text[:12000]),
    ]
    response = llm.invoke(messages)
    return response.content


def compare_documents(
    text_a: str,
    name_a: str,
    text_b: str,
    name_b: str,
    model_name: str | None = None,
) -> str:
    """Compare two document texts."""
    llm = get_llm(model_name=model_name, streaming=False)
    combined = (
        f"=== DOCUMENT A: {name_a} ===\n{text_a[:6000]}\n\n"
        f"=== DOCUMENT B: {name_b} ===\n{text_b[:6000]}"
    )
    messages = [
        SystemMessage(content=COMPARE_SYSTEM_PROMPT),
        HumanMessage(content=combined),
    ]
    response = llm.invoke(messages)
    return response.content
