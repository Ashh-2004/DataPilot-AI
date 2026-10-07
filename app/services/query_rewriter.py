"""Conversation follow-up rewriter.

Resolves references like "what about last month?" by rewriting the follow-up
into a standalone question using the previous turn before it hits the Planner.
"""

import logging
from typing import Any

from langchain_ollama import ChatOllama

LOGGER = logging.getLogger(__name__)


def rewrite_followup(
    question: str,
    history: list[dict[str, str]],
    llm: ChatOllama,
) -> str:
    """Rewrite a follow-up question into a standalone question.

    If the question appears to be self-contained, returns it unchanged.
    If it references prior conversation context, rewrites it to be standalone.

    Parameters
    ----------
    question : str
        The user's current question.
    history : list[dict[str, str]]
        Recent conversation history (role/content dicts).
    llm : ChatOllama
        The language model to use for rewriting.

    Returns
    -------
    str
        The standalone version of the question.
    """
    if not history:
        return question

    # Quick heuristic: skip rewriting if the question looks self-contained
    if _looks_standalone(question):
        return question

    recent = history[-4:]  # Last 2 turns (user + assistant)
    history_text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in recent)

    prompt = (
        "You are a query rewriter. Given the conversation history and the user's follow-up "
        "question, rewrite the follow-up into a single standalone question that can be "
        "understood without the conversation history. Keep the question concise and specific. "
        "Do NOT answer the question, just rewrite it.\n\n"
        "If the follow-up is already a complete, standalone question, return it unchanged.\n\n"
        f"Conversation history:\n{history_text}\n\n"
        f"Follow-up question: {question}\n\n"
        "Standalone question:"
    )

    try:
        response = llm.invoke(prompt)
        rewritten = response.content if isinstance(response.content, str) else str(response.content)
        rewritten = rewritten.strip().strip('"').strip("'")
        if rewritten and len(rewritten) > 5:
            LOGGER.info("Rewrote follow-up: %r -> %r", question, rewritten)
            return rewritten
    except Exception:
        LOGGER.warning("Follow-up rewriting failed, using original question", exc_info=True)

    return question


_REFERENCE_WORDS = {
    "that", "those", "these", "it", "them", "this",
    "the same", "the previous", "the last", "above",
    "instead", "also", "too", "as well",
    "what about", "how about", "and for", "but for",
    "last month", "last quarter", "last year", "this month",
    "compared to", "versus", "vs",
}


def _looks_standalone(question: str) -> bool:
    """Heuristic: does the question reference prior conversation context?"""
    lower = question.lower().strip()
    # Short questions are likely follow-ups
    if len(lower.split()) <= 3:
        return False
    # Check for reference words
    for ref in _REFERENCE_WORDS:
        if ref in lower:
            return False
    return True

