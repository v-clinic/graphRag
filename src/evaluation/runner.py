"""
Runs the VClinic LangGraph agent and captures tool-call outputs (contexts)
alongside the final answer for RAGAS evaluation.

Usage
-----
    from src.evaluation.runner import run_with_context

    result = run_with_context("Which patients have type 2 diabetes?")
    print(result.answer)
    print(result.contexts)   # list of raw JSON strings returned by tools
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from src.agent.graph_analysis_agent import get_agent


@dataclass
class AgentRunResult:
    """Container for a single agent execution."""

    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    """Raw tool-call payloads (JSON strings) — used as RAG contexts for RAGAS."""
    tool_names: list[str] = field(default_factory=list)
    """Names of every tool called during the run, in invocation order."""
    error: Optional[str] = None


def run_with_context(question: str) -> AgentRunResult:
    """
    Invoke the LangGraph agent for *question*, then parse the resulting
    message list to extract:

    - Every ToolMessage payload as a retrieval context string.
    - The names of all tools called (from AIMessage.tool_calls).
    - The final AIMessage content as the answer.

    Any exception is caught and returned as ``AgentRunResult.error`` so the
    caller can continue evaluating other samples.
    """
    try:
        agent = get_agent()
        state = agent.invoke({"messages": [HumanMessage(content=question)]})
        messages: list[BaseMessage] = state["messages"]

        contexts: list[str] = []
        tool_names: list[str] = []

        for msg in messages:
            # Collect retrieval contexts from tool responses.
            if isinstance(msg, ToolMessage):
                raw = msg.content if isinstance(msg.content, str) else str(msg.content)
                # Skip empty / null results — they add no signal to RAGAS.
                if raw and raw.strip() not in ("", "No results found.", "null", "[]"):
                    contexts.append(raw)

            # Track which tools were called.
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "unknown")
                    tool_names.append(name or "unknown")

        final_msg = messages[-1] if messages else None
        answer = (
            final_msg.content
            if final_msg is not None and isinstance(final_msg.content, str)
            else str(final_msg.content) if final_msg is not None else ""
        )

        return AgentRunResult(
            question=question,
            answer=answer,
            contexts=contexts,
            tool_names=tool_names,
        )

    except Exception as exc:  # noqa: BLE001
        return AgentRunResult(question=question, answer="", error=str(exc))
