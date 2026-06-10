"""
LangGraph-based medical GraphRAG agent.

Graph topology:
    [START] → agent → (tool_calls?) → tools → agent → ... → [END]

The agent node calls the LLM with bound tools.
If the LLM requests tool calls, the tools node executes them and
feeds results back to the agent.  Otherwise the loop terminates.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.agent.prompts import VCLINIC_AGENT_SYSTEM_PROMPT
from src.agent.tools import ALL_TOOLS
from src.config import settings


# ── Agent state ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── LLM with tools bound ──────────────────────────────────────────────────────

def _build_llm():
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )
    return llm.bind_tools(ALL_TOOLS)


# ── Node functions ────────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> AgentState:
    """Call the LLM, prepending the system prompt on the first turn."""
    llm_with_tools = _build_llm()
    messages = state["messages"]

    # Inject system prompt if not already present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=VCLINIC_AGENT_SYSTEM_PROMPT)] + list(messages)

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Route to tools if the last message has tool calls, else end."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_agent():
    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def ask(question: str) -> str:
    """
    Ask a medical question and get a GraphRAG-powered answer.
    Returns the final assistant message content.
    """
    from langchain_core.messages import HumanMessage

    agent = get_agent()
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content
