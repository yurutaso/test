from __future__ import annotations

from typing import Annotated, List
from langgraph.graph.message import AnyMessage
from langgraph.graph.state import MessagesState


class AgentState(MessagesState):
    """State object for the agent graph.

    The ``messages`` field handles conversational context; LangGraph merges
    messages automatically between nodes when the graph branches.
    """

    messages: Annotated[List[AnyMessage], MessagesState.MessagesField(add=True)]


__all__ = ["AgentState"]
