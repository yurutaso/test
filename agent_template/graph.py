from __future__ import annotations

from collections.abc import Callable
from typing import Any, List, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from .state import AgentState


class AgentBuilder:
    """Factory for assembling LangGraph-powered agents.

    This class keeps the graph construction logic in one place so it can be
    reused across scripts, notebooks, or production services.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: Sequence[Callable[..., Any]] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tuple(tools or [])
        self.system_prompt = system_prompt or "You are a helpful AI assistant."

    def _call_model(self, state: AgentState, config: RunnableConfig) -> AgentState:
        bound_model = self.llm.bind_tools(self.tools)
        messages = list(state["messages"])
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            messages.insert(0, SystemMessage(content=self.system_prompt))

        response = bound_model.invoke(messages, config)
        return {"messages": [response]}

    def _call_tools(self, state: AgentState) -> AgentState:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return {"messages": []}

        tool_messages: List[ToolMessage] = []
        for tool_call in last_message.tool_calls or []:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool = self._resolve_tool(tool_name)
            result = tool(**tool_args)
            tool_messages.append(
                ToolMessage(tool_call_id=tool_call["id"], name=tool_name, content=str(result))
            )
        return {"messages": tool_messages}

    def _resolve_tool(self, tool_name: str) -> Callable[..., Any]:
        for tool in self.tools:
            if getattr(tool, "name", tool.__name__) == tool_name:
                return tool
        raise KeyError(f"Unknown tool '{tool_name}'")

    def build(self) -> StateGraph[AgentState]:
        graph = StateGraph(AgentState)

        graph.add_node("model", self._call_model)
        graph.add_node("tools", self._call_tools)

        graph.add_edge(START, "model")
        graph.add_conditional_edges(
            "model",
            self._route_tools,
            path_map={"tools": "tools", END: END},
        )
        graph.add_edge("tools", "model")

        return graph.compile()

    def _route_tools(self, state: AgentState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END


__all__ = ["AgentBuilder"]
