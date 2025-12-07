"""Minimal example of wiring the reusable agent graph."""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from agent_template import AgentBuilder, AgentState, create_default_tools


def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = list(create_default_tools())

    builder = AgentBuilder(llm=llm, tools=tools)
    graph = builder.build()

    memory = MemorySaver()
    app = graph.with_config(checkpointer=memory)

    events = app.stream({"messages": ["What's 12 + 30, then multiply by 2?"]}, stream_mode="values")
    for event in events:
        for message in event["messages"]:
            print(message)


if __name__ == "__main__":
    main()
