# LangChain + LangGraph Agent Template

Reusable scaffolding for building tool-using agents with LangChain and LangGraph.

## Components
- `agent_template/state.py`: typed conversation state shared across graph nodes.
- `agent_template/graph.py`: `AgentBuilder` that wires an LLM with tools in a LangGraph workflow.
- `agent_template/tools.py`: example `BaseToolset` interface plus basic math tools.
- `examples/basic_agent.py`: runnable sample showing how to assemble and stream from the agent.

## Usage
1. Install dependencies (e.g. `langchain`, `langgraph`, `langchain-openai`).
2. Provide your preferred `BaseChatModel` instance (the example uses `ChatOpenAI`).
3. Create or import toolsets and pass them to `AgentBuilder`.
4. Compile the graph and stream results:

```python
from langchain_openai import ChatOpenAI
from agent_template import AgentBuilder, create_default_tools

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
builder = AgentBuilder(llm=llm, tools=create_default_tools())
app = builder.build()

for event in app.stream({"messages": ["What's 2 + 2?"]}, stream_mode="values"):
    print(event["messages"][-1])
```

Swap in your own tools or prompts to adapt the template to other domains.
