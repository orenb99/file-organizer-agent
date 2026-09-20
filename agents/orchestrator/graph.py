from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from config import MODEL_NAME
from agents.orchestrator.prompts import SYSTEM_PROMPT
from agents.utils.state import AgentState
from agents.orchestrator.tools import TOOLS

model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0).bind_tools(TOOLS)


def agent_node(state: AgentState, config: RunnableConfig) -> dict:
    # custom_instruction is optional and set once per run in main.py (either
    # via --instruction or the boot-time prompt), then carried through
    # config the same way src_root/dst_root are — never stored in state, so
    # it doesn't bloat every message in the history.
    custom_instruction = config.get("configurable", {}).get("custom_instruction")
    system_prompt = SYSTEM_PROMPT
    if custom_instruction:
        system_prompt += f"\n\nAdditional instructions for this run:\n{custom_instruction}"

    messages = [SystemMessage(system_prompt)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def build_graph(verbose: bool = False):
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "agent")
    # tools_condition inspects the last AIMessage: routes to "tools" if it
    # has tool_calls, else to END. This is what lets the agent decide for
    # itself when the batch is finished — no explicit "done" tool needed.
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    graph = builder.compile()
    if verbose:
        print(graph.get_graph().draw_ascii())
    return graph