from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph import graph
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from config import MODEL_NAME
from agents.orchestrator.prompts import SYSTEM_PROMPT
from agents.utils.state import AgentState
from agents.orchestrator.tools import TOOLS

model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0).bind_tools(TOOLS)


def agent_node(state: AgentState) -> dict:
    # System prompt is injected here rather than stored in state, so the
    # driver only ever has to seed state with the batch's file listing.
    messages = [SystemMessage(SYSTEM_PROMPT)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "agent")
    # tools_condition inspects the last AIMessage: routes to "tools" if it
    # has tool_calls, else to END. This is what lets the agent decide for
    # itself when the batch is finished — no explicit "done" tool needed.
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    print(graph.get_graph().draw_ascii())

    return graph.compile()
