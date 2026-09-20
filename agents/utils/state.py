from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # No src_root/dst_root/batch_files here on purpose: this state gets
    # passed to the LLM as message history, so the real filesystem roots
    # never enter it. Those live in RunnableConfig instead (see tools.py).
    messages: Annotated[list, add_messages]
