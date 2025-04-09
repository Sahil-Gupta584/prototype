from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import Annotated, Literal, Dict, List, Optional, Union, Any
from dotenv import load_dotenv
from langchain_core.messages import (
    ToolMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
    SystemMessage,
)
import json
from langgraph.types import Command
from utils import get_tree, make_llm_call
from tools import (
    builder_tool,
    available_tools,
    get_codebase_content,
)
import re
from pydantic import BaseModel
import ast

load_dotenv()


def extract_final_answer(content):
    """Extract the final answer from between <finalAnswer> tags"""
    match = re.search(r"<finalAnswer>(.*?)</finalAnswer>", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content


class GraphState(BaseModel):
    messages: List[BaseMessage]
    context: Optional[Dict[str, Any]] = {}
    return_to_agent_node: bool = False


count = 0
AGENT_NODE = "agent_node"
TOOLS_NODE = "tools_node"


def agent_node(state: GraphState) -> Dict[str, AnyMessage]:
    """Main agent that coordinates responses and delegates to specialized tools/agents"""
    global count
    count += 1
    state_messages = list(state.messages)
    last_human_message = next(
        (msg for msg in reversed(state_messages) if isinstance(msg, HumanMessage)), None
    )
    return_to_agent_node = state.return_to_agent_node
    print()
    print("last_human_message", last_human_message)
    # print("message", state_messages)

    if not last_human_message:
        return Command(
            goto=END,
            update={
                "messages": [
                    AIMessage(content="I'm ready to help with your coding tasks.")
                ]
            },
        )

    tree_structure = get_tree("C:/zeropointlab/user_project")

    prompt = f"""
        User Query: {last_human_message.content}
        
        Instructions:
            1. You are a primary agent in an architecture of specialized coding agents.
            3. Use the get_codebase_content tool freely to examine files before modifying them.
            4. Give intructions to builder_tool at once for making changes to the codebase to fulfill user requirements related to codebase.
            
        Current Codebase:
            TechStack -> Nextjs 15 with app router, backend in api route, DaisyUI for UI framework
        
        Structure ->
            user_project/
            {tree_structure}
            
        Context from previous interactions:
        {state.context if return_to_agent_node else 'Nothing Yet.'}
    """

    if count == 4:
        print("prompt", prompt)
    new_messages = []

    response = make_llm_call(
        input=[*state_messages, HumanMessage(content=prompt)],
        tools=[builder_tool, get_codebase_content],
    )

    # print("response.content", response.content)
    # print("response.tools", response.tool_calls)

    new_messages.append(
        AIMessage(
            content=response.content if response.content else "None",
            kwargs={"tool_calls": response.tool_calls},
        )
    )

    context_updates = {"files_contents": {}}
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool = available_tools.get(tool_name)

        if tool is None:
            raise Exception(f"Tool '{tool_name}' not found.")

        if tool_name == "builder_tool":
            tool_msg, ai_msg = builder_tool.invoke(
                {
                    "detailedQuery": tool_call["args"]["detailedQuery"],
                    "state_messages": state_messages,
                },
            )

            changes_dict = ast.literal_eval(tool_msg.content)

            new_messages.append(ai_msg)

            for filePath, file_content in changes_dict.items():
                context_updates["files_contents"][filePath] = file_content

        elif tool_name == "get_codebase_content":
            files_content = get_codebase_content.invoke(tool_call["args"])
            for filePath, file_content in files_content.items():
                context_updates["files_contents"][filePath] = files_content

            return Command(
                goto=TOOLS_NODE,
                update={
                    "messages": state_messages + new_messages,
                    "context": {**state.context, **context_updates},
                    "return_to_agent_node": True,
                },
            )

        else:
            print("unregistered tool_call!")
            output = tool.invoke(tool_call["args"])

    return Command(
        goto=END,
        update={
            "messages": state_messages + new_messages,
            "context": {**state.context, **context_updates},
            "return_to_agent_node": False,
        },
    )


def tools_node(state: GraphState) -> Dict[str, AnyMessage]:
    """Specialized node for code building and implementation"""

    # Find the tools_node tool call
    if state.return_to_agent_node:
        return Command(goto=AGENT_NODE)

    return Command(
        goto=END,
    )


def get_runnable():
    """Create and configure the workflow graph"""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node(AGENT_NODE, agent_node)
    workflow.add_node(TOOLS_NODE, tools_node)

    # Set the entry point
    workflow.set_entry_point(AGENT_NODE)

    # Add conditional edges to determine the flow
    # workflow.add_conditional_edges(
    #     "agent_node", should_continue, {"agent_node": "agent_node", TOOLS_NODE: TOOLS_NODE, "done": END}
    # )

    # workflow.add_conditional_edges(
    #     TOOLS_NODE,
    #     should_continue,
    #     {"agent_node": "agent_node", TOOLS_NODE: TOOLS_NODE, "done": END},
    # )

    # Compile the workflow
    app = workflow.compile()
    return app


app = get_runnable()
