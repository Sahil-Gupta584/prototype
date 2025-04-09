import os
from langchain_core.language_models import LanguageModelInput
from typing import Callable, Sequence, Any
from google.ai.generativelanguage_v1beta.types import (
    Tool as GoogleTool,
)
from langchain_core.tools import BaseTool
from config import client
from langchain_core.messages import BaseMessage

IGNORED = {
    "node_modules",
    ".git",
    "__pycache__",
    ".next",
}  # Add folders/files to ignore


def get_tree(directory, prefix=""):
    tree_str = ""
    items = sorted(os.listdir(directory))

    for index, item in enumerate(items):
        if item in IGNORED:  # Skip ignored items
            continue

        path = os.path.join(directory, item)
        is_last = index == len(items) - 1
        connector = "└── " if is_last else "├── "

        tree_str += prefix + connector + item + "\n"

        if os.path.isdir(path):  # If it's a folder, recurse
            new_prefix = prefix + ("    " if is_last else "│   ")
            tree_str += get_tree(path, new_prefix)

    return tree_str


tools_type = Sequence[
    dict[str, Any] | type | Callable[..., Any] | BaseTool | GoogleTool
]


def make_llm_call(input: LanguageModelInput, tools: tools_type) -> BaseMessage:
    agent = client
    if tools:
        agent = client.bind_tools(tools)
    response = agent.invoke(input)

    if isinstance(response.content, list):
        response.content = "\n".join(response.content)

    return response
