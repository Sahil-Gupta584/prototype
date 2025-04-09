from pydantic import BaseModel, Field
from config import client
from langchain.tools import tool, Tool
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage, BaseMessage
from utils import get_tree, make_llm_call
import os
import json
from typing import List, Dict, Any, Optional, Union
import re


class EditFileSchema(BaseModel):
    filePath: str = Field(
        description="The relative path of the file to edit. Root folder is user_project."
    )
    fileContent: str = Field(description="The complete content to write to the file.")


class ReadFileSchema(BaseModel):
    filePath: str = Field(
        description="The relative path of the file to read. Root folder is user_project."
    )


class BuilderSchema(BaseModel):
    detailedQuery: str = Field(
        description="Detailed explanation of the coding task to perform."
    )
    state_messages: List[BaseMessage]


class GetCodebaseContentSchema(BaseModel):
    filesPaths: List[str] = Field(
        description="List of file paths to retrieve content from. Example: ['src/package.json', 'src/app/page.tsx']"
    )


class AnalyzeCodeSchema(BaseModel):
    filePath: str = Field(description="The relative path of the file to analyze.")
    analysisType: Optional[str] = Field(
        default="general",
        description="Type of analysis to perform: 'general', 'performance', 'security', or 'best-practices'.",
    )


@tool(
    args_schema=ReadFileSchema,
    description="Reads the content of a specified file from the user_project directory.",
)
def read_file(filePath: str) -> str:
    """Read a file from the project directory."""
    print(f"Reading file: {filePath}")
    full_path = os.path.join(os.getcwd(), "user_project", filePath)

    if not os.path.isfile(full_path):
        print(f"File {filePath} doesn't exist")
        return f"File '{filePath}' does not exist."

    try:
        with open(full_path, "r", encoding="utf-8") as file:
            fileContent = file.read()
        print(f"Successfully read file: {filePath}")
        return fileContent
    except Exception as e:
        print(f"Error reading file {filePath}: {str(e)}")
        return f"Error reading file: {str(e)}"


@tool(
    args_schema=EditFileSchema,
    description="Writes content to a specified file path, creating directories if needed. Root folder is user_project.",
)
def edit_file(filePath: str, fileContent: str) -> str:
    """Edit or create a file in the project directory."""
    print(f"Editing file: {filePath}")

    # Get full file path
    full_path = os.path.join(os.getcwd(), "user_project", filePath)

    try:
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Open and write to the file
        with open(full_path, "w", encoding="utf-8") as file:
            file.write(fileContent)

        print(f"Successfully edited file: {filePath}")
        return fileContent
    except Exception as e:
        print(f"Error editing file {filePath}: {str(e)}")
        return f"Error editing file: {str(e)}"


@tool(
    args_schema=BuilderSchema,
    description="Specialized coding agent that can implement features and modify code across multiple files.",
)
def builder_tool(
    detailedQuery: str, state_messages: List[BaseMessage]
) -> tuple[ToolMessage, AIMessage]:
    """Implements code changes across multiple files to fulfill requirements."""
    print(f"Builder tool invoked with query: {detailedQuery[:60]}...")

    prompt = f"""
        User Query: {detailedQuery}
        
        Instructions:
            1. Analyze the user requirements and create a detailed execution plan.
            2. For each file that needs modification:
               a. Use `read_file` to check the current content
               b. Use `edit_file` to update the file with your changes
            3. Execute ALL necessary steps - don't stop after the first tool call.
            4. Break complex changes into smaller, verifiable steps.
            5. After making changes, verify that they work as expected.
            6. When including existing code, maintain the original structure and style.

        Current project structure:
        TechStack: Next.js 15 with app router, backend in API routes, DaisyUI for UI framework
        
        user_project/
        {get_tree("C:/zeropointlab/user_project")}
    """

    print("state_messages_length", len(state_messages))
    agent = client.bind_tools(tools=[edit_file, read_file])
    response = agent.invoke(input=[*state_messages, HumanMessage(content=prompt)])
    response
    messages = []
    files_content = {}

    messages.append(HumanMessage(content=prompt))
    if response.content:
        messages.append(AIMessage(content=response.content))

    # Process initial tool calls
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool = available_tools.get(tool_name)

            if tool is None:
                raise Exception(f"Tool '{tool_name}' not found.")

            tool_output = tool.invoke(tool_call["args"])

            if tool_name in ["read_file", "edit_file"]:
                files_content[tool_call["args"]["filePath"]] = tool_output

    # Continue until all steps are completed
    attempt_count = 0
    max_attempts = 10  # Prevent infinite loops

    while attempt_count < max_attempts:
        attempt_count += 1

        follow_up_prompt = f"""
            Current status update:
            
            I've processed your coding task and made the following changes:
            - Modified files: {list(files_content.keys()) if files_content else "None yet"}
            
            Current project structure:
            user_project/
            {get_tree("C:/zeropointlab/user_project")}

            Have all planned steps been completed ? If yes, simply reply yes and summarize what was done.
            If no, reply with 'No' and continue implementation by using edit_file, read_file tools if requires.

            Performed changes:
            {files_content if files_content else 'Nothing'}
        """
        agent = client.bind_tools(tools=[edit_file, read_file])
        follow_up_response = agent.invoke(
            input=[*messages, HumanMessage(content=follow_up_prompt)]
        )

        content = follow_up_response.content
        if isinstance(content, list):
            content = content[0]

        messages.append(AIMessage(content=content))

        # Check if we're done
        if "yes" in content.lower():
            print(f"Implementation completed after {attempt_count} iterations")
            break

        if "no" in content.lower() and not response.tool_calls:
            print(f"Unextpected response occured.")
            break

        # Process additional tool calls
        if follow_up_response.tool_calls:
            for tool_call in follow_up_response.tool_calls:
                tool_name = tool_call["name"]
                tool = available_tools.get(tool_name)

                if tool is None:
                    raise Exception(f"Tool '{tool_name}' not found.")

                tool_output = tool.invoke(tool_call["args"])

                if tool_name in ["read_file", "edit_file"]:
                    files_content[tool_call["args"]["filePath"]] = tool_output

        # If no more tool calls and not done, we might be stuck
        if not follow_up_response.tool_calls and attempt_count > max_attempts:
            print("No more tool calls but implementation not confirmed complete")
            break
    last_ai_message = [msg for msg in messages if isinstance(msg, AIMessage)][-1]
    return (
        ToolMessage(
            content=str(files_content),
            tool_call_id="builder-summary",
            name="build_summary",
        ),
        last_ai_message,
    )


@tool(
    args_schema=GetCodebaseContentSchema,
    description="Retrieves content from multiple files in the codebase.",
)
def get_codebase_content(filesPaths: List[str]) -> Dict[str, str]:
    """Get content from multiple files in the codebase."""
    print(f"Getting content for {len(filesPaths)} files")
    files_content = {}

    for filepath in filesPaths:
        try:
            file_content = read_file.invoke({"filePath": filepath})
            files_content[filepath] = file_content
        except Exception as e:
            files_content[filepath] = f"Error reading file: {str(e)}"

    return files_content


@tool(
    args_schema=AnalyzeCodeSchema,
    description="Analyzes code for quality, potential issues, and improvement suggestions.",
)
def analyze_code(filePath: str, analysisType: str = "general") -> Dict[str, Any]:
    """Analyzes code in a specified file and provides insights."""
    print(f"Analyzing code in {filePath} with type: {analysisType}")

    # Get the file content
    file_content = read_file.invoke({"filePath": filePath})
    if file_content.startswith("File") and "does not exist" in file_content:
        return {"error": file_content}

    # Create a prompt for code analysis
    analysis_prompts = {
        "general": "Analyze this code for general quality, structure, and potential issues.",
        "performance": "Analyze this code for performance issues and optimization opportunities.",
        "security": "Analyze this code for security vulnerabilities and best practices.",
        "best-practices": "Analyze this code against best practices and suggest improvements.",
    }

    prompt = f"""
    {analysis_prompts.get(analysisType, analysis_prompts["general"])}
    
    File: {filePath}
    
    ```
    {file_content}
    ```
    
    Provide a structured analysis with:
    1. Summary of code quality
    2. Identified issues/concerns
    3. Specific improvement recommendations
    4. Code architecture assessment
    
    Format your response as JSON with these keys: summary, issues, recommendations, architecture.
    """

    # Use the LLM to analyze the code
    analyzer = client.invoke(input=[HumanMessage(content=prompt)])

    raw_content = analyzer.content.strip()
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_content, re.DOTALL)
    json_str = json_match.group(1) if json_match else raw_content

    print("analyzer.c", analyzer.content)
    try:
        analysis_result = json.loads(json_str)
        return analysis_result
    except json.JSONDecodeError:
        # Fallback in case the response isn't valid JSON
        return {
            "summary": analyzer.content[:500],
            "issues": [],
            "recommendations": [],
            "architecture": "Analysis format error",
        }


# Register all available tools
available_tools = {
    "builder_tool": builder_tool,
    "edit_file": edit_file,
    "read_file": read_file,
    "get_codebase_content": get_codebase_content,
    "analyze_code": analyze_code,
}
