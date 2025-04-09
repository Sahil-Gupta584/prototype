import streamlit as st
import json
from runnable import get_runnable
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
import time
import re

# Create a more descriptive title and sidebar
st.set_page_config(
    page_title="CodeHelper - AI Coding Assistant", page_icon="💻", layout="wide"
)


@st.cache_resource
def create_agent_instance():
    return get_runnable()


def extract_code_blocks(content):
    """Extract code blocks for syntax highlighting"""
    # Find all code blocks with language specification
    pattern = r"```(\w+)\n([\s\S]*?)```"
    blocks = re.findall(pattern, content)

    # Also find code blocks without language specification
    simple_pattern = r"```\n([\s\S]*?)```"
    simple_blocks = re.findall(simple_pattern, content)

    return blocks, simple_blocks


def prettify_message(content):
    """Format message content with syntax highlighting"""
    blocks, simple_blocks = extract_code_blocks(content)

    # Replace code blocks with syntax highlighted versions
    for lang, code in blocks:
        highlighted = f"```{lang}\n{code}\n```"
        content = content.replace(f"```{lang}\n{code}```", highlighted)

    for code in simple_blocks:
        highlighted = f"```\n{code}\n```"
        content = content.replace(f"```\n{code}```", highlighted)

    return content


def main():
    # Initialize the agent
    agent = create_agent_instance()

    # Sidebar for configuration and project info
    with st.sidebar:
        st.title("💻 CodeHelper")
        st.markdown("Your AI-powered coding assistant")

        st.subheader("Project Information")
        st.info(
            """
        **Current Project**: user_project
        **Tech Stack**: Next.js 15, App Router, DaisyUI
        """
        )

        st.subheader("Commands")
        st.markdown(
            """
        Try these commands:
        - "Create a login page"
        - "Add a dark mode toggle"
        - "Fix the navigation bar"
        - "Analyze the API routes"
        """
        )

        if st.button("Clear Chat History"):
            st.session_state.messages = [SystemMessage(content=system_message)]
            st.session_state.display_messages = []
            st.rerun()

    # Main content area
    st.title("CodeHelper AI Assistant")

    # System message
    system_message = """
    You are a coding assistant who must **always** use available tools to edit and modify code. 
    Always use builder_tool for coding related tasks.
    Your tech stack is only: Next.js 15 with app router, backend in API routes, DaisyUI for UI framework.
    
    Before making changes, analyze the current code structure to maintain consistency.
    Provide step-by-step explanations of what you're doing and why.
    """

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content=system_message)]

    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []

    # Display chat messages from history
    for message in st.session_state.display_messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(prettify_message(message.content))
        elif isinstance(message, HumanMessage):
            with st.chat_message("user", avatar="👤"):
                st.markdown(message.content)

    # Input area
    prompt = st.chat_input("What would you like to do with your code?")
    if prompt:
        # Show spinner during processing
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Add to both actual messages (for agent) and display messages (for UI)
        user_message = HumanMessage(content=prompt)
        st.session_state.messages.append(user_message)
        st.session_state.display_messages.append(user_message)

        # Get AI response with spinner
        with st.spinner("Thinking and coding..."):
            response = agent.invoke(
                input={"messages": st.session_state.messages},
            )

        # Process AI messages
        ai_messages = [
            msg for msg in response["messages"] if isinstance(msg, AIMessage)
        ]
        ai_message = (
            ai_messages[-1]
            if ai_messages
            else AIMessage(content="(No response generated)")
        )

        # Display the response
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(prettify_message(ai_message.content))

        # Update session state
        st.session_state.messages.extend(response["messages"])
        st.session_state.display_messages.append(ai_message)


if __name__ == "__main__":
    main()
