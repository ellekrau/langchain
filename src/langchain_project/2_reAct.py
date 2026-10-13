# For ReAct understanding

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openrouter import ChatOpenRouter
from phoenix.otel import register

from langchain.chat_models import init_chat_model
from langchain.tools import tool

# Load environment variables from .env file
load_dotenv()

# Enable OpenTelemetry tracing and metrics for the project
tracer_provider = register(project_name="cat_color_and_age", auto_instrument=True)
tracer = tracer_provider.get_tracer(__name__)

MODEL = "deepseek/deepseek-v4.1-flash"
MAX_ITERATIONS = 10  # Control the maximum number of iterations for the agent loop


@tool
def get_cat_color(name: str) -> str:
    """Get the color of a cat by name"""
    cats_colors = {
        "rey": "tabby",
        "felina": "black",
        "nora": "grey and white",
        "joe": "orange",
    }
    return cats_colors.get(name.lower(), "unknown")


@tool
def get_cat_age(name: str) -> int:
    """Get the age of a cat by name"""
    cats_ages = {
        "rey": 6,
        "felina": 6,
        "nora": 5,
        "joe": 5,
    }
    return cats_ages.get(name.lower(), -1)


# Agent loop
@tracer.chain
def agent_loop(question: str) -> None:
    tools = [get_cat_color, get_cat_age]
    tools_dict = {tool.name: tool for tool in tools}

    llm = ChatOpenRouter(model=MODEL, temperature=0.0)
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content="You are a helpful assistant that can answer questions about cats."
            "You have access to catalogs of cat colors and ages. \n\n"
            "Strict rules:\n"
            "1. You must use the tools provided to answer questions about cats.\n"
            "2. You must not make up information about cats.\n"
            "3. If you don't know the answer, you must say 'I don't know'."
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(MAX_ITERATIONS):
        print()
        print(f"Iteration {iteration + 1}:")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        # If the AI message does not contain any tool calls, it means the AI has provided a final answer
        if not tool_calls:
            print("No tool calls made. Final answer:")
            print(ai_message.content)
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"Tool call made: {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            print(f"Tool {tool_name} not found. Skipping this tool call.")
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)
        print(f"Observation from tool '{tool_name}': {observation}")

        messages.append(ai_message)
        print(
            f"Updated messages for next iteration: {messages} with ai_message: {ai_message.content}"
        )
        tool_message = ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        messages.append(tool_message)
        print(f"Updated messages adding ToolMessage: {tool_message}")

    print("Maximum iterations without a final answer reached.")
    return None


agent_loop("What is the color and age of Rey?")
