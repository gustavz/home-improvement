import json
import os
from typing import Any
import openai
from backend.models import ChatMessage, ChatRole, OpenAIModel, ToolChoice
from backend.tools import TOOL_REGISTRY, TOOLS_DEFINITIONS
from dotenv import load_dotenv
from openai.types.chat import (
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
    ChatCompletionMessage,
)

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def execute_tool_call(tool_call: ChatCompletionMessageToolCall) -> Any:
    """Execute a tool call based on the registry."""
    tool_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    if tool_name in TOOL_REGISTRY:
        print(f"> Executing tool '{tool_name}' with arguments {arguments}'")
        tool = TOOL_REGISTRY[tool_name].implementation
        result = tool(**arguments)
        print(f"> Tool result: {result}")
        return result
    raise ValueError(f"tool '{tool_name}' not found in the registry.")


def handle_user_query(
    user_query: str,
    messages: list[ChatMessage] | None = None,
    model: OpenAIModel = OpenAIModel.DEFAULT,
    tools: list[ChatCompletionToolParam] | None = TOOLS_DEFINITIONS,
    tool_choice: ToolChoice = ToolChoice.AUTO,
) -> list[ChatMessage | ChatCompletionMessage]:
    """Handle a user's query by interacting with OpenAI and registered tools."""
    messages = messages or []
    messages.append(ChatMessage(content=user_query, role=ChatRole.USER.value))

    while True:
        response = openai.chat.completions.create(
            model=model.value,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice.value,
        )

        message = response.choices[0].message
        if message.tool_calls:
            for tool_call in message.tool_calls:
                try:
                    tool_result = execute_tool_call(tool_call)
                    # Append the assistant's message with the tool call
                    messages.append(
                        ChatMessage(
                            role=ChatRole.ASSISTANT.value,
                            tool_calls=[tool_call],
                            content=None,
                        )
                    )
                    # Append the tool result
                    messages.append(
                        ChatMessage(
                            role=ChatRole.TOOL.value,
                            tool_call_id=tool_call.id,
                            content=json.dumps(tool_result),
                        )
                    )
                except Exception as e:
                    messages.append(
                        ChatMessage(
                            role=ChatRole.ASSISTANT.value,
                            content=(
                                "An error occurred while executing "
                                f"'{tool_call.function.name}': {str(e)}"
                            ),
                        )
                    )
                    return messages
        else:
            print(f"> Assistant's Response:\n{message.content}")
            messages.append(message)
            return messages
