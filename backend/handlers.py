import json
from typing import Any
import backend.metering
from backend.models import ChatRole, Models, ToolChoice
from backend.tools import TOOL_REGISTRY, TOOLS_DEFINITIONS
from litellm import (
    ChatCompletionToolParam,
    ChatCompletionMessageToolCall,
    Message,
    completion,
)


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
    messages: list[Message] | None = None,
    model: Models = Models.DEFAULT,
    tools: list[ChatCompletionToolParam] | None = TOOLS_DEFINITIONS,
    tool_choice: ToolChoice = ToolChoice.AUTO,
) -> list[Message]:
    """Handle a user's query by interacting with OpenAI and registered tools."""
    messages = messages or []
    messages.append(Message(content=user_query, role=ChatRole.USER.value))

    while True:
        response = completion(
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
                        Message(
                            role=ChatRole.ASSISTANT.value,
                            tool_calls=[tool_call],
                            content=None,
                        )
                    )
                    # Append the tool result
                    messages.append(
                        Message(
                            role=ChatRole.TOOL.value,
                            tool_call_id=tool_call.id,
                            content=json.dumps(tool_result),
                        )
                    )
                except Exception as e:
                    messages.append(
                        Message(
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
