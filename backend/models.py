from enum import StrEnum
from typing import Callable
from openai.types.chat import ChatCompletionToolParam, ChatCompletionMessageToolCall
from openai import BaseModel


class OpenAIModel(StrEnum):
    O_1 = "o1"
    O_1_MINI = "o1-mini"
    GPT_4 = "gpt-4"
    GPT_4_O = "gpt-4o"
    GPT_4_O_MINI = "gpt-4o-mini"
    DEFAULT = GPT_4_O_MINI


class ChatRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolType(StrEnum):
    FUNCTION = "function"


class ToolChoice(StrEnum):
    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


class ToolName(StrEnum):
    FETCH_WIKIPEDIA_SUMMARY = "fetch_wikipedia_summary"
    EXECUTE_PYTHON = "execute_python"
    BROWSE_WEB = "browse_web"
    SEARCH_WEB = "search_web"
    PARSE_ONLINE_PDF = "parse_online_pdf"


class Tool(BaseModel):
    name: ToolName
    implementation: Callable
    definition: ChatCompletionToolParam


class ChatMessage(BaseModel):
    role: str | None = None
    content: str | None = None
    tool_calls: list[ChatCompletionMessageToolCall] | None = None
