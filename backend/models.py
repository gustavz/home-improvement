from enum import StrEnum
from typing import Callable
from litellm import ChatCompletionToolParam
from pydantic import BaseModel


class Models(StrEnum):
    O_1 = "openai/o1"
    O_1_MINI = "openai/o1-mini"
    GPT_4 = "openai/gpt-4"
    GPT_4_O = "openai/gpt-4o"
    GPT_4_O_MINI = "openai/gpt-4o-mini"
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
