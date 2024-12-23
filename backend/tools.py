from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk
import requests
from backend.models import Tool, ToolName, ToolType
import subprocess
from duckduckgo_search import DDGS
import io
from unstructured.partition.pdf import partition_pdf


def fetch_wikipedia_summary(topic: str) -> dict:
    """Fetch a summary of a topic from Wikipedia."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "topic": topic,
            "summary": data.get("extract", "No summary available."),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        }
    return {"topic": topic, "summary": "Error fetching summary.", "url": ""}


fetch_wikipedia_summary_definition = ChatCompletionToolParamFunctionChunk(
    name=ToolName.FETCH_WIKIPEDIA_SUMMARY.value,
    description="Fetch a summary of a topic from Wikipedia.",
    parameters=dict(
        type="object",
        properties={
            "topic": {"type": "string", "description": "The topic to fetch."},
        },
        required=["topic"],
        additionalProperties=False,
    ),
)


def execute_python_script(code: str, timeout: int = 5) -> dict:
    """Execute a Python script and return the output."""
    try:
        if not "print(" in code:
            code = f"print({code})"
        # Execute the code in a separate process with timeout
        result = subprocess.run(
            ["python", "-c", code], capture_output=True, text=True, timeout=timeout
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "Execution timed out", "success": False}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "success": False}


execute_python_definition = ChatCompletionToolParamFunctionChunk(
    name=ToolName.EXECUTE_PYTHON.value,
    description=(
        "Execute a Python script and return the output. "
        "Use for all tasks that can be done with Python. "
        "Examples are: coding tasks, math calculations, data analysis, text processing, "
        "working with dates and times, file and data operations."
    ),
    parameters=dict(
        type="object",
        properties={
            "code": {"type": "string", "description": "The Python code to execute"},
        },
        required=["code"],
        additionalProperties=False,
    ),
)


def browse_web(url: str, timeout: int = 10) -> dict:
    """Fetch content from a web page."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # Raise an exception for bad status codes
        return {
            "url": url,
            "content": response.text,
            "status_code": response.status_code,
            "success": True,
        }
    except requests.RequestException as e:
        return {
            "url": url,
            "content": str(e),
            "status_code": getattr(e.response, "status_code", None),
            "success": False,
        }


browse_web_definition = ChatCompletionToolParamFunctionChunk(
    name=ToolName.BROWSE_WEB.value,
    description="Fetch content from a web page, open links or execute api requests",
    parameters=dict(
        type="object",
        properties={
            "url": {"type": "string", "description": "The URL to fetch content from"},
        },
        required=["url"],
        additionalProperties=False,
    ),
)


def search_web(query: str, num_results: int = 5) -> dict:
    """Search the web using DuckDuckGo search API."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results, safesearch="off"))
        return {"query": query, "results": results, "success": True}
    except Exception as e:
        return {"query": query, "results": [], "success": False, "error": str(e)}


search_web_definition = ChatCompletionToolParamFunctionChunk(
    name=ToolName.SEARCH_WEB.value,
    description=(
        "Search the web using a search engine."
        "Use this tool for all tasks that require searching the web."
        "Examples are: finding information, news, articles, websites, etc."
    ),
    parameters=dict(
        type="object",
        properties={
            "query": {"type": "string", "description": "The search query"},
        },
        required=["query"],
        additionalProperties=False,
    ),
)


def parse_online_pdf(url: str) -> dict:
    """Parse text content from an online PDF file."""
    try:
        # Download the PDF
        response = requests.get(url)
        response.raise_for_status()
        pdf_stream = io.BytesIO(response.content)

        # Parse the PDF
        elements = partition_pdf(file=pdf_stream)
        # Convert elements to text and join them
        text_content = "\n".join([str(element) for element in elements])

        return {"url": url, "content": text_content, "success": True}
    except Exception as e:
        return {"url": url, "content": f"Error parsing PDF: {str(e)}", "success": False}


parse_pdf_definition = ChatCompletionToolParamFunctionChunk(
    name=ToolName.PARSE_ONLINE_PDF.value,
    description="Parse and extract text content from an online PDF file.",
    parameters=dict(
        type="object",
        properties={
            "url": {
                "type": "string",
                "description": "The URL of the PDF file to parse",
            },
        },
        required=["url"],
        additionalProperties=False,
    ),
)


TOOL_REGISTRY = {
    ToolName.FETCH_WIKIPEDIA_SUMMARY: Tool(
        name=ToolName.FETCH_WIKIPEDIA_SUMMARY,
        implementation=fetch_wikipedia_summary,
        definition=ChatCompletionToolParam(
            type=ToolType.FUNCTION.value,
            function=fetch_wikipedia_summary_definition,
        ),
    ),
    ToolName.EXECUTE_PYTHON: Tool(
        name=ToolName.EXECUTE_PYTHON,
        implementation=execute_python_script,
        definition=ChatCompletionToolParam(
            type=ToolType.FUNCTION.value,
            function=execute_python_definition,
        ),
    ),
    ToolName.BROWSE_WEB: Tool(
        name=ToolName.BROWSE_WEB,
        implementation=browse_web,
        definition=ChatCompletionToolParam(
            type=ToolType.FUNCTION.value,
            function=browse_web_definition,
        ),
    ),
    ToolName.SEARCH_WEB: Tool(
        name=ToolName.SEARCH_WEB,
        implementation=search_web,
        definition=ChatCompletionToolParam(
            type=ToolType.FUNCTION.value,
            function=search_web_definition,
        ),
    ),
    ToolName.PARSE_ONLINE_PDF: Tool(
        name=ToolName.PARSE_ONLINE_PDF,
        implementation=parse_online_pdf,
        definition=ChatCompletionToolParam(
            type=ToolType.FUNCTION.value,
            function=parse_pdf_definition,
        ),
    ),
}


TOOLS_DEFINITIONS = [tool.definition for tool in TOOL_REGISTRY.values()]
