import os
import json
import asyncio
import traceback
import httpcore
from typing import Awaitable, Callable, TypedDict, Dict, Any, Optional
from dotenv import load_dotenv
from mcp.types import TextContent
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult
from pydantic import SecretStr

AGENT_NAME = "licenseguard-agent"

# import the environment variables and set up the MCP server
# TODO: add a Pydantic Settings setup so that we're not using `python-dotenv` anymore
load_dotenv()
MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "http://localhost")
MCP_SERVER_PORT = os.getenv("MCP_SERVER_PORT", "8000")
MCP_URL = f"{MCP_SERVER_HOST}:{MCP_SERVER_PORT}/mcp"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if OPENAI_API_KEY == "":
    raise RuntimeError("Missing OpenAI API key!")


SYSTEM_PROMPT = """
You are a specialized License Compliance Assistant and Python Ecosystem Expert. Your only job is to analyze the JSON data from a tool call and generate a high-value, clear, human-readable, and actionable risk report in Markdown.

Instructions:
    1. **Analyze**: Review the list of packages and their licenses provided in the JSON.
    - *Crucial*: You must use the `analyze_dependencies` tool to get the raw JSON analysis data.
    2. **Verify**: Compare the detected licenses against your internal knowledge base of common Python packages. 
    - *Crucial*: If a package is known to be strictly copyleft but the tool detected a permissive license, FLAG IT as a "High Risk Verification Needed" item. Make sure to note that the confidence score in this particular license's accuracy has declined due to that discovery.
    3. **Synthesize**: Do not just list conflicts; explain *why* they are conflicts (e.g., "Viral effect of GPL linked with proprietary code").
    4. **Recommend**: For every high-risk or conflicting package, you MUST provide a specific, drop-in alternative from the Python ecosystem if one exists.

Your response must be a summary in Markdown format. Follow the template below exactly.

Markdown Template:
## 🛡️ LicenseGuard Report

Here is the compliance analysis for your project.

### 📦 Package Summary
(If no packages were provided, explicitly state: "No packages were provided.")
(If packages exist, list all packages formatted as: `package_name`: **License**)

### ⚠️ License Conflicts
(If no conflicts, explicitly state: "No obvious license conflicts detected here.")
(If conflicts/risks exist, use the format below:)
* **Conflict**: [License A] vs [License B]
    * **Source**: `[Package Name]`
    * **Risk**: [Brief explanation, e.g., "GPL requires the entire project to be open-sourced."]

### 🪜 Actionable Next Steps
(If no conflicts in the previous section, explicitly state: "No further actions need to be taken.")
(If there were conflicts in the previous section, you must provide specific alternatives here. Do not give generic legal advice.)
**Recommended Swaps:**
* **Problem**: `[Problem Package]` ([License])
    * **Solution**: Switch to `[Alternative Package]` ([License])
    * **Migration Note**: [Brief note, e.g., "Direct drop-in replacement" or "Requires minor code changes."]

**General Steps:**
1. [Specific step 1]
2. [Specific step 2]
"""


class AgentState(TypedDict):
    project_name: str
    requirements_content: str
    auth_token: (
        str  # TODO: remove this once OAuth Client Credentials login flow is implemented
    )
    analysis_json: dict  # the tool's output
    final_report: str  # the LLM"s summary


# code inspired by the "Retry on error" interceptor from the LangChain docs: 
# https://docs.langchain.com/oss/python/langchain/mcp#custom-interceptors
async def retry_tool_call(
        request: MCPToolCallRequest,
        handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
        max_retries: int = 3,
        delay: float = 1.0
) -> Optional[MCPToolCallResult]:
    """
    Retry the tool call on failure up to `max_retries` times with a delay.
    
    :param request: the tool call request to the MCP server
    :type request: MCPToolCallRequest
    :param handler: the tool call handler function
    :type handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]]
    :param max_retries: the maximum number of retry attempts
    :type max_retries: int
    :param delay: the delay (in seconds) between retry attempts
    :type delay: float
    :return: either the result from successfully calling the tool, or raises the last exception encountered
    :rtype: MCPToolCallResult | None
    """
    last_exception = Exception()
    for _ in range(max_retries):
        try:
            result = await handler(request)
            return result
        except Exception as e:
            last_exception = e
            await asyncio.sleep(delay)
    raise last_exception

# code taken from the "Error handling with fallback" interceptor from the LangChain docs:
# https://docs.langchain.com/oss/python/langchain/mcp#custom-interceptors
async def fallback_interceptor(
    request: MCPToolCallRequest,
    handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]]
):
    """
    Return a fallback value if tool execution fails.
    
    :param request: the tool call request to the MCP server
    :type request: MCPToolCallRequest
    :param handler: the tool call handler function
    :type handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]]
    """
    try:
        return await retry_tool_call(request, handler)
    except TimeoutError:
        return f"Tool {request.name} timed out. Please try again later."
    except ConnectionError:
        return f"Could not connect to the `{request.name}` service."
    except Exception as e:
        return f"An unknown error occurred while executing {request.name}: {str(e)}"

async def call_analysis_tool(state: AgentState) -> Dict[str, Any]:
    """
    An async function that calls the `analyze_dependencies` tool on the MCP server. Takes in the agent's current state (includes project name, the content of the requirements.txt file, and the current user's JWT) as an argument in order to receive a JSON from the MCP server.

    :param state: Models the agent's current state, which includes project name, the content of the requirements.txt file, and the current user's JWT.
    :type state: AgentState
    :return: A JSON that corresponds to an analysis of the licenses of the packages in the requirements.txt file. Should detail the licenses, name, current version, and confidence of the license being correct.
    :rtype: Dict[str, Any]
    """
    project_name = state["project_name"]
    requirements_content = state["requirements_content"]
    auth_token = state["auth_token"]

    analysis_json: str = ""

    try:
        # create the MCP client so that we can call the MCP server
        client = MultiServerMCPClient(
            connections={AGENT_NAME: {"url": MCP_URL, "transport": "streamable_http"}}
        )
        async with client.session(AGENT_NAME) as session:
            # invoke the specific tool we need
            tool_result = await session.call_tool(
                "analyze_dependencies",
                arguments={
                    "project_name": project_name,
                    "requirements_content": requirements_content,
                    "user_token": auth_token,
                },
            )

            # if possible, retrieve the JSON from the tool result
            if hasattr(tool_result, "content") and isinstance(
                tool_result.content[0], TextContent
            ):
                analysis_json = tool_result.content[0].text

        # Return the JSON result to be put into the "analysis_json" key in our state
        return {"analysis_json": analysis_json}

    except* httpcore.ConnectError as ce:
        raise ce
    except* Exception as e:
        traceback.print_exc()
        # raise Exception("An unknown error occurred while calling the analysis tool. Please try again later.")
    


async def summarize_report(state: AgentState) -> Dict[str, str]:
    """
    An async function that summarizes the analysis of the requirements.txt file and returns a final risk report. Takes in the agent's current state (includes the analysis of the requirements.txt file) as an argument in order to summarize the analysis JSON with the internal agent.

    :param state: Models the agent's current state, which includes the analysis of the requirements.txt file.
    :type state: AgentState
    :return: Returns the final report in a dictionary.
    :rtype: Dict[str, str]
    """
    analysis_json: dict = state["analysis_json"]

    if "error" in analysis_json:
        return {"final_report": f"Failed to generate report: {analysis_json['error']}"}

    # set up the llm and the prompt
    llm = ChatOpenAI(
        model="gpt-4o-mini", temperature=0.0, api_key=SecretStr(OPENAI_API_KEY)
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "user",
                "Here is the JSON analysis data. Please generate the report.\n\n{json_data}",
            ),
        ]
    )
    # set up the parser
    summarize_chain = prompt | llm | StrOutputParser()

    # invoke the summarizer chain
    report = await summarize_chain.ainvoke({"json_data": json.dumps(analysis_json)})

    # return the final report to be put into the "final_report" key
    return {"final_report": report}


# code inspired by this video: https://youtu.be/wuHXAzXzaM8
def build_agent_graph():
    builder = StateGraph(AgentState)

    # add our nodes for calling the analysis tool and summarizing the report
    builder.add_node("call_tool", call_analysis_tool)
    builder.add_node("summarize", summarize_report)

    # defining a simple & linear flow:
    builder.add_edge(START, "call_tool")
    # after calling the tool, summarize
    builder.add_edge("call_tool", "summarize")
    # after summarizing, end
    builder.add_edge("summarize", END)

    # finally, compile the graph
    graph = builder.compile()
    return graph


graph = build_agent_graph()
