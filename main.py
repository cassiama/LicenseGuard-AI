import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# import the environment variables and set up the MCP server
# TODO: add a Pydantic Settings setup so that we're not using `python-dotenv` anymore
load_dotenv()
MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "http://localhost")
MCP_SERVER_PORT = os.getenv("MCP_SERVER_PORT", "8000")
MCP_URL = f"{MCP_SERVER_HOST}:{MCP_SERVER_PORT}/mcp"

app = FastAPI()


# allows the frontend to make requests to this REST API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # TODO: for production, change to be specifically the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


async def test_mcp_client_streaming_response():
    try:
        client = MultiServerMCPClient(
            { "licenseguard-agent": { "url": MCP_URL, "transport": "streamable_http" } }
        )
        async with client.session("licenseguard-agent") as session:
            yield "Connecting to MCP server... \n\n"
            tools = await load_mcp_tools(session)
            yield "Connection successful! \n\n"
            yield f"Found {len(tools)} tools: \n\n"

            for tool in tools:
                yield f"Tool: {tool.name} \n\n"

    except Exception as e:
        yield f"Error connecting to MCP server: {e}\n\n"


@app.get("/generate/report")
async def main():
    return StreamingResponse(
        test_mcp_client_streaming_response(),
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    asyncio.run(main())
