import os
from datetime import datetime
from typing import Annotated
from dotenv import load_dotenv
from fastapi import FastAPI, Form, UploadFile, File, Header, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from agent.graph import agent as agent_graph
    from agent.graph import AgentState
except ImportError:
    # Handle the case where the file is run directly
    from agent import graph as agent_graph, AgentState

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

class AnalysisRequest(BaseModel):
    project_name: str
    requirements_content: str


async def stream_agent_response(initial_state: AgentState):
    """
    Calls the agent graph and streams only the final report.
    """
    print(f"[{datetime.now()}]: Starting the agent's streaming response.")
    
    # get detailed events and filter for just the 'final_report' chunk from the 'summarize' node
    async for event in agent_graph.astream_events(
        initial_state,
        version="v2",
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            # stream the response token by token
            chunk = event["data"]["chunk"]
            print(chunk)
            yield chunk.content

    print(f"[{datetime.now()}]: Stream complete.")


@app.post("/generate/report")
async def generate_report(
    requirements_file: Annotated[UploadFile, File(
        description="A requirements.txt file (text/plain).")],
    project_name: Annotated[str, Form(
        description="The name of the project")],
    authorization: str = Header(..., description="The user's Bearer token")
):
    """
    Accepts a requirements.txt file upload, a project name & the user's authentication token, analyzes each license associated with the dependencies in the 'requirements.txt' file, and initiates an intelligent analysis of the requirements.txt file.

    Throws a 401 if:
        - the authorization header is missing.
        - the authorization header is in the correct form.

    Keyword arguments:

    file -- an non-empty 'requirements.txt'

    project_name -- the name of your project

    authorization -- a header that contains the user's Bearer token
    """    
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is missing.")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header should be in the form of 'Bearer <token>'.")

    requirements_content: str = (await requirements_file.read()).decode("utf-8")

    # initialize the internal agent with an empty JSON & report, the requirements.txt & 
    # project name, and the user's auth token
    # NOTE: passing in the user's auth token is a TEMPORARY solution; token passthrough is 
    # heavily frowned upon (source: https://modelcontextprotocol.io/specification/2025-06-18/basic/security_best_practices#token-passthrough)
    initial_state = AgentState(
        project_name=project_name,
        requirements_content=requirements_content,
        auth_token=authorization.split(" ")[1], # this gets the token from the Authorization header
        analysis_json={},
        final_report=""
    )

    # return the streaming response from the internal agent
    return StreamingResponse(
        stream_agent_response(initial_state),
        media_type="text/plain",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


if __name__ == "__main__":
    pass
