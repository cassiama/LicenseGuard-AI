"""
Error handling tests for main.py module.
Tests FastAPI endpoint error handling and streaming responses.
"""

import sys
import os
import pytest
import httpx
from unittest.mock import MagicMock
from fastapi import HTTPException

# add the agent directory to the path so we can import the main module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))


@pytest.fixture
def mock_env(monkeypatch):
    """Set up dummy environment variables to avoid RuntimeError."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key-for-testing")
    monkeypatch.setenv("MCP_SERVER_HOST", "http://localhost")
    monkeypatch.setenv("MCP_SERVER_PORT", "8000")


@pytest.mark.asyncio
async def test_generate_final_report_httpx_connect_error(mock_env, monkeypatch):
    """Verify generate_final_report_from_agent raises HTTPException(503) on httpx.ConnectError."""
    # mock the agent graph to raise httpx.ConnectError
    mock_graph = MagicMock()
    
    async def mock_astream_events(*args, **kwargs):
        raise ExceptionGroup("Connection error", [httpx.ConnectError("Connection refused")])
        yield  # make it a generator
    
    mock_graph.astream_events = mock_astream_events
    
    monkeypatch.setattr("main.agent_graph", mock_graph)
    
    # import after mocking
    from main import generate_final_report_from_agent, AgentState
    
    initial_state = AgentState(
        project_name="test",
        requirements_content="requests==2.0.0",
        auth_token="test-token",
        analysis_json={},
        final_report=""
    )
    
    # assert that a HTTP 503 error is thrown
    with pytest.raises(HTTPException) as exc_info:
        await generate_final_report_from_agent(initial_state)
    
    assert exc_info.value.status_code == 503
    assert "Unable to connect to the MCP server" in exc_info.value.detail


@pytest.mark.asyncio
async def test_generate_final_report_generic_exception(mock_env, monkeypatch):
    """Verify generate_final_report_from_agent raises HTTPException(500) on generic Exception."""
    # mock the agent graph to raise a generic exception
    mock_graph = MagicMock()
    
    async def mock_astream_events(*args, **kwargs):
        raise ExceptionGroup("Generic error", [ValueError("Unexpected error")])
        yield  # make it a generator
    
    mock_graph.astream_events = mock_astream_events
    
    monkeypatch.setattr("main.agent_graph", mock_graph)
    
    # import after mocking
    from main import generate_final_report_from_agent, AgentState
    
    initial_state = AgentState(
        project_name="test",
        requirements_content="requests==2.0.0",
        auth_token="test-token",
        analysis_json={},
        final_report=""
    )
    
    # assert that a HTTP 500 error is thrown
    with pytest.raises(HTTPException) as exc_info:
        await generate_final_report_from_agent(initial_state)
    
    assert exc_info.value.status_code == 500
    assert "An unknown error occurred" in exc_info.value.detail


@pytest.mark.asyncio
async def test_generate_final_report_returns_chunks_successfully(mock_env, monkeypatch):
    """Verify generate_final_report_from_agent successfully collects chunks."""
    # mock the agent graph to return successful chunks
    mock_graph = MagicMock()
    
    async def mock_astream_events(*args, **kwargs):
        # simulate streaming events
        events = [
            {"event": "on_chat_model_stream", "data": {"chunk": MagicMock(content="Test ")}},
            {"event": "on_chat_model_stream", "data": {"chunk": MagicMock(content="report")}},
            {"event": "other_event", "data": {}},
        ]
        for event in events:
            yield event
    
    mock_graph.astream_events = mock_astream_events
    
    monkeypatch.setattr("main.agent_graph", mock_graph)
    
    # import after mocking
    from main import generate_final_report_from_agent, AgentState
    
    initial_state = AgentState(
        project_name="test",
        requirements_content="requests==2.0.0",
        auth_token="test-token",
        analysis_json={},
        final_report=""
    )
    
    # generate the final report
    chunks = await generate_final_report_from_agent(initial_state)
    
    # assert that chunks were collected
    assert chunks is not None
    assert len(chunks) == 2
    assert chunks[0] == "Test "
    assert chunks[1] == "report"


@pytest.mark.asyncio
async def test_stream_agent_response_yields_all_chunks(mock_env):
    """Verify stream_agent_response yields all chunks correctly."""
    # import the function
    from main import stream_agent_response
    
    # prepare test chunks
    test_chunks = ["chunk1", "chunk2", "chunk3"]
    
    # collect yielded chunks
    yielded_chunks = []
    async for chunk in stream_agent_response(test_chunks):
        yielded_chunks.append(chunk)
    
    # assert all chunks were yielded
    assert yielded_chunks == test_chunks


@pytest.mark.asyncio
async def test_main_exception_group_httpx_error(mock_env, monkeypatch):
    """Test that exception groups with httpx.ConnectError are handled in main context."""
    # mock the agent graph to raise an `ExceptionGroup` with `httpx.ConnectError`
    mock_graph = MagicMock()
    
    async def mock_astream_events(*args, **kwargs):
        raise ExceptionGroup("Connection errors", [
            httpx.ConnectError("Connection failed"),
            ValueError("Some other error")
        ])
        yield  # make it a generator
    
    mock_graph.astream_events = mock_astream_events
    
    monkeypatch.setattr("main.agent_graph", mock_graph)
    
    # import after mocking
    from main import generate_final_report_from_agent, AgentState
    
    initial_state = AgentState(
        project_name="test",
        requirements_content="requests==2.0.0",
        auth_token="test-token",
        analysis_json={},
        final_report=""
    )
    
    # assert that a HTTP 503 error is thrown (httpx.ConnectError should be caught)
    with pytest.raises(HTTPException) as exc_info:
        await generate_final_report_from_agent(initial_state)
    
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_main_exception_group_generic(mock_env, monkeypatch):
    """Test that exception groups with generic exceptions are handled in main context."""
    # mock the agent graph to raise an ExceptionGroup with generic exceptions
    mock_graph = MagicMock()
    
    async def mock_astream_events(*args, **kwargs):
        raise ExceptionGroup("Generic errors", [
            ValueError("Error 1"),
            RuntimeError("Error 2")
        ])
        yield  # make it a generator
    
    mock_graph.astream_events = mock_astream_events
    
    monkeypatch.setattr("main.agent_graph", mock_graph)
    
    # import after mocking
    from main import generate_final_report_from_agent, AgentState
    
    initial_state = AgentState(
        project_name="test",
        requirements_content="requests==2.0.0",
        auth_token="test-token",
        analysis_json={},
        final_report=""
    )
    
    # assert that a HTTP 500 error is thrown
    with pytest.raises(HTTPException) as exc_info:
        await generate_final_report_from_agent(initial_state)
    
    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_generate_report_empty_chunks_raises_500(mock_env, monkeypatch):
    """Verify that empty chunks from agent raises HTTPException(500)."""
    # mock generate_final_report_from_agent to return empty list
    async def mock_generate_empty(*args, **kwargs):
        return []
    
    monkeypatch.setattr("main.generate_final_report_from_agent", mock_generate_empty)
    
    # import after mocking
    from main import app
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    
    # prepare request
    files = {"requirements_file": ("requirements.txt", b"requests==2.0.0", "text/plain")}
    data = {"project_name": "test-project"}
    headers = {"Authorization": "Bearer test-token"}
    
    # make request
    response = client.post("/generate/report", files=files, data=data, headers=headers)
    
    # assert a HTTP 500 error
    assert response.status_code == 500
    assert "An unknown error occurred" in response.json()["detail"]


@pytest.mark.asyncio
async def test_generate_report_none_chunks_raises_500(mock_env, monkeypatch):
    """Verify that None chunks from agent raises HTTPException(500)."""
    # mock generate_final_report_from_agent to return None
    async def mock_generate_none(*args, **kwargs):
        return None
    
    monkeypatch.setattr("main.generate_final_report_from_agent", mock_generate_none)
    
    # import after mocking
    from main import app
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    
    # prepare request
    files = {"requirements_file": ("requirements.txt", b"requests==2.0.0", "text/plain")}
    data = {"project_name": "test-project"}
    headers = {"Authorization": "Bearer test-token"}
    
    # make request
    response = client.post("/generate/report", files=files, data=data, headers=headers)
    
    # assert 500 error
    assert response.status_code == 500
    assert "An unknown error occurred" in response.json()["detail"]


@pytest.mark.asyncio
async def test_http_exception_propagation(mock_env, monkeypatch):
    """Verify HTTPException is properly propagated without being wrapped."""
    # mock generate_final_report_from_agent to raise HTTPException
    async def mock_generate_http_exception(*args, **kwargs):
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    monkeypatch.setattr("main.generate_final_report_from_agent", mock_generate_http_exception)
    
    # import after mocking
    from main import app
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    
    # prepare request
    files = {"requirements_file": ("requirements.txt", b"requests==2.0.0", "text/plain")}
    data = {"project_name": "test-project"}
    headers = {"Authorization": "Bearer test-token"}
    
    # make request
    response = client.post("/generate/report", files=files, data=data, headers=headers)
    
    # assert a HTTP 503 error is propagated
    assert response.status_code == 503
    assert "Service unavailable" in response.json()["detail"]
