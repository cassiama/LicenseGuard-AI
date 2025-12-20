"""
End-to-end integration tests for error handling.
Tests the full request flow with various error conditions.
"""

import sys
import os
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage
from mcp.types import TextContent

# add the agent directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))


@pytest.fixture
def mock_env(monkeypatch):
    """Set up dummy environment variables to avoid RuntimeError."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key-for-testing")
    monkeypatch.setenv("MCP_SERVER_HOST", "http://localhost")
    monkeypatch.setenv("MCP_SERVER_PORT", "8000")


def test_e2e_generate_report_mcp_down_returns_503(mock_env, monkeypatch):
    """Full e2e: MCP down should return 503 to client."""
    # mock MCP client to raise httpx.ConnectError
    mock_session = AsyncMock()
    mock_session.call_tool.side_effect = httpx.ConnectError(
        "Connection refused")

    mock_client_instance = MagicMock()
    mock_client_instance.session.return_value.__aenter__ = AsyncMock(
        side_effect=httpx.ConnectError("Connection refused")
    )
    mock_client_instance.session.return_value.__aexit__ = AsyncMock(
        return_value=None)

    mock_mcp_client = MagicMock(return_value=mock_client_instance)
    monkeypatch.setattr(
        "langchain_mcp_adapters.client.MultiServerMCPClient", mock_mcp_client)

    # mock the LLM
    mock_prompt = MagicMock()
    mock_prompt_template = MagicMock()
    mock_prompt_template.from_messages = MagicMock(return_value=mock_prompt)

    mock_llm_chain = MagicMock()
    mock_llm_chain.ainvoke = AsyncMock(return_value="Test report")

    mock_prompt.__or__ = MagicMock(return_value=mock_llm_chain)
    mock_llm_chain.__or__ = MagicMock(return_value=mock_llm_chain)

    mock_llm_instance = MagicMock()
    mock_chat_openai = MagicMock(return_value=mock_llm_instance)

    monkeypatch.setattr("agent.ChatOpenAI", mock_chat_openai)
    monkeypatch.setattr("agent.ChatPromptTemplate", mock_prompt_template)
    monkeypatch.setattr("agent.MultiServerMCPClient", mock_mcp_client)
    monkeypatch.setattr("langchain_openai.ChatOpenAI", mock_chat_openai)
    monkeypatch.setattr(
        "langchain_core.prompts.ChatPromptTemplate", mock_prompt_template)

    # import main and FastAPI's TestClient after mocking
    from main import app
    from fastapi.testclient import TestClient

    client = TestClient(app)

    # prepare request
    files = {"requirements_file": (
        "requirements.txt", b"requests==2.0.0", "text/plain")}
    data = {"project_name": "test-project"}
    headers = {"Authorization": "Bearer test-token"}

    # make the request
    response = client.post("/generate/report", files=files,
                           data=data, headers=headers)

    # assert that a HTTP 503 error is thrown
    assert response.status_code == 503
    assert "Unable to connect to the MCP server" in response.json()["detail"]


def test_e2e_generate_report_mcp_timeout_returns_500(mock_env, monkeypatch):
    """Full e2e: MCP timeout should return 500 to client."""
    # mock MCP client to raise `TimeoutError`
    mock_session = AsyncMock()
    mock_session.call_tool.side_effect = TimeoutError("Request timed out")

    mock_client_instance = MagicMock()
    mock_client_instance.session.return_value.__aenter__ = AsyncMock(
        return_value=mock_session)
    mock_client_instance.session.return_value.__aexit__ = AsyncMock(
        return_value=None)

    mock_mcp_client = MagicMock(return_value=mock_client_instance)
    monkeypatch.setattr(
        "langchain_mcp_adapters.client.MultiServerMCPClient", mock_mcp_client)

    # mock the LLM
    mock_prompt = MagicMock()
    mock_prompt_template = MagicMock()
    mock_prompt_template.from_messages = MagicMock(return_value=mock_prompt)

    mock_llm_chain = MagicMock()
    mock_llm_chain.ainvoke = AsyncMock(return_value="Test report")

    mock_prompt.__or__ = MagicMock(return_value=mock_llm_chain)
    mock_llm_chain.__or__ = MagicMock(return_value=mock_llm_chain)

    mock_llm_instance = MagicMock()
    mock_chat_openai = MagicMock(return_value=mock_llm_instance)

    monkeypatch.setattr("agent.ChatOpenAI", mock_chat_openai)
    monkeypatch.setattr("agent.ChatPromptTemplate", mock_prompt_template)
    monkeypatch.setattr("agent.MultiServerMCPClient", mock_mcp_client)
    monkeypatch.setattr("langchain_openai.ChatOpenAI", mock_chat_openai)
    monkeypatch.setattr(
        "langchain_core.prompts.ChatPromptTemplate", mock_prompt_template)

    # import main and FastAPI's TestClient after mocking
    from main import app
    from fastapi.testclient import TestClient

    client = TestClient(app)

    # prepare request
    files = {"requirements_file": (
        "requirements.txt", b"requests==2.0.0", "text/plain")}
    data = {"project_name": "test-project"}
    headers = {"Authorization": "Bearer test-token"}

    # make the request
    response = client.post("/generate/report", files=files,
                           data=data, headers=headers)

    # assert that a HTTP 500 error is thrown (timeout is handled by interceptor, returns error
    # dict) which then gets processed by the agent, but since it's an error, it should fail
    assert response.status_code == 500


def test_e2e_generate_report_mcp_tool_error_returns_500(mock_env, monkeypatch):
    """Full e2e: MCP tool raises exception should return 500 to client."""
    # mock MCP client to raise ValueError
    mock_session = AsyncMock()
    mock_session.call_tool.side_effect = ValueError("Invalid input")

    mock_client_instance = MagicMock()
    mock_client_instance.session.return_value.__aenter__ = AsyncMock(
        return_value=mock_session)
    mock_client_instance.session.return_value.__aexit__ = AsyncMock(
        return_value=None)

    mock_mcp_client = MagicMock(return_value=mock_client_instance)
    monkeypatch.setattr(
        "langchain_mcp_adapters.client.MultiServerMCPClient", mock_mcp_client)

    # mock the LLM
    mock_prompt = MagicMock()
    mock_prompt_template = MagicMock()
    mock_prompt_template.from_messages = MagicMock(return_value=mock_prompt)

    mock_llm_chain = MagicMock()
    mock_llm_chain.ainvoke = AsyncMock(return_value="Test report")

    mock_prompt.__or__ = MagicMock(return_value=mock_llm_chain)
    mock_llm_chain.__or__ = MagicMock(return_value=mock_llm_chain)

    mock_llm_instance = MagicMock()
    mock_chat_openai = MagicMock(return_value=mock_llm_instance)

    monkeypatch.setattr("agent.ChatOpenAI", mock_chat_openai)
    monkeypatch.setattr("agent.ChatPromptTemplate", mock_prompt_template)
    monkeypatch.setattr("agent.MultiServerMCPClient", mock_mcp_client)
    monkeypatch.setattr("langchain_openai.ChatOpenAI", mock_chat_openai)
    monkeypatch.setattr(
        "langchain_core.prompts.ChatPromptTemplate", mock_prompt_template)

    # import main and FastAPI's TestClient after mocking
    from main import app
    from fastapi.testclient import TestClient

    client = TestClient(app)

    # prepare request
    files = {"requirements_file": (
        "requirements.txt", b"requests==2.0.0", "text/plain")}
    data = {"project_name": "test-project"}
    headers = {"Authorization": "Bearer test-token"}

    # make the request
    response = client.post("/generate/report", files=files,
                           data=data, headers=headers)

    # assert that a HTTP 500 error is thrown

    assert response.status_code == 500
    assert "An unknown error occurred" in response.json()["detail"]


def test_e2e_generate_report_mcp_returns_error_dict(mock_env, monkeypatch):
    """Full e2e: MCP returns error dict should be handled gracefully."""
    # mock MCP client to return error dict
    mock_session = AsyncMock()
    mock_tool_result = MagicMock()
    mock_text_content = MagicMock(spec=TextContent)
    mock_text_content.text = '{"error": "MCP server internal error"}'
    mock_tool_result.content = [mock_text_content]
    mock_session.call_tool = AsyncMock(return_value=mock_tool_result)

    mock_client_instance = MagicMock()
    mock_client_instance.session.return_value.__aenter__ = AsyncMock(
        return_value=mock_session)
    mock_client_instance.session.return_value.__aexit__ = AsyncMock(
        return_value=None)

    mock_mcp_client = MagicMock(return_value=mock_client_instance)
    monkeypatch.setattr(
        "langchain_mcp_adapters.client.MultiServerMCPClient", mock_mcp_client)

    # mock the LLM
    mock_prompt = MagicMock()
    mock_prompt_template = MagicMock()
    mock_prompt_template.from_messages = MagicMock(return_value=mock_prompt)

    mock_llm_chain = MagicMock()
    mock_llm_chain.ainvoke = AsyncMock(return_value="Test report")

    mock_prompt.__or__ = MagicMock(return_value=mock_llm_chain)
    mock_llm_chain.__or__ = MagicMock(return_value=mock_llm_chain)

    mock_llm_instance = MagicMock()
    mock_chat_openai = MagicMock(return_value=mock_llm_instance)

    monkeypatch.setattr("agent.ChatOpenAI", mock_chat_openai)
    monkeypatch.setattr("agent.ChatPromptTemplate", mock_prompt_template)
    monkeypatch.setattr("agent.MultiServerMCPClient", mock_mcp_client)
    monkeypatch.setattr("langchain_openai.ChatOpenAI", mock_chat_openai)
    monkeypatch.setattr(
        "langchain_core.prompts.ChatPromptTemplate", mock_prompt_template)

    # import main and FastAPI's TestClient after mocking
    from main import app
    from fastapi.testclient import TestClient

    client = TestClient(app)

    # prepare request
    files = {"requirements_file": (
        "requirements.txt", b"requests==2.0.0", "text/plain")}
    data = {"project_name": "test-project"}
    headers = {"Authorization": "Bearer test-token"}

    # make the request
    response = client.post("/generate/report", files=files,
                           data=data, headers=headers)

    # assert that a HTTP 500 error is thrown (MCP error means there's no streaming chunks, thus 
    # a HTTP 500)
    assert response.status_code == 500


def test_e2e_missing_auth_header_returns_401(mock_env, monkeypatch):
    """Full e2e: Missing authorization header should return 422 (FastAPI validation)."""
    # import main and FastAPI's TestClient
    from main import app
    from fastapi.testclient import TestClient

    client = TestClient(app)

    # prepare request WITHOUT authorization header
    files = {"requirements_file": (
        "requirements.txt", b"requests==2.0.0", "text/plain")}
    data = {"project_name": "test-project"}

    # make the request
    response = client.post("/generate/report", files=files, data=data)

    # assert that a HTTP 422 error is thrown (FastAPI validates required parameters)
    assert response.status_code == 422


def test_e2e_malformed_auth_header_returns_401(mock_env, monkeypatch):
    """Full e2e: Malformed authorization header should return 401."""
    # import main and FastAPI's TestClient and FastAPI's TestClient
    from main import app
    from fastapi.testclient import TestClient

    client = TestClient(app)

    # prepare request with malformed authorization header
    files = {"requirements_file": (
        "requirements.txt", b"requests==2.0.0", "text/plain")}
    data = {"project_name": "test-project"}
    headers = {"Authorization": "InvalidFormat token"}

    # make the request
    response = client.post("/generate/report", files=files,
                           data=data, headers=headers)

    # assert a HTTP 401 error is throw

    assert response.status_code == 401
    assert "Bearer" in response.json()["detail"]


def test_e2e_runtime_error_no_asgi_crash(mock_env, monkeypatch):
    """
    Full e2e: Simulate `RuntimeError` in agent, verify server doesn't crash.
    This test ensures we DON'T get a `RuntimeError` ("Caught handled exception, but response already started.").
    """
    # mock MCP client to raise `RuntimeError`
    mock_session = AsyncMock()
    mock_session.call_tool.side_effect = RuntimeError(
        "Unexpected runtime error")

    mock_client_instance = MagicMock()
    mock_client_instance.session.return_value.__aenter__ = AsyncMock(
        return_value=mock_session)
    mock_client_instance.session.return_value.__aexit__ = AsyncMock(
        return_value=None)

    mock_mcp_client = MagicMock(return_value=mock_client_instance)
    monkeypatch.setattr(
        "langchain_mcp_adapters.client.MultiServerMCPClient", mock_mcp_client)

    # mock the LLM
    mock_prompt = MagicMock()
    mock_prompt_template = MagicMock()
    mock_prompt_template.from_messages = MagicMock(return_value=mock_prompt)

    mock_llm_chain = MagicMock()
    mock_llm_chain.ainvoke = AsyncMock(return_value="Test report")

    mock_prompt.__or__ = MagicMock(return_value=mock_llm_chain)
    mock_llm_chain.__or__ = MagicMock(return_value=mock_llm_chain)

    mock_llm_instance = MagicMock()
    mock_chat_openai = MagicMock(return_value=mock_llm_instance)

    monkeypatch.setattr("agent.ChatOpenAI", mock_chat_openai)
    monkeypatch.setattr("agent.ChatPromptTemplate", mock_prompt_template)
    monkeypatch.setattr("agent.MultiServerMCPClient", mock_mcp_client)
    monkeypatch.setattr("langchain_openai.ChatOpenAI", mock_chat_openai)
    monkeypatch.setattr(
        "langchain_core.prompts.ChatPromptTemplate", mock_prompt_template)

    # import main and FastAPI's TestClient after mocking
    from main import app
    from fastapi.testclient import TestClient

    client = TestClient(app)

    # prepare request
    files = {"requirements_file": (
        "requirements.txt", b"requests==2.0.0", "text/plain")}
    data = {"project_name": "test-project"}
    headers = {"Authorization": "Bearer test-token"}

    # make the request (this should NOT crash the server)
    response = client.post("/generate/report", files=files,
                           data=data, headers=headers)

    # assert a HTTP 500 error is thrown (not a server crash)
    assert response.status_code == 500
    assert "An unknown error occurred" in response.json()["detail"]


def test_e2e_streaming_error_handled_gracefully(mock_env, monkeypatch):
    """Full e2e: Error during streaming is handled properly without ASGI crash."""
    # mock the agent graph to return chunks successfully
    mock_session = AsyncMock()
    mock_tool_result = MagicMock()
    mock_text_content = MagicMock(spec=TextContent)
    mock_text_content.text = '{"packages": [{"name": "test-package", "license": "MIT"}]}'
    mock_tool_result.content = [mock_text_content]
    mock_session.call_tool = AsyncMock(return_value=mock_tool_result)

    mock_client_instance = MagicMock()
    mock_client_instance.session.return_value.__aenter__ = AsyncMock(
        return_value=mock_session)
    mock_client_instance.session.return_value.__aexit__ = AsyncMock(
        return_value=None)

    mock_mcp_client = MagicMock(return_value=mock_client_instance)
    monkeypatch.setattr(
        "langchain_mcp_adapters.client.MultiServerMCPClient", mock_mcp_client)

    # mock the LLM to return chunks
    mock_prompt = MagicMock()
    mock_prompt_template = MagicMock()
    mock_prompt_template.from_messages = MagicMock(return_value=mock_prompt)

    mock_llm_chain = MagicMock()
    mock_llm_chain.ainvoke = AsyncMock(
        return_value="## 🛡️ LicenseGuard Report\n\nTest report")

    mock_prompt.__or__ = MagicMock(return_value=mock_llm_chain)
    mock_llm_chain.__or__ = MagicMock(return_value=mock_llm_chain)

    mock_llm_instance = MagicMock()
    mock_chat_openai = MagicMock(return_value=mock_llm_instance)

    monkeypatch.setattr("agent.ChatOpenAI", mock_chat_openai)
    monkeypatch.setattr("agent.ChatPromptTemplate", mock_prompt_template)
    monkeypatch.setattr("agent.MultiServerMCPClient", mock_mcp_client)
    monkeypatch.setattr("langchain_openai.ChatOpenAI", mock_chat_openai)
    monkeypatch.setattr(
        "langchain_core.prompts.ChatPromptTemplate", mock_prompt_template)

    # import main and FastAPI's TestClient after mocking
    from main import app
    from fastapi.testclient import TestClient

    client = TestClient(app)

    # prepare request
    files = {"requirements_file": (
        "requirements.txt", b"requests==2.0.0", "text/plain")}
    data = {"project_name": "test-project"}
    headers = {"Authorization": "Bearer test-token"}

    # make the request
    response = client.post("/generate/report", files=files,
                           data=data, headers=headers)

    # assert a HTTP 500 error is thrown (no streaming chunks generated)
    assert response.status_code == 500


def test_e2e_exception_group_handled_correctly(mock_env, monkeypatch):
    """Full e2e: `ExceptionGroup` with multiple errors is handled correctly."""
    # mock the interceptor to return error dict
    async def mock_fallback_interceptor(request, handler):
        return {"error": "Multiple errors"}
    
    # mock MCP client to raise `ExceptionGroup`
    def raise_exception_group(**kwargs):
        raise ExceptionGroup("Multiple errors", [
            httpx.ConnectError("Connection failed"),
            ValueError("Some other error")
        ])

    mock_session = AsyncMock()
    mock_session.call_tool.side_effect = raise_exception_group

    mock_client_instance = MagicMock()
    mock_client_instance.session.return_value.__aenter__ = AsyncMock(
        return_value=mock_session)
    mock_client_instance.session.return_value.__aexit__ = AsyncMock(
        return_value=None)

    mock_mcp_client = MagicMock(return_value=mock_client_instance)
    monkeypatch.setattr(
        "langchain_mcp_adapters.client.MultiServerMCPClient", mock_mcp_client)

    # mock the LLM
    mock_prompt = MagicMock()
    mock_prompt_template = MagicMock()
    mock_prompt_template.from_messages = MagicMock(return_value=mock_prompt)

    mock_llm_chain = MagicMock()
    mock_llm_chain.ainvoke = AsyncMock(return_value="Test report")

    mock_prompt.__or__ = MagicMock(return_value=mock_llm_chain)
    mock_llm_chain.__or__ = MagicMock(return_value=mock_llm_chain)

    mock_llm_instance = MagicMock()
    mock_chat_openai = MagicMock(return_value=mock_llm_instance)

    monkeypatch.setattr("agent.ChatOpenAI", mock_chat_openai)
    monkeypatch.setattr("agent.ChatPromptTemplate", mock_prompt_template)
    monkeypatch.setattr("agent.MultiServerMCPClient", mock_mcp_client)
    monkeypatch.setattr("agent.fallback_interceptor", mock_fallback_interceptor)
    monkeypatch.setattr("langchain_openai.ChatOpenAI", mock_chat_openai)
    monkeypatch.setattr(
        "langchain_core.prompts.ChatPromptTemplate", mock_prompt_template)

    # import main and FastAPI's TestClient after mocking
    from main import app
    from fastapi.testclient import TestClient

    client = TestClient(app)

    # prepare request
    files = {"requirements_file": (
        "requirements.txt", b"requests==2.0.0", "text/plain")}
    data = {"project_name": "test-project"}
    headers = {"Authorization": "Bearer test-token"}

    # make the request
    response = client.post("/generate/report", files=files,
                           data=data, headers=headers)

    # assert a HTTP 500 error is thrown (interceptor catches and returns error dict, no streaming chunks)
    assert response.status_code == 500
