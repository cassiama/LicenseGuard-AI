"""
Smoke test for the AI agent service to run in CI.
This test mocks all external dependencies (LLM and MCP server) to ensure
the agent graph can be initialized and executed without real API keys.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
import sys
import os

# Add the agent directory to the path so we can import the agent module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))


@pytest.fixture
def mock_env(monkeypatch):
    """Set up dummy environment variables to avoid RuntimeError."""
    monkeypatch.setenv('OPENAI_API_KEY', 'dummy-key-for-testing')
    monkeypatch.setenv('MCP_SERVER_HOST', 'http://localhost')
    monkeypatch.setenv('MCP_SERVER_PORT', '8000')


@pytest.fixture
def mock_dependencies(monkeypatch):
    """Mock all external dependencies (LLM and MCP server)."""
    # mock the LLM
    mock_llm_instance = MagicMock()
    mock_chat_openai = MagicMock(return_value=mock_llm_instance)
    
    # mock the agent's prompt template and chain
    mock_prompt = MagicMock()
    mock_prompt_template = MagicMock()
    mock_prompt_template.from_messages = MagicMock(return_value=mock_prompt)
    
    # create a mock agent chain that returns our dummy response
    # NOTE: this chain needs to support chaining with the "|" (pipe) operator and have an async `ainvoke` function
    mock_final_chain = MagicMock()
    mock_final_chain.ainvoke = AsyncMock(return_value="Mocked Summary")
    
    # mock the pipe operators to build the chain: prompt | llm | parser
    mock_llm_chain = MagicMock()
    mock_llm_chain.__or__ = MagicMock(return_value=mock_final_chain)
    
    mock_prompt.__or__ = MagicMock(return_value=mock_llm_chain)
    
    # mock the MCP client to return dummy analysis data (to mirror the MCP server returning a real JSON)
    mock_session = AsyncMock()
    mock_tool_result = MagicMock()
    mock_text_content = MagicMock()
    mock_text_content.text = '{"packages": [{"name": "test-package", "license": "MIT"}]}'
    mock_tool_result.content = [mock_text_content]
    mock_session.call_tool = AsyncMock(return_value=mock_tool_result)
    
    # set up the context manager for the MCP client session
    mock_client_instance = MagicMock()
    mock_client_instance.session = MagicMock(return_value=mock_session)
    mock_client_instance.session().__aenter__ = AsyncMock(return_value=mock_session)
    mock_client_instance.session().__aexit__ = AsyncMock(return_value=None)
    mock_mcp_client = MagicMock(return_value=mock_client_instance)
    
    # apply the mocks using monkeypatch
    monkeypatch.setattr('langchain_mcp_adapters.client.MultiServerMCPClient', mock_mcp_client)
    monkeypatch.setattr('langchain_openai.ChatOpenAI', mock_chat_openai)
    monkeypatch.setattr('langchain_core.prompts.ChatPromptTemplate', mock_prompt_template)
    
    return {
        'mcp_client': mock_mcp_client,
        'chat_openai': mock_chat_openai,
        'prompt_template': mock_prompt_template
    }


@pytest.mark.asyncio
async def test_agent_smoke_test(mock_env, mock_dependencies):
    """
    Smoke test: Initialize agent graph, pass dummy input, assert no crash.
    Mocks both the LLM client and MCP server to avoid external dependencies.
    """
    # import and build the agent graph
    # NOTE: this should happen after mocks are set up
    from agent.agent import build_agent_graph
    
    graph = build_agent_graph()
    
    # prepare dummy input
    dummy_input = {
        "project_name": "test-project",
        "requirements_content": "requests==2.28.0\nnumpy==1.24.0",
        "auth_token": "dummy-token"
    }
    
    # run the agent graph
    result = await graph.ainvoke(dummy_input)
    
    assert result is not None, "Agent should return a result"
    assert "final_report" in result, "Result should contain final_report"
    assert result["final_report"] == "Mocked Summary", \
        "Final report should match mocked LLM response"
    
    # verify that the mocks were called
    mock_dependencies['chat_openai'].assert_called_once()
    mock_dependencies['mcp_client'].assert_called_once()
