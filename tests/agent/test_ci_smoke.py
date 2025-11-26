"""
Smoke test for the AI agent service to run in CI.
This test mocks all external dependencies (LLM and MCP server) to ensure
the agent graph can be initialized and executed without real API keys.
"""
import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import asyncio
import sys
import os

# Add the agent directory to the path so we can import the agent module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))


class TestAgentSmoke(unittest.TestCase):
    """Smoke test for the agent graph execution."""

    def setUp(self):
        """Set up test fixtures."""
        # Set dummy environment variables to avoid RuntimeError
        os.environ['OPENAI_API_KEY'] = 'dummy-key-for-testing'
        os.environ['MCP_SERVER_HOST'] = 'http://localhost'
        os.environ['MCP_SERVER_PORT'] = '8000'

    def tearDown(self):
        """Clean up after tests."""
        # Clean up environment variables
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']

    @patch('langchain_mcp_adapters.client.MultiServerMCPClient')
    @patch('langchain_openai.ChatOpenAI')
    @patch('langchain_core.prompts.ChatPromptTemplate.from_messages')
    def test_agent_smoke_test(self, mock_prompt_template, mock_chat_openai, mock_mcp_client):
        """
        Smoke test: Initialize agent graph, pass dummy input, assert no crash.
        Mocks both the LLM client and MCP server to avoid external dependencies.
        """
        # Mock the LLM
        mock_llm_instance = MagicMock()
        mock_chat_openai.return_value = mock_llm_instance
        
        # Mock the prompt template and chain
        mock_prompt = MagicMock()
        mock_prompt_template.return_value = mock_prompt
        
        # Create a mock chain that returns our dummy response
        # The chain needs to support chaining with | operator and have an async ainvoke
        mock_parser = MagicMock()
        mock_parser.ainvoke = AsyncMock(return_value="Mocked Summary")
        
        mock_llm_chain = MagicMock()
        mock_llm_chain.__or__ = MagicMock(return_value=mock_parser)
        
        # Mock the pipe operator to return our mock chain
        mock_prompt.__or__ = MagicMock(return_value=mock_llm_chain)
        
        # Mock the MCP client to return dummy analysis data
        mock_session = AsyncMock()
        mock_tool_result = MagicMock()
        mock_text_content = MagicMock()
        mock_text_content.text = '{"packages": [{"name": "test-package", "license": "MIT"}]}'
        mock_tool_result.content = [mock_text_content]
        mock_session.call_tool = AsyncMock(return_value=mock_tool_result)
        
        # Set up the context manager for the MCP client session
        mock_client_instance = MagicMock()
        mock_client_instance.session = MagicMock(return_value=mock_session)
        mock_client_instance.session().__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_instance.session().__aexit__ = AsyncMock(return_value=None)
        mock_mcp_client.return_value = mock_client_instance

        # Import and build the agent graph (after mocks are set up)
        from agent.agent import build_agent_graph
        
        graph = build_agent_graph()
        
        # Prepare dummy input
        dummy_input = {
            "project_name": "test-project",
            "requirements_content": "requests==2.28.0\nnumpy==1.24.0",
            "auth_token": "dummy-token"
        }
        
        # Run the agent graph
        async def run_test():
            result = await graph.ainvoke(dummy_input)
            return result
        
        # Execute the async test
        result = asyncio.run(run_test())
        
        # Assertions
        self.assertIsNotNone(result, "Agent should return a result")
        self.assertIn("final_report", result, "Result should contain final_report")
        self.assertEqual(result["final_report"], "Mocked Summary", 
                        "Final report should match mocked LLM response")
        
        # Verify that the mocks were called
        mock_chat_openai.assert_called_once()
        mock_mcp_client.assert_called_once()
        
        print("✓ Smoke test passed: Agent initialized and executed without crashing")
        print(f"✓ Agent returned expected mocked response: '{result['final_report']}'")


if __name__ == '__main__':
    unittest.main()
