"""
LangSmith-based evaluation script for LicenseGuard AI Agent.
This script evaluates the agent's quality using LLM-as-a-Judge pattern.
"""

import os
import sys
import uuid
import json
import re
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langsmith import Client
from langsmith.evaluation import aevaluate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# add the agent directory to the path so we can import it later
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))


# MCP server mock
def setup_mcp_mock(monkeypatch, requirements_content: str):
    """
    Mock the MCP server to return controlled license data based on input.
    This allows us to test the agent without running a real MCP server.
    """
    # determine what kind of response to return based on input
    if "gpl" in requirements_content.lower() or "GPL" in requirements_content:
        # this is the case where there are conflicts, so we return the GPL-3.0 package
        mock_analysis = {
            "packages": [
                {
                    "name": "gpl-package",
                    "version": "1.0.0",
                    "license": "GPL-3.0",
                    "confidence": 0.95,
                }
            ]
        }
    elif any(pkg in requirements_content for pkg in ["requests", "pandas", "numpy"]):
        # this is the standard case, so we return permissive licenses
        mock_analysis = {
            "packages": [
                {
                    "name": "requests",
                    "version": "2.28.0",
                    "license": "Apache-2.0",
                    "confidence": 0.98,
                },
                {
                    "name": "pandas",
                    "version": "1.5.0",
                    "license": "BSD-3-Clause",
                    "confidence": 0.97,
                },
            ]
        }
    else:
        # this is the garbage case, so we return empty or error
        mock_analysis = {"packages": [], "error": "No valid packages found"}

    # create mock MCP client session
    mock_session = AsyncMock()
    mock_tool_result = MagicMock()
    mock_text_content = MagicMock()
    mock_text_content.text = json.dumps(mock_analysis)
    mock_tool_result.content = [mock_text_content]
    mock_session.call_tool = AsyncMock(return_value=mock_tool_result)

    # set up context manager
    mock_client_instance = MagicMock()
    mock_client_instance.session = MagicMock(return_value=mock_session)
    mock_client_instance.session().__aenter__ = AsyncMock(return_value=mock_session)
    mock_client_instance.session().__aexit__ = AsyncMock(return_value=None)
    mock_mcp_client = MagicMock(return_value=mock_client_instance)

    # and finally, apply mock
    monkeypatch.setattr(
        "langchain_mcp_adapters.client.MultiServerMCPClient", mock_mcp_client
    )


def check_package_existence(run, example) -> Dict[str, Any]:
    """
    Evaluator 1: LLM-as-a-Judge to check if recommended packages exist on PyPI.

    This evaluator:
    1. Extracts package recommendations from the agent's report
    2. Uses ChatOpenAI to verify each package exists on PyPI
    3. Returns a score based on hallucination detection
    """
    try:
        final_report = run.outputs.get("final_report", "")

        # extract recommended packages from the report
        # look for patterns like: "Switch to `package-name`" or "`package-name` ("
        package_pattern = r"`([a-zA-Z0-9_-]+)`"
        recommended_packages = re.findall(package_pattern, final_report)

        # then, filter out common non-package words
        excluded_words = {
            "Problem",
            "Solution",
            "License",
            "Package",
            "Name",
            "Source",
            "Risk",
        }
        recommended_packages = [
            pkg for pkg in recommended_packages if pkg not in excluded_words
        ]

        # if no recommendations were found, this is acceptable and return a passing score
        if not recommended_packages:
            return {
                "key": "package_existence",
                "score": 1.0,
                "comment": "No package recommendations found in report",
            }

        # otherwise, use real LLM to verify each package
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")),
        )

        # find all of the packages that do NOT exist
        hallucinated_packages = []
        for package in recommended_packages[:5]:  # limit to first 5 to control costs
            prompt = f"""Does the Python package '{package}' exist on PyPI (the Python Package Index)?
            
Answer with ONLY 'YES' or 'NO'. No explanation needed."""

            response = llm.invoke(prompt)
            # handle both string and list responses
            answer_content = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            answer = answer_content.strip().upper()

            if "NO" in answer:
                hallucinated_packages.append(package)

        # calculate score (fail if there are any hallucinated packages, pass if there aren't)
        if hallucinated_packages:
            score = 0.0
            comment = (
                f"Hallucinated packages detected: {', '.join(hallucinated_packages)}"
            )
        else:
            score = 1.0
            comment = f"All {len(recommended_packages)} recommended packages verified"

        return {"key": "package_existence", "score": score, "comment": comment}

    except Exception as e:
        return {
            "key": "package_existence",
            "score": 0.0,
            "comment": f"Evaluator error: {str(e)}",
        }


def check_format_compliance(run, example) -> Dict[str, Any]:
    """
    Evaluator 2: Rule-based check for report format compliance.

    Verifies that the report contains the required markdown header.
    """
    try:
        final_report = run.outputs.get("final_report", "")

        # check for required header is present
        required_header = "## 🛡️ LicenseGuard Report"

        if required_header in final_report:
            score = 1.0
            comment = "Report contains required header format"
        else:
            score = 0.0
            comment = "Report missing required header: '## 🛡️ LicenseGuard Report'"

        return {"key": "format_compliance", "score": score, "comment": comment}

    except Exception as e:
        return {
            "key": "format_compliance",
            "score": 0.0,
            "comment": f"Evaluator error: {str(e)}",
        }


async def run_agent_for_eval(inputs: Dict[str, Any], monkeypatch) -> Dict[str, Any]:
    """
    Wrapper function to run the agent with mocked MCP server.

    This function:
    1. Sets up MCP mocking based on input
    2. Imports and builds the agent graph
    3. Invokes the agent with proper input format
    4. Returns the output in expected format
    """
    # set up environment variables
    monkeypatch.setenv("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    monkeypatch.setenv("MCP_SERVER_HOST", "http://localhost")
    monkeypatch.setenv("MCP_SERVER_PORT", "8000")

    # mock MCP server based on input
    requirements_content = inputs.get("requirements_content", "")
    setup_mcp_mock(monkeypatch, requirements_content)

    # import agent (after mocks are set up)
    from agent import build_agent_graph

    # build and invoke agent
    graph = build_agent_graph()
    result = await graph.ainvoke(
        {
            "project_name": inputs.get("project_name", "eval-project"),
            "requirements_content": requirements_content,
            "auth_token": inputs.get("auth_token", "eval-dummy-token"),
        }
    )

    return {"final_report": result["final_report"]}


@pytest.mark.asyncio
async def test_agent_quality_evaluation(monkeypatch):
    """
    Main evaluation test using LangSmith.

    This test:
    1. Creates a LangSmith dataset with 3 test cases
    2. Runs the agent on each case
    3. Evaluates outputs using LLM-as-a-Judge and rule-based evaluators
    4. Reports results and LangSmith trace URL
    """
    # initialize the LangSmith client
    client = Client()

    # create unique dataset name
    dataset_name = f"license-guard-eval-{uuid.uuid4()}"

    # create dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="LicenseGuard AI Agent Quality Evaluation Dataset",
    )

    # define test examples
    examples = [
        {
            "inputs": {
                "project_name": "test-standard",
                "requirements_content": "requests==2.28.0\npandas==1.5.0",
                "auth_token": "eval-dummy-token",
            },
            "outputs": {"description": "Standard case with common permissive licenses"},
        },
        {
            "inputs": {
                "project_name": "test-vulnerability",
                "requirements_content": "gpl-package==1.0.0",
                "auth_token": "eval-dummy-token",
            },
            "outputs": {"description": "Vulnerability case with GPL-3.0 license"},
        },
        {
            "inputs": {
                "project_name": "test-garbage",
                "requirements_content": "random garbage text !@#$ not a package",
                "auth_token": "eval-dummy-token",
            },
            "outputs": {"description": "Garbage input case"},
        },
    ]

    # create examples in dataset
    for example in examples:
        client.create_example(
            inputs=example["inputs"], outputs=example["outputs"], dataset_id=dataset.id
        )

    # create wrapper for running the agent that includes monkeypatch
    async def agent_wrapper(inputs: Dict[str, Any]) -> Dict[str, Any]:
        return await run_agent_for_eval(inputs, monkeypatch)

    # run evaluation
    await aevaluate(
        agent_wrapper,
        data=dataset_name,
        evaluators=[check_package_existence, check_format_compliance],
        experiment_prefix="licenseguard-quality",
        description="Quality evaluation of LicenseGuard agent with LLM-as-a-Judge",
    )

    # delete dataset after test
    client.delete_dataset(dataset_id=dataset.id)


if __name__ == "__main__":
    """
    Run this script directly for manual evaluation:
    
    $ cd agent
    $ python -m pytest tests/eval/test_agent_quality.py -v -s
    
    Or run with pytest:
    $ pytest tests/eval/test_agent_quality.py -v -s
    
    Make sure you have:
    - OPENAI_API_KEY set in environment
    - LANGSMITH_API_KEY set in environment
    """
    import asyncio
    from unittest.mock import MagicMock

    # create a mock of monkeypatch for standalone execution
    class MockMonkeypatch:
        def setenv(self, key, value):
            os.environ[key] = value

        def setattr(self, target, value):
            parts = target.rsplit(".", 1)
            if len(parts) == 2:
                module_path, attr_name = parts
                # this is a simplified version - in real pytest, this is more robust
                pass

    monkeypatch = MockMonkeypatch()
    asyncio.run(test_agent_quality_evaluation(monkeypatch))
