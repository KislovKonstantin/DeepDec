from deepdec import SystemOrchestrator
import json
from unittest.mock import patch
import pytest

@pytest.fixture
def mock_config():
    return {
    "api": {
        "api_type": "openrouter",
        "api_key": "<api_key>",
        "api_url": "https://openrouter.ai/api/v1/chat/completions"
    },
    "models": {
        "analysis": "deepseek/deepseek-r1-distill-llama-70b:free",
        "eval_analysis": "deepseek/deepseek-r1-distill-llama-70b:free",
        "commentary": "deepseek/deepseek-r1-distill-llama-70b:free",
        "eval_comment": "deepseek/deepseek-r1-distill-llama-70b:free",
        "reconstruct": "deepseek/deepseek-r1-distill-llama-70b:free",
        "eval_reconstruct": "deepseek/deepseek-r1-distill-llama-70b:free",
        "aggregator": "deepseek/deepseek-r1-distill-llama-70b:free",
        "eval_aggregator": "deepseek/deepseek-r1-distill-llama-70b:free"
    },
    "prompts": {
        "analysis": "code_alalyst.txt",
        "eval_analysis": "eval_analysis.txt",
        "commentary": "commentator.txt",
        "eval_comment": "eval_comments.txt",
        "reconstruct": "reconstructor.txt",
        "eval_reconstruct": "eval_orig.txt",
        "aggregator": "aggregator.txt",
        "eval_aggregator": "eval_aggr.txt"
    }
    }

@patch("deepdec.APIGateway.send_request")
def test_output_file_structure(mock_api, tmp_path, mock_config):
    mock_api.side_effect = [
        "Analysis Response", "[MARK] YES",
        "Comment Response", "[MARK] YES",
        "Reconstruct Response", "[MARK] YES",
        "Aggregated Result", "[MARK] YES"
    ]
    orchestrator = (
        SystemOrchestrator.Builder(mock_config)
        .with_workflow("analysis", "analysis", "eval_analysis")
        .with_workflow("commentary", "commentary", "eval_comment")
        .with_workflow("reconstruction", "reconstruct", "eval_reconstruct")
        .with_workflow("aggregator", "aggregator", "eval_aggregator")
        .build()
    )
    output_file = tmp_path / "output.json"
    orchestrator.execute("test.txt", str(output_file))
    with open(output_file) as f:
        data = json.load(f)
        assert set(data.keys()) == {
            "analysis", "commentary",
            "reconstruction", "aggregated"
        }
        assert isinstance(data["aggregated"], str)
        assert data["aggregated"] == "Aggregated Result"
