from deepdec import SystemOrchestrator
import json
from unittest.mock import patch
import pytest

@pytest.fixture
def mock_config():
    return {
    "api": {
        "api_type": "openrouter",
        "api_key": "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",
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
def test_retry_logic(mock_api, tmp_path, mock_config):
    mock_api.side_effect = [
        "Bad Response", "[MARK] NO",
        "Better Response", "[MARK] NO",
        "Good Response", "[MARK] YES",
        "Aggregated Result", "[MARK] YES"
    ]
    orchestrator = (
        SystemOrchestrator.Builder(mock_config)
        .with_workflow("analysis", "analysis", "eval_analysis")
        .with_workflow("aggregator", "aggregator", "eval_aggregator")
        .build()
    )
    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.json"
    input_file.write_text("problematic code")
    orchestrator.execute(str(input_file), str(output_file))
    with open(output_file) as f:
        result = json.load(f)
        assert result["aggregated"] == "Aggregated Result"