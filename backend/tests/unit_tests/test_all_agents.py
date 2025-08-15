from unittest.mock import Mock
from deepdec import CodeAgent, EvaluatorAgent, AggregatorAgent, ModelConfig, Path, EventSystem

def test_code_agent():
    mock_gateway = Mock()
    mock_gateway.send_request.return_value = "[ANSWER] Test Code Response"
    agent = CodeAgent(
        mock_gateway,
        ModelConfig(model_name="code-model"),
        Path("code_alalyst.txt"),
        "code_agent",
        EventSystem()
    )
    response = agent.generate_response("print('Hello')")
    assert response == "[ANSWER] Test Code Response"

def test_evaluator_agent():
    mock_gateway = Mock()
    mock_gateway.send_request.return_value = "[MARK] YES"
    evaluator = EvaluatorAgent(
        mock_gateway,
        ModelConfig(model_name="eval-model"),
        Path("eval_orig.txt"),
        "eval_agent",
        EventSystem()
    )
    response = evaluator.generate_response("Test prompt", "Test response")
    assert "[MARK] YES" in response

def test_aggregator_agent():
    mock_gateway = Mock()
    mock_gateway.send_request.return_value = "Final Aggregated Result"
    aggregator = AggregatorAgent(
        mock_gateway,
        ModelConfig(model_name="agg-model"),
        Path("aggregator.txt"),
        "agg_agent",
        EventSystem()
    )
    response = aggregator.generate_response("Context")
    assert response == "Final Aggregated Result"