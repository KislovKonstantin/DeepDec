from unittest.mock import Mock
from deepdec import BaseAgent, ModelConfig, EventSystem
from pathlib import Path

def test_base_agent():
    mock_gateway = Mock()
    mock_gateway.send_request.return_value = "Test response"
    agent = BaseAgent(
        mock_gateway,
        ModelConfig(model_name="test"),
        Path("code_alalyst.txt"),
        "test_agent",
        EventSystem()
    )
    assert agent.generate_response("test") == "Test response"
