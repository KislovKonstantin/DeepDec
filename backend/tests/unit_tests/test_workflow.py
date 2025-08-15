from deepdec import CodeWorkflow, WorkflowAggregate, ResponseParser, EventSystem
from unittest.mock import Mock

def test_workflow_retry_logic():
    mock_agent = Mock()
    mock_agent.generate_response.side_effect = ["1", "2", "3"]
    mock_evaluator = Mock()
    mock_evaluator.generate_response.return_value = "[MARK] YES"
    workflow = CodeWorkflow(
        WorkflowAggregate("test", mock_agent, mock_evaluator, ResponseParser(), EventSystem()),
        max_attempts=3
    )
    assert workflow.run("code") == "1"