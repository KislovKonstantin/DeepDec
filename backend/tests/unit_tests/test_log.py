from unittest.mock import patch
from deepdec import EventSystem, Event, SystemEvents, AgentObserver

def test_agent_observer_logging():
    with patch("logging.Logger.info") as mock_logger:
        event_system = EventSystem()
        observer = AgentObserver()
        event_system.register(observer)
        test_event = Event(
            SystemEvents.AGENT_STARTED,
            {"agent_name": "test_agent", "timestamp": "1234-56-78"}
        )
        event_system.post(test_event)
        mock_logger.assert_called_once_with(
            "Agent Event: SystemEvents.AGENT_STARTED, Data: {'agent_name': 'test_agent', 'timestamp': '1234-56-78'}"
        )