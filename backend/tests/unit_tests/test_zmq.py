from unittest.mock import patch, call
import zmq
from deepdec import SocketMessenger


@patch('zmq.Context')
def test_socket_initialization(mock_context):
    messenger = SocketMessenger()
    mock_context.return_value.socket.assert_called_once_with(zmq.PUB)
    messenger.socket.connect.assert_called_once_with("tcp://localhost:5555")


@patch('zmq.Socket.send_string')
def test_socket_send(mock_send):
    messenger = SocketMessenger()
    messenger.send("Test", "Message")
    expected_calls = [
        call('System: ZeroMQ connection established'),
        call('Test: Message')
    ]
    mock_send.assert_has_calls(expected_calls, any_order=False)
    assert mock_send.call_count == 2