import pytest
from unittest.mock import Mock, patch
from deepdec import APIGateway, APIClientFactory, SocketMessenger

@pytest.fixture
def mock_socket_messenger():
    return Mock(spec=SocketMessenger)

@pytest.fixture
def mock_api_client():
    client = Mock()
    client.send_request.side_effect = Exception("API Error")
    return client

@patch("deepdec.APIClientFactory")
def test_api_gateway_error_handling(mock_api_client_factory, mock_api_client, mock_socket_messenger):
    mock_api_client_factory.return_value.create_api_client.return_value = mock_api_client
    gateway = APIGateway(config={})
    gateway.messenger = mock_socket_messenger
    result = gateway.send_request(model="test_model", prompt="test_prompt")
    assert result is None
    mock_socket_messenger.send.assert_called_once_with(
        "API", "Generic Error: API Error"
    )
