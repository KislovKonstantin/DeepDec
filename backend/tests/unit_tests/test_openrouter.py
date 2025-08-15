import json
from deepdec import OpenRouterAPIClient, APIConfig
from unittest.mock import patch, MagicMock

def test_openrouter_client():
    with open("config.json") as f:
        config_data = json.load(f)
    config = APIConfig(**config_data["api"])
    client = OpenRouterAPIClient(config)
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test Response"}}]
        }
        mock_post.return_value = mock_response
        response = client.send_request("test-model", "Test prompt")
        mock_post.assert_called_once_with(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Test prompt"}]
            },
            timeout=60
        )
        assert response == "Test Response"