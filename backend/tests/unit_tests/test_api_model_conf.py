from deepdec import APIConfig, ModelConfig
import pytest

def test_api_config_validation():
    valid = APIConfig(api_type="openai", api_key="sk-123")
    valid.validate()
    with pytest.raises(ValueError):
        APIConfig(api_type="", api_key="test").validate()

def test_model_config():
    model = ModelConfig(model_name="gpt-4")
    assert model.model_name == "gpt-4"
    model.validate()