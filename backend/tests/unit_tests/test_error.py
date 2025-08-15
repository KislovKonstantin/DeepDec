from deepdec import APIErrorHandler, JSONDecodeErrorHandler, GenericErrorHandler
import requests


def test_error_chain():
    chain = APIErrorHandler()
    chain.set_next(JSONDecodeErrorHandler()).set_next(GenericErrorHandler())
    error = requests.exceptions.ConnectionError()
    assert "API Request Error" in chain.handle_error(error, {})
    assert "Generic Error" in chain.handle_error(ValueError(), {})