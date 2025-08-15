import unittest
from unittest.mock import patch, mock_open
from PyQt5.QtWidgets import QApplication
from app_project import RegistrationWindow

class TestRegistrationWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication([])

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_config_save(self, mock_json, _):
        window = RegistrationWindow()
        window.api_type_edit.setText("openai")
        window.api_key_edit.setText("test_key")
        window.model_edit.setText("test_model")
        window.save_config()
        self.assertTrue(mock_json.called)

if __name__ == '__main__':
    unittest.main()