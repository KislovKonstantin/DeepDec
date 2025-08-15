import unittest
from unittest.mock import patch, mock_open
from PyQt5.QtWidgets import QApplication
from app_project import AnalysisCompleteWindow

class TestAnalysisCompleteWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication([])

    @patch('builtins.open', mock_open(read_data='{"test":"data"}'))
    def test_report_display(self):
        window = AnalysisCompleteWindow("full.json", "test.txt", None)
        window.messages_report()
        self.assertIn("data", window.message_display.text_edit.toPlainText())

if __name__ == '__main__':
    unittest.main()
