import unittest
from unittest.mock import patch, mock_open
from PyQt5.QtWidgets import QApplication
from app_project import FileUploaderApp

class TestFileUploader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication([])

    @patch('os.path.exists', return_value=True)
    @patch('os.remove')
    def test_file_removal(self, mock_remove, mock_exists):
        app = FileUploaderApp()
        app.remove_files()
        self.assertEqual(mock_remove.call_count, 2)

    @patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName', return_value=("test.txt", ""))
    @patch('builtins.open', new_callable=mock_open, read_data="test")
    def test_file_upload(self, mock_open_file, mock_file_dialog):
        app = FileUploaderApp()
        app.upload_file()
        self.assertEqual(app.file_loaded, 1)

if __name__ == '__main__':
    unittest.main()