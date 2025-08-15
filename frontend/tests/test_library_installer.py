import unittest
from unittest.mock import patch
from app_project import LibraryInstaller

class TestLibraryInstaller(unittest.TestCase):
    @patch('importlib.import_module')
    @patch('subprocess.check_call')
    def test_all_libraries_installed(self, mock_check, mock_import):
        mock_import.return_value = None
        LibraryInstaller.install_required_libraries()
        self.assertEqual(mock_import.call_count, 5)
        mock_check.assert_not_called()

    @patch('importlib.import_module', side_effect=ImportError)
    @patch('subprocess.check_call')
    def test_missing_library(self, mock_check, _):
        LibraryInstaller.install_required_libraries()
        mock_check.assert_called()

if __name__ == '__main__':
    unittest.main()
