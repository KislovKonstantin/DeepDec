import unittest
from PyQt5.QtWidgets import QApplication
from app_project import MessageDisplayApp


class TestMessageDisplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication([])

    def test_message_display(self):
        app = MessageDisplayApp()
        app.display_message("Test")
        self.assertIn("Test", app.text_edit.toPlainText())

    def test_clear_function(self):
        app = MessageDisplayApp()
        app.text_edit.setText("Test")
        app.clear()
        self.assertEqual(app.text_edit.toPlainText(), "")


if __name__ == '__main__':
    unittest.main()