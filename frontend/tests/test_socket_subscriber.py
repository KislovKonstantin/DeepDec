import unittest
from unittest.mock import patch
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication
from app_project import ButtonFactory
from app_project import SocketSubscriber
import zmq

class TestButtonFactory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication([])

    @patch('PyQt5.QtGui.QFontDatabase.addApplicationFont', return_value=-1)
    def test_font_fallback(self, _):
        factory = ButtonFactory()
        font = factory.custom_font()
        self.assertIsInstance(font, QFont)

    @patch('PyQt5.QtGui.QIcon')
    def test_button_creation(self, mock_icon):
        factory = ButtonFactory()
        button = factory.create_button("test.png", "Test", lambda: None)
        self.assertIsNotNone(button.layout())

    @patch('PyQt5.QtGui.QFontDatabase.addApplicationFont', return_value=-1)
    def test_font_fallback_again(self, _):
        factory = ButtonFactory()
        font = factory.custom_font()
        self.assertIsInstance(font, QFont)

class TestSocketSubscriber(unittest.TestCase):
    @patch('zmq.Context')
    def test_socket_initialization(self, mock_context):
        subscriber = SocketSubscriber("tcp://*:5555")
        subscriber.start()
        subscriber.stop()
        mock_context.return_value.socket.assert_called_once_with(zmq.SUB)

if __name__ == '__main__':
    unittest.main()
