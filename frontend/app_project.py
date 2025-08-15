import sys
import os
import threading
import time
import subprocess
import importlib
import json


class LibraryInstaller:
    REQUIRED_LIBRARIES = [
        'requests',
        'pyzmq',
        'zmq',
        'openai',
        'PyQt5'
    ]

    @classmethod
    def install_required_libraries(cls):
        for library in cls.REQUIRED_LIBRARIES:
            try:
                importlib.import_module(library)
                print(f"{library} is already installed.")
            except ImportError:
                print(f"{library} is not installed. Installing...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", library])
                print(f"{library} has been installed.")

if __name__ == "__main__":
    LibraryInstaller.install_required_libraries()


import zmq
from PyQt5.QtWidgets import (QApplication, QWidget, QMainWindow, QFileDialog,
                             QMessageBox, QLabel, QPushButton, QVBoxLayout,
                             QHBoxLayout, QStyle, QProgressBar, QTextEdit, QSizePolicy, QLineEdit)
from PyQt5.QtGui import QIcon, QPixmap, QImage, QPainter, QBrush, QColor, QFont, QFontDatabase
from PyQt5.QtCore import Qt, QSize, QRect, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal, QThread




class ButtonFactory:
    def create_button(self, image_path, text, command):
        button = QPushButton()
        button.setStyleSheet("background: none; border: none;")
        button.setFixedSize(150, 160)
        icon = QIcon(image_path)
        button.setIcon(icon)
        button.setIconSize(QSize(150, 160))
        button.clicked.connect(command)

        font = self.custom_font()
        label = QLabel(text)
        label.setFont(font)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("background: none; color: white; font-weight: bold; font-size: 17px")

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(0)
        layout.addWidget(button, 0, Qt.AlignCenter)
        layout.addWidget(label, 0, Qt.AlignCenter)

        button_widget = QWidget()
        button_widget.setLayout(layout)
        button_widget.setStyleSheet("background: none;")
        return button_widget

    def custom_font(self):
        font_id = QFontDatabase.addApplicationFont("images/KodeMono-Regular.ttf")
        if font_id == -1:
            print("Error in loading of the font")
            return QFont()
        families = QFontDatabase.applicationFontFamilies(font_id)
        font_family = families[0]
        return QFont(font_family, 16)

class SocketSubscriber(QThread):
    message_received = pyqtSignal(str)

    def __init__(self, socket_addr: str):
        super().__init__()
        self.socket_addr = socket_addr
        self.running = True

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.bind(self.socket_addr)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")

        while self.running:
            message = socket.recv_string()
            self.message_received.emit(message)

    def stop(self):
        self.running = False

class MessageDisplayApp(QWidget, ButtonFactory):
    def __init__(self, socket_addr=None):
        super().__init__()
        self.initUI()
        if socket_addr != None:
            self.socket_subscriber = SocketSubscriber(socket_addr)
            self.socket_subscriber.message_received.connect(self.display_message)
            self.socket_subscriber.start()
        self.process = None

    def initUI(self):
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)

        font = self.custom_font()
        self.text_edit.setFont(font)
        self.text_edit.setStyleSheet("background: transparent; color: white; border: none;")

        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        self.setLayout(layout)

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

    def display_message(self, message):
        self.text_edit.append(message)

    def start_process(self):
        self.process = subprocess.Popen(['python', 'deepdec.py'])

    def stop_process(self):
        if self.process:
            self.process.terminate()
            self.process = None
            self.display_message("Interuption of work has been forced")
    def clear(self):
        self.text_edit.clear()
    def closeEvent(self, event):
        self.socket_subscriber.stop()
        event.accept()


class RegistrationWindow(QWidget, ButtonFactory):
    registration_complete = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(800, 600)

        self.initUI()
        self.setup_connections()

    def initUI(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)


        self.background = QLabel(self)
        pixmap = QPixmap("images/оо.jpg")
        if pixmap.isNull():
            print("Error: Could not load background image")
            pixmap = QPixmap(800, 600)
            pixmap.fill(Qt.gray)
        self.background.setPixmap(pixmap)
        self.background.setGeometry(0, 0, 800, 600)


        container = QWidget()
        container.setFixedSize(600, 500)
        container.setStyleSheet("""
            background-color: rgba(0, 0, 0, 180);
            border-radius: 15px;
        """)


        form_layout = QVBoxLayout(container)
        form_layout.setAlignment(Qt.AlignCenter)
        form_layout.setContentsMargins(40, 40, 40, 40)
        form_layout.setSpacing(20)


        title = QLabel("Configuring the API")
        title.setStyleSheet("""
            color: white;
            font-size: 28px;
            font-weight: bold;
        """)
        title.setAlignment(Qt.AlignCenter)
        form_layout.addWidget(title)


        self.api_type_edit = self.create_input_field("LLM API type (openai/huggingface/openrouter)")
        self.api_key_edit = self.create_input_field("API key")
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_url_edit = self.create_input_field("API URL (e.g. https://openrouter.ai/api/v1/chat/completions)")
        self.model_edit = self.create_input_field("Model name (e.g. deepseek/deepseek-r1-distill-llama-70b:free)")

        form_layout.addWidget(self.api_type_edit)
        form_layout.addWidget(self.api_key_edit)
        form_layout.addWidget(self.api_url_edit)
        form_layout.addWidget(self.model_edit)


        self.save_btn = QPushButton("Save configuration")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(70, 130, 180, 200);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(70, 130, 180, 250);
            }
            QPushButton:disabled {
                background-color: rgba(70, 130, 180, 100);
            }
        """)
        form_layout.addWidget(self.save_btn, 0, Qt.AlignCenter)

        main_layout.addWidget(container, 0, Qt.AlignCenter)

    def create_input_field(self, placeholder):
        line_edit = QLineEdit()
        line_edit.setPlaceholderText(placeholder)
        line_edit.setStyleSheet("""
            QLineEdit {
                background-color: rgba(255, 255, 255, 220);
                border: 1px solid rgba(70, 130, 180, 150);
                border-radius: 8px;
                padding: 10px;
                color: black;
                font-size: 14px;
                min-width: 300px;
            }
        """)
        return line_edit

    def setup_connections(self):
        self.save_btn.clicked.connect(self.save_config)

    def save_config(self):
        try:
            config = {
                "api": {
                    "api_type": self.api_type_edit.text().strip(),
                    "api_key": self.api_key_edit.text().strip(),
                    "api_url": self.api_url_edit.text().strip()
                },
                "models": {
                    "analysis": self.model_edit.text().strip(),
                    "eval_analysis": self.model_edit.text().strip(),
                    "commentary": self.model_edit.text().strip(),
                    "eval_comment": self.model_edit.text().strip(),
                    "reconstruct": self.model_edit.text().strip(),
                    "eval_reconstruct": self.model_edit.text().strip(),
                    "aggregator": self.model_edit.text().strip(),
                    "eval_aggregator": self.model_edit.text().strip()
                },
                "prompts": {
                    "analysis": "prompts/code_analyst.txt",
                    "eval_analysis": "prompts/eval_analysis.txt",
                    "commentary": "prompts/commentator.txt",
                    "eval_comment": "prompts/eval_comments.txt",
                    "reconstruct": "prompts/reconstructor.txt",
                    "eval_reconstruct": "prompts/eval_orig.txt",
                    "aggregator": "prompts/aggregator.txt",
                    "eval_aggregator": "prompts/eval_aggr.txt"
                }
            }

            # Проверка обязательных полей
            if not all([config["api"]["api_type"], config["api"]["api_key"], config["models"]["analysis"]]):
                QMessageBox.warning(self, "Error", "Fill in all required fields!")
                return

            config_path = os.path.join(os.path.dirname(__file__), "config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)

            self.registration_complete.emit()
            self.close()

        except Exception as e:
            QMessageBox.critical(self, "Error", f": {str(e)}")

class FileUploaderApp(QMainWindow, ButtonFactory):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ReCodeTangle")
        image_path = "images/main_window.jpg"
        pixmap = QPixmap(image_path)
        self.setFixedSize(pixmap.width(), pixmap.height())
        self.setWindowIcon(QIcon("images/icon.ico"))
        self.uploaded_file_path = None
        self.new_file_path = None
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.file_loaded = 0
        self.file_isnt_running = 1


        self.remove_files()
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(config_path):
            try:
                os.remove(config_path)
            except Exception as e:
                print(f"Error removing config: {e}")
        # Подключение к сокету
        self.message_display = MessageDisplayApp("tcp://*:5555")

        self.initUI()

        self.show_registration_window()

        # Таймер для проверки наличия файла full.json
        self.check_file_timer = QTimer(self)
        self.check_file_timer.timeout.connect(self.check_for_report_file)


    def initUI(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.central_widget.setStyleSheet("""
            QWidget {
                background-image: url("images/main_window.jpg");
                background-repeat: no-repeat;
                background-position: center;
            }
        """)

        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_panel.setStyleSheet("background: transparent;")
        buttons_layout = QVBoxLayout(left_panel)
        buttons_layout.setAlignment(Qt.AlignTop)
        buttons_layout.setContentsMargins(90, 40, 20, 0)
        # buttons_layout.setSpacing(10)

        # Первый ряд кнопок
        top_row = QHBoxLayout()
        top_row.setSpacing(15)

        self.upload_button = self.create_button("images/upload_button.png", "Upload txt\nfile with\ndecompiled \nC code",
                                                self.upload_file)
        top_row.addWidget(self.upload_button, 0, Qt.AlignCenter)

        self.view_file_button = self.create_button("images/view_file.png", " View\n uploaded\n  txt file",
                                                   self.download_uploaded_file)
        top_row.addWidget(self.view_file_button, 0, Qt.AlignCenter)

        buttons_layout.addLayout(top_row)

        # Вторая кнопка (центр)
        self.start_comp = self.create_button("images/start_dec.png", "Start\nreverse engineering",
                                                  self.start_decomp)
        buttons_layout.addWidget(self.start_comp, 0, Qt.AlignCenter)

        # Третий ряд (стоп и другие кнопки)
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(25)


        buttons_layout.addLayout(bottom_row)

        right_panel = QWidget()
        right_panel.setStyleSheet("background: transparent;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(140, 60, 50, 50)

        self.message_display.setMinimumSize(300, 300)
        right_layout.addWidget(self.message_display)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

        self.central_widget.setLayout(main_layout)


    def remove_files(self):
        files_to_remove = ["full.json", "test.txt"]
        for file_name in files_to_remove:
            file_path = os.path.join(os.path.dirname(__file__), file_name)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"File is deleted: {file_path}")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Не удалось удалить файл {file_name}: {e}")



    def show_registration_window(self):
        # Проверяем, есть ли уже конфиг
        if os.path.exists("config.json"):
            self.show_main_window()
        else:
            self.registration_window = RegistrationWindow()
            self.registration_window.registration_complete.connect(self.show_main_window)
            self.registration_window.show()
    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload file", "", "Text files (*.txt)")
        if file_path:
            self.uploaded_file_path = file_path
            new_file_path = os.path.join(os.path.dirname(__file__), "test.txt")
            try:
                with open(file_path, 'r', encoding='utf-8') as source_file:
                    content = source_file.read()
                    with open(new_file_path, 'w', encoding='utf-8') as target_file:
                        target_file.write(content)
                self.file_loaded = 1
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Saving file: {e}")
                self.file_loaded = 0
            self.message_display.display_message("New file is loaded")

    def start_decomp(self):
        if self.file_loaded == 1 and self.file_isnt_running == 1:
            self.message_display.start_process()
            self.file_isnt_running = 0
            self.message_display.display_message("Processing")
            self.check_file_timer.start(50)
        elif self.file_loaded == 0:
            QMessageBox.information(self, "", "There is no file\nto be processed")
        else:
            QMessageBox.information(self, "", "Reverse engineering is \nalready running")


    def keyPressEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_Plus:
                self.message_display.change_font_size(1)
            elif event.key() == Qt.Key_Minus:
                self.message_display.change_font_size(-1)
        super().keyPressEvent(event)


    def open_analysis_complete_window(self, report_file_path, analyzed_file_path):
        self.analysis_complete_window = AnalysisCompleteWindow(report_file_path, analyzed_file_path, self)
        self.analysis_complete_window.show()
        self.analysis_complete_window.activateWindow()
        #self.hide()

    def download_uploaded_file(self):
        if self.uploaded_file_path:
            file_download_path, _ = QFileDialog.getSaveFileName(self, "Сохранить загруженный файл как...", "",
                                                                "Text files (*.txt)")
            if file_download_path:
                try:
                    with open(self.uploaded_file_path, 'r', encoding='utf-8') as source_file:
                        content = source_file.read()
                        with open(file_download_path, 'w', encoding='utf-8') as target_file:
                            target_file.write(content)
                    QMessageBox.information(self, "Success", "File is downloaded")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Downloading file: {e}")
        else:
            QMessageBox.warning(self, "Warning", "File is not loaded.")

    def check_for_report_file(self):
        report_file_path = os.path.join(os.path.dirname(__file__), "full.json")
        if os.path.exists(report_file_path):
            self.check_file_timer.stop()
            self.file_loaded = 0
            self.file_isnt_running = 1


            self.open_analysis_complete_window(report_file_path, self.uploaded_file_path)

    def show_main_window(self):
        self.show()


class AnalysisCompleteWindow(QWidget, ButtonFactory):
    def __init__(self, report_file_path, analyzed_file_path, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.report_file_path = report_file_path
        self.analyzed_file_path = analyzed_file_path
        self.initUI()

    def initUI(self):
        # Установите фоновое изображение
        background_image = QPixmap("images/analysis.jpg")
        if background_image.isNull():
            print("Couldn't load an image analysis.jpg")
            return

        # Установите фиксированный размер окна в соответствии с размерами изображения
        self.setFixedSize(background_image.width(), background_image.height())

        # Установите начальную позицию окна со смещением

        background_label = QLabel(self)
        background_label.setPixmap(background_image)
        background_label.setGeometry(0, 0, background_image.width(), background_image.height())
        background_label.lower()

        # Основной макет
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Левая панель с кнопками
        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_panel.setStyleSheet("background: transparent;")
        buttons_layout = QVBoxLayout(left_panel)
        buttons_layout.setAlignment(Qt.AlignTop)
        buttons_layout.setContentsMargins(90, 20, 0, 0)

        # Первый ряд кнопок
        top_row = QHBoxLayout()
        top_row.setSpacing(15)



        download_report_button = self.create_button("images/report.png", "Download\nreport", self.download_report)
        top_row.addWidget(download_report_button, 0, Qt.AlignCenter)

        download_analyzed_file_button = self.create_button("images/decomp.png", " Download\n original\n txt file", self.download_analyzed_file)
        top_row.addWidget(download_analyzed_file_button, 0, Qt.AlignCenter)

        return_button = self.create_button("images/return.png", "Return to\nUpload Window\n(report will be deleted)", self.return_to_previous_window)
        buttons_layout.addWidget(return_button, 0, Qt.AlignCenter)

        buttons_layout.addLayout(top_row)

        # Правая панель с дисплеями
        right_panel = QWidget()
        right_panel.setStyleSheet("background: transparent;")
        right_layout = QVBoxLayout(right_panel)

        right_layout.setContentsMargins(140, 60, 50, 50)


        self.message_display = MessageDisplayApp()
        self.messages_report()
        right_layout.addWidget(self.message_display)



        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

        self.setLayout(main_layout)
        self.setWindowTitle("The analysis is completed")

    def download_report(self):
        report_file_path = os.path.join(os.path.dirname(__file__), "full.json")
        if os.path.exists(report_file_path):
            file_download_path, _ = QFileDialog.getSaveFileName(self, "Save a report as...", "",
                                                                "JSON files (*.json)")
            if file_download_path:
                with open(report_file_path, 'r', encoding='utf-8') as source_file:
                    content = source_file.read()
                    with open(file_download_path, 'w', encoding='utf-8') as target_file:
                        target_file.write(content)

        else:
            QMessageBox.warning(self, "Warning", "ReportFileNotFound.")
    def download_analyzed_file(self):
        file_download_path, _ = QFileDialog.getSaveFileName(self, "Save analyzed file as...", "", "Text files (*.txt)")
        if file_download_path:
            try:
                with open(self.analyzed_file_path, 'r', encoding='utf-8') as source_file:
                    content = source_file.read()
                    with open(file_download_path, 'w', encoding='utf-8') as target_file:
                        target_file.write(content)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Downloading file: {e}")

    def return_to_previous_window(self):
        self.parent.remove_files()
        self.parent.message_display.clear()
        self.message_display.clear()
        self.hide()
    def messages_report(self):
        self.message_display.display_message("Your decompiled file is ready!")
        f = open('full.json', 'r')
        s = ''
        for line in f:
            s = s + line
        split_text = s.replace('\\n', '\n')
        self.message_display.display_message(split_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = FileUploaderApp()
    sys.exit(app.exec_())
