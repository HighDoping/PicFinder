# -*- coding: utf-8 -*-

import logging
import os
import sys
from pathlib import Path

import onnxruntime
from PySide6.QtCore import QObject, QSettings, Qt, QThread, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from backend.qtworkers import IndexWorker, SearchWorker
from MainWindow_ui import Ui_MainWindow
from ResultList import ResultListWidget
from SettingsWindow import SettingsWindow


class QLogSignal(QObject):
    log = Signal(str)


class QLogHandler(logging.Handler):
    def __init__(self, emitter):
        super().__init__()
        self._emitter = emitter

    @property
    def emitter(self):
        return self._emitter

    def emit(self, record):
        msg = self.format(record)
        self.emitter.log.emit(msg)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.setAcceptDrops(True)
        
        self.progressBar.setVisible(False)

        q_log_signal = QLogSignal()
        h = QLogHandler(q_log_signal)
        h.setLevel(logging.ERROR)
        logging.getLogger().addHandler(h)
        q_log_signal.log.connect(self.error_pop_up)

        self.actionClear_DB.triggered.connect(self.clear_db)

        # add actions to menubar
        self.actionSettings.triggered.connect(self.open_settings)
        self.actionAbout.triggered.connect(self.open_about)

        self.pushButton_folder_browse.clicked.connect(self.browse_folder)
        self.pushButton_index.clicked.connect(self.index_folder)
        self.pushButton_search.clicked.connect(self.search)

        self.lineEdit_folder.textChanged.connect(self.lineEdit_folder_textChanged)

        self.lineEdit_search.returnPressed.connect(self.search)

        self.folder_path = Path()

        self.result_list_widget = ResultListWidget(self.folder_path, [])
        # add the list widget to the frame
        self.list_layout = QVBoxLayout()
        self.list_layout.addWidget(self.result_list_widget)
        self.frame.setLayout(self.list_layout)

        self.pushButton_index.setEnabled(False)
        self.pushButton_search.setEnabled(False)

        self.update_settings()

    def lineEdit_folder_textChanged(self, text):
        if text:
            self.pushButton_index.setEnabled(True)
        else:
            self.pushButton_index.setEnabled(False)
        self.folder_path = Path(text)
        self.db_path = self.folder_path / "PicFinder.db"
        self.result_list_widget.update_folder(self.folder_path)
        if self.db_exists_check():
            self.statusbar.showMessage(
                f"Folder: {self.folder_path.as_posix()} , Index Found"
            )
            if self.settings["load_all"] == True:
                self.search()
        else:
            self.statusbar.showMessage(
                f"Folder: {self.folder_path.as_posix()} , Index Not Found"
            )

    def db_exists_check(self):
        if self.db_path.exists():
            self.pushButton_search.setEnabled(True)
            return True
        else:
            self.pushButton_search.setEnabled(False)
            return False

    def browse_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if path:
            self.lineEdit_folder.setText(path)

    def index_folder(self):
        try:
            if self.index_worker_thread.isRunning():
                logging.error("Indexing already in progress")
                return
        except:
            pass
        try:
            if self.search_worker_thread.isRunning():
                logging.error("Search in progress, please wait")
                return
        except:
            pass
        self.update_settings()
        if self.folder_path.exists() and self.folder_path.is_dir():
            self.result_list_widget.update_folder(self.folder_path)
            self.index_worker = IndexWorker(self.folder_path, **self.settings)
            self.index_worker_thread = QThread()
            self.index_worker.moveToThread(self.index_worker_thread)
            self.index_worker_thread.started.connect(self.index_worker.run)
            self.index_worker.finished.connect(self.index_finished)
            self.index_worker.finished.connect(self.index_worker_thread.quit)
            self.index_worker.finished.connect(self.index_worker.deleteLater)
            self.index_worker_thread.finished.connect(
                self.index_worker_thread.deleteLater
            )
            self.index_worker.progress.connect(self.index_progress)
            self.index_worker_thread.start()
            self.statusbar.showMessage("Indexing...")
            self.progressBar.setVisible(True)
        else:
            self.statusbar.showMessage("Invalid Folder Path")

    def index_progress(self, value, total):
        self.progressBar.setMaximum(total)
        self.progressBar.setValue(value)
        self.statusbar.showMessage(f"Indexing... {value}/{total}")

    def index_finished(self):
        self.statusbar.showMessage("Indexing Finished")
        self.progressBar.setVisible(False)
        self.db_exists_check()

    def search(self):
        try:
            if self.search_worker_thread.isRunning():
                logging.error("Search already in progress")
                return
        except:
            pass
        try:
            if self.index_worker_thread.isRunning():
                logging.error("Indexing in progress, please wait")
                return
        except:
            pass
        query = self.lineEdit_search.text()
        if self.db_exists_check():
            self.search_worker = SearchWorker(self.db_path, query)
            self.search_worker_thread = QThread()
            self.search_worker.moveToThread(self.search_worker_thread)
            self.search_worker_thread.started.connect(self.search_worker.run)
            self.search_worker.finished.connect(self.search_finished)
            self.search_worker.finished.connect(self.search_worker_thread.quit)
            self.search_worker.finished.connect(self.search_worker.deleteLater)
            self.search_worker_thread.finished.connect(
                self.search_worker_thread.deleteLater
            )
            self.search_worker.result.connect(self.search_result)
            self.search_worker_thread.start()
            self.statusbar.showMessage("Searching...")
        else:
            self.statusbar.showMessage("Database not found")

    def search_finished(self):
        # self.statusbar.showMessage("Search Finished")
        pass

    def search_result(self, result):
        self.statusbar.showMessage(f"Search Finished, {len(result)} results Found.")
        self.result_list_widget.update_results(result)

    def open_settings(self):
        self.settings_window = SettingsWindow()
        self.settings_window.setWindowTitle("Settings")
        self.settings_window.show()

    def update_settings(self):
        settings = QSettings("HighDoping", "PicFinder")
        self.settings = {}
        self.settings["classification_model"] = settings.value(
            "classification_model", "yolo26n"
        )
        self.settings["classification_threshold"] = float(
            settings.value("classification_threshold", 0.7)
        )
        self.settings["object_detection_model"] = settings.value(
            "object_detection_model", "yolo26n"
        )
        self.settings["object_detection_dataset"] = settings.value(
            "object_detection_dataset", ["COCO"]
        )
        self.settings["object_detection_conf_threshold"] = float(
            settings.value("object_detection_conf_threshold", 0.7)
        )
        self.settings["OCR_model"] = settings.value("OCR_model", "RapidOCR")
        self.settings["OCR_parallel"] = settings.value("OCR_parallel", 3)
        self.settings["FullUpdate"] = settings.value("FullUpdate", False, type=bool)
        self.settings["load_all"] = settings.value("load_all", False)

    def open_about(self):
        self.about_window = AboutWindow()
        self.about_window.show()

    def clear_db(self):
        try:
            if self.index_worker_thread.isRunning():
                logging.error("Indexing in progress, please wait")
                return
        except:
            pass
        try:
            if self.search_worker_thread.isRunning():
                logging.error("Search in progress, please wait")
                return
        except:
            pass
        try:
            os.remove(self.db_path)
        except AttributeError:
            self.error_pop_up("Database not found")
            return
        except Exception as e:
            logging.error(e, exc_info=True)

        self.statusbar.showMessage("Database Cleared")

    def error_pop_up(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(message)
        msg.setWindowTitle("Error")
        msg.setWindowModality(Qt.NonModal)
        msg.exec_()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            # decide where is the mouse position
            if self.lineEdit_folder.underMouse():
                # check if the dropped file is a directory
                url = Path(url.toLocalFile())
                if url.is_dir():
                    self.lineEdit_folder.setText(url.as_posix())
                else:
                    self.statusbar.showMessage("Invalid Folder Path", 3000)
            else:
                event.ignore()


class AboutWindow(QWidget):

    def __init__(self):
        super(AboutWindow, self).__init__()
        self.setWindowTitle("About")
        self.setWindowIcon(QIcon("icon.ico"))
        self.setWindowModality(Qt.NonModal)

        self.label_1 = QLabel(
            f"PicFinder Version: {self.get_version()}, commit: {self.get_commit_hash()}\n"
        )
        self.label_2 = QLabel("Author: HighDoping\n")
        self.label_3 = QLabel(
            "For license and source code, please visit:\n"
            + "<a href=https://github.com/HighDoping/PicFinder>GitHub</a>\n"
        )
        self.label_3.setOpenExternalLinks(True)
        self.label_3.setWordWrap(True)
        self.label_3.setTextFormat(Qt.RichText)

        self.label_4 = QLabel(
            "System Information:\n"
            + f"Python version: {sys.version}\n"
            + "libsimple version: 0.4.0\n"
            + f"onnxruntime version: {onnxruntime.__version__}\n"
            + f"onnxruntime hardware: {onnxruntime.get_device()}\n"
            + f"onnxruntime available providers: {onnxruntime.get_available_providers()}\n"
        )
        self.label_4.setWordWrap(True)

        self.layout_1 = QVBoxLayout()
        self.layout_1.addWidget(self.label_1)
        self.layout_1.addWidget(self.label_2)
        self.layout_1.addWidget(self.label_3)
        self.layout_1.addWidget(self.label_4)

        self.setLayout(self.layout_1)

    def get_version(self):
        try:
            with open(Path(__file__).parent.resolve() / "VERSION") as f:
                # return first line
                version = f.readline().strip()
                return version
        except:
            return "Unknown"

    def get_commit_hash(self):
        try:
            with open(Path(__file__).parent.resolve() / "VERSION") as f:
                # return second line
                f.readline()
                commit_hash = f.readline().strip()
                return commit_hash
        except:
            return "Unknown"
