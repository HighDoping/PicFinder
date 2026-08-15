# -*- coding: utf-8 -*-
"""Non-blocking UI for the model download scripts used during development."""

import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)


class ModelDownloadDialog(QDialog):
    """Run a bundled model-download script without blocking the settings window."""

    models_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Download models")
        self.resize(580, 360)

        self.description = QLabel(
            "Download models into the application's models folder. "
            "YOLO downloads may take several minutes because the ONNX models are "
            "exported after download."
        )
        self.description.setWordWrap(True)

        self.download_yolo_minimal = QPushButton("Download YOLO (minimal)")
        self.download_yolo_full = QPushButton("Download YOLO (all sizes)")
        self.download_clip = QPushButton("Download multilingual CLIP")
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.close_button = QDialogButtonBox(QDialogButtonBox.Close)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.download_yolo_minimal)
        button_layout.addWidget(self.download_yolo_full)
        button_layout.addWidget(self.download_clip)

        layout = QVBoxLayout(self)
        layout.addWidget(self.description)
        layout.addLayout(button_layout)
        layout.addWidget(self.output)
        layout.addWidget(self.close_button)

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.SeparateChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.readyReadStandardError.connect(self._read_error)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)

        self.download_yolo_minimal.clicked.connect(
            lambda: self._start_download("download_models.py")
        )
        self.download_yolo_full.clicked.connect(
            lambda: self._start_download("download_models.py", "--full")
        )
        self.download_clip.clicked.connect(
            lambda: self._start_download("download_clip.py")
        )
        self.close_button.rejected.connect(self.reject)

    @property
    def scripts_dir(self):
        return Path(__file__).resolve().parent / "dev"

    def _start_download(self, script_name, *arguments):
        if self.process.state() != QProcess.NotRunning:
            return

        script_path = self.scripts_dir / script_name
        if not script_path.is_file():
            self.output.appendPlainText(f"Download script not found: {script_path}")
            return

        self.output.clear()
        self.output.appendPlainText(f"Starting {script_name}...")
        self._set_download_buttons_enabled(False)
        self.close_button.setEnabled(False)
        self.process.setWorkingDirectory(str(self.scripts_dir.parent))
        self.process.start(sys.executable, [str(script_path), *arguments])

    def _read_output(self):
        self._append_process_data(self.process.readAllStandardOutput())

    def _read_error(self):
        self._append_process_data(self.process.readAllStandardError())

    def _append_process_data(self, data):
        text = bytes(data).decode(errors="replace").rstrip()
        if text:
            self.output.appendPlainText(text)

    def _process_error(self, error):
        if error == QProcess.FailedToStart:
            self.output.appendPlainText(
                "Could not start the downloader. Run the project with its development "
                "Python environment so the downloader dependencies are available."
            )
            self._set_download_buttons_enabled(True)
            self.close_button.setEnabled(True)

    def _finished(self, exit_code, _exit_status):
        self._read_output()
        self._read_error()
        if exit_code == 0:
            self.output.appendPlainText("\nDownload completed.")
            self.models_changed.emit()
        else:
            self.output.appendPlainText(f"\nDownload failed (exit code {exit_code}).")
        self._set_download_buttons_enabled(True)
        self.close_button.setEnabled(True)

    def _set_download_buttons_enabled(self, enabled):
        self.download_yolo_minimal.setEnabled(enabled)
        self.download_yolo_full.setEnabled(enabled)
        self.download_clip.setEnabled(enabled)

    def reject(self):
        if self.process.state() == QProcess.NotRunning:
            super().reject()
