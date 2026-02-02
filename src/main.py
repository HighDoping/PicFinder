# -*- coding: utf-8 -*-
import argparse
import logging
import os
import sys
import tempfile
from multiprocessing import freeze_support


def main_gui():
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from MainWindow import MainWindow

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    window = MainWindow()
    icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")
    window.setWindowIcon(QIcon(icon_path))
    window.show()
    code = app.exec()
    os._exit(code)

if __name__ == "__main__":
    temp_dir = tempfile.gettempdir()
    is_nuitka = "__compiled__" in globals()
    if sys.stdout is None or is_nuitka:
        # sys.stdout = open(os.devnull, "w")
        sys.stdout = open(os.path.join(temp_dir, "stdout.log"), "w")
    if sys.stderr is None or is_nuitka:
        # sys.stderr = open(os.devnull, "w")
        sys.stderr = open(os.path.join(temp_dir, "stderr.log"), "w")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    freeze_support()
    main_gui()
