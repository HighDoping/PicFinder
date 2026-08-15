# -*- coding: utf-8 -*-
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QWidget

from ResultList_ui import Ui_ResultListWidget


class ResultListWidget(QWidget, Ui_ResultListWidget):
    def __init__(self, folder_path, result_list):
        super(ResultListWidget, self).__init__()
        self.setupUi(self)
        self.folder_path = folder_path
        self.result_list = result_list

        self.listWidget_results.setViewMode(QListWidget.IconMode)
        self.listWidget_results.setIconSize(QSize(300, 300))
        self.listWidget_results.setResizeMode(QListWidget.Adjust)
        self.listWidget_results.setWordWrap(True)
        self.listWidget_results.setFlow(QListWidget.LeftToRight)
        self.listWidget_results.setWrapping(True)
        self.listWidget_results.setGridSize(QSize(320, 320))
        self.listWidget_results.setSpacing(20)
        self.listWidget_results.setUniformItemSizes(False)
        self.listWidget_results.setTextElideMode(Qt.ElideMiddle)
        self.listWidget_results.setWordWrap(True)
        self.listWidget_results.itemDoubleClicked.connect(self.open_file)

        self.items_per_page = int(self.comboBox_items_per_page.currentText())
        self.comboBox_items_per_page.currentIndexChanged.connect(
            self.update_items_per_page
        )
        self.toolButton_toFirst.clicked.connect(self.first_page)
        self.toolButton_toPrev.clicked.connect(self.previous_page)
        self.toolButton_toNext.clicked.connect(self.next_page)
        self.toolButton_toLast.clicked.connect(self.last_page)

    def update_items_per_page(self):
        self.items_per_page = int(self.comboBox_items_per_page.currentText())
        self.page_results()

    def update_results(self, result_list: list):
        self.result_list = result_list
        self.page_results()

    def page_results(self):
        # group results by items per page
        self.paged_result_list = [
            self.result_list[i : i + self.items_per_page]
            for i in range(0, len(self.result_list), self.items_per_page)
        ]
        self.page_index = 0
        self.page_count = len(self.paged_result_list)
        self.show_page()

    def show_page(self):
        if self.page_count == 0:
            self.label_page_count.setText("0/0")
            self.listWidget_results.clear()
            return
        self.label_page_count.setText(f"{self.page_index+1}/{self.page_count}")
        self.populate_list(self.paged_result_list[self.page_index])

    def first_page(self):
        self.page_index = 0
        self.show_page()

    def previous_page(self):
        if self.page_index > 0:
            self.page_index -= 1
            self.show_page()

    def next_page(self):
        if self.page_index < self.page_count - 1:
            self.page_index += 1
            self.show_page()

    def last_page(self):
        self.page_index = self.page_count - 1
        self.show_page()

    def populate_list(self, result: list):
        self.listWidget_results.clear()
        for file in result:
            file_path = self.folder_path / Path(file[2])
            file_classification = file[3]
            file_classification_confidence = file[4]
            file_object = file[5]
            file_object_confidence = file[6]
            file_ocr = file[7]
            file_ocr_confidence = file[8]
            clip_similarity = file[10] if len(file) > 10 else None

            file_info = (
                f"File: {file_path.as_posix()}\n"
                f"Classification: {file_classification} ({file_classification_confidence:.2f})\n"
                f"Object: {file_object} ({file_object_confidence:.2f})\n"
                f"OCR: {file_ocr} ({file_ocr_confidence:.2f})"
            )
            if clip_similarity is not None:
                file_info += f"\nCLIP similarity: {clip_similarity:.4f}"

            item = QListWidgetItem()
            item.setIcon(QPixmap(file_path.as_posix()))
            item.setText(file[2])
            item.setToolTip(file_info)
            self.listWidget_results.addItem(item)

    def open_file(self, item: QListWidgetItem):
        file_path = self.folder_path / Path(item.text())
        url = QUrl.fromLocalFile(file_path)
        QDesktopServices.openUrl(url)

    def clear_list(self):
        self.listWidget_results.clear()
        self.result_list = []

    def update_folder(self, folder_path: Path):
        self.folder_path = folder_path
        self.clear_list()
