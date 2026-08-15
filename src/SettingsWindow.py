# -*- coding: utf-8 -*-

import logging
from pathlib import Path

from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import QPushButton, QWidget

from ModelDownloadDialog import ModelDownloadDialog
from SettingsWindow_ui import Ui_Settings


class SettingsWindow(QWidget, Ui_Settings):
    finished = Signal()
    def __init__(self):
        super(SettingsWindow, self).__init__()
        self.setupUi(self)
        self.get_models()
        self.settings = QSettings("HighDoping", "PicFinder")
        self.load_settings()
        self.pushButton_save.clicked.connect(self.gui_save)
        self.pushButton_download_models = QPushButton("Download models")
        self.verticalLayout_6.insertWidget(1, self.pushButton_download_models)
        self.pushButton_download_models.clicked.connect(self.open_model_downloader)
        self.comboBox_object_detection_model.currentIndexChanged.connect(
            self.check_models
        )

    def get_models(self):
        model_dir = Path(__file__).parent / "models"
        self.models_cls = []
        self.models_coco = []

        model_files = {
            "yolo26n": ["yolo26n.onnx", "yolo26n-cls.onnx"],
            "yolo26s": ["yolo26s.onnx", "yolo26s-cls.onnx"],
            "yolo26m": ["yolo26m.onnx", "yolo26m-cls.onnx"],
            "yolo26l": ["yolo26l.onnx", "yolo26l-cls.onnx"],
            "yolo26x": ["yolo26x.onnx", "yolo26x-cls.onnx"],
        }

        for model, files in model_files.items():
            if (model_dir / files[1]).exists():
                self.models_cls.append(model)
            if (model_dir / files[0]).exists():
                self.models_coco.append(model)

        self.comboBox_classification_model.clear()
        self.comboBox_object_detection_model.clear()
        self.comboBox_classification_model.addItem("None")
        self.comboBox_object_detection_model.addItem("None")
        self.comboBox_classification_model.addItems(self.models_cls)
        self.comboBox_object_detection_model.addItems(self.models_coco)

    def open_model_downloader(self):
        self.model_download_dialog = ModelDownloadDialog(self)
        self.model_download_dialog.models_changed.connect(self.refresh_models)
        self.model_download_dialog.show()

    def refresh_models(self):
        classification_model = self.comboBox_classification_model.currentText()
        object_detection_model = self.comboBox_object_detection_model.currentText()
        self.get_models()
        self.comboBox_classification_model.setCurrentText(classification_model)
        self.comboBox_object_detection_model.setCurrentText(object_detection_model)

    def check_models(self):
        self.object_detection_model = self.comboBox_object_detection_model.currentText()

    def load_settings(self):
        self.classification_model = self.settings.value(
            "classification_model", "yolo26n"
        )
        if self.classification_model not in self.models_cls:
            self.classification_model = "None"
        self.comboBox_classification_model.setCurrentText(self.classification_model)
        self.doubleSpinBox_classification_threshold.setValue(
            float(self.settings.value("classification_threshold", 0.7))
        )
        self.object_detection_model = self.settings.value(
            "object_detection_model", "yolo26n"
        )
        if self.object_detection_model not in self.models_coco:
            self.object_detection_model = "None"
        self.comboBox_object_detection_model.setCurrentText(self.object_detection_model)

        self.object_detection_dataset = self.settings.value(
            "object_detection_dataset", ["COCO"]
        )
        self.doubleSpinBox_object_detection_confidence.setValue(
            float(self.settings.value("object_detection_conf_threshold", 0.7))
        )
        self.comboBox_OCR_model.setCurrentText(
            self.settings.value("OCR_model", "RapidOCR")
        )

        self.comboBox_CLIP_model.setCurrentText(
            self.settings.value("CLIP_model", "CLIP-ViT-B-32-multilingual")
        )

        self.spinBox_parallel.setValue(int(self.settings.value("parallel", 3)))

        self.checkBox_update.setChecked(
            self.settings.value("FullUpdate", False, type=bool)
        )

        self.checkBox_load_all.setChecked(
            self.settings.value("load_all", False, type=bool)
        )

        self.checkBox_enable_CLIP.setChecked(
            self.settings.value("enable_CLIP", False, type=bool)
        )

        self.doubleSpinBox_CLIP_threshold.setValue(
            float(self.settings.value("CLIP_threshold", 0.5))
        )
        self.save_settings()

    def save_settings(self):
        self.settings.setValue(
            "classification_model", self.comboBox_classification_model.currentText()
        )
        self.settings.setValue(
            "classification_threshold",
            self.doubleSpinBox_classification_threshold.value(),
        )
        self.settings.setValue(
            "object_detection_model",
            self.comboBox_object_detection_model.currentText(),
        )
        self.settings.setValue(
            "object_detection_dataset", self.object_detection_dataset
        )
        self.settings.setValue(
            "object_detection_conf_threshold",
            self.doubleSpinBox_object_detection_confidence.value(),
        )
        self.settings.setValue("OCR_model", self.comboBox_OCR_model.currentText())
        self.settings.setValue("CLIP_model", self.comboBox_CLIP_model.currentText())
        self.settings.setValue("parallel", self.spinBox_parallel.value())
        self.settings.setValue("FullUpdate", self.checkBox_update.isChecked())
        self.settings.setValue("load_all", self.checkBox_load_all.isChecked())
        self.settings.setValue("enable_CLIP", self.checkBox_enable_CLIP.isChecked())
        self.settings.setValue(
            "CLIP_threshold", self.doubleSpinBox_CLIP_threshold.value()
        )

    def gui_save(self):
        self.save_settings()
        self.finished.emit()
        self.close()
