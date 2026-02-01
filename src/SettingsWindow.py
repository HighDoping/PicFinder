# -*- coding: utf-8 -*-

import logging
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QWidget

from SettingsWindow_ui import Ui_Settings


class SettingsWindow(QWidget, Ui_Settings):
    def __init__(self):
        super(SettingsWindow, self).__init__()
        self.setupUi(self)
        self.get_models()
        self.settings = QSettings("HighDoping", "PicFinder")
        self.load_settings()
        self.pushButton_save.clicked.connect(self.gui_save)
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
            if Path(model_dir / files[0]).exists():
                self.models_cls.append(model)
            if Path(model_dir / files[1]).exists():
                self.models_coco.append(model)

        self.comboBox_classification_model.addItems(self.models_cls)
        self.comboBox_object_detection_model.addItems(
            model for model in self.models_coco
        )

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
        self.spinBox_OCR_parallel.setValue(int(self.settings.value("OCR_parallel", 3)))

        self.checkBox_update.setChecked(
            self.settings.value("FullUpdate", False, type=bool)
        )
        self.spinBox_batch_size.setValue(int(self.settings.value("batch_size", 100)))
        self.checkBox_load_all.setChecked(
            self.settings.value("load_all", False, type=bool)
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
        self.settings.setValue("OCR_parallel", self.spinBox_OCR_parallel.value())
        self.settings.setValue("FullUpdate", self.checkBox_update.isChecked())
        self.settings.setValue("batch_size", self.spinBox_batch_size.value())
        self.settings.setValue("load_all", self.checkBox_load_all.isChecked())

    def gui_save(self):
        self.save_settings()
        self.close()
