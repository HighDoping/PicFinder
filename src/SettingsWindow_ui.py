# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'SettingsWindow.ui'
##
## Created by: Qt User Interface Compiler version 6.10.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QSpinBox, QVBoxLayout, QWidget)

class Ui_Settings(object):
    def setupUi(self, Settings):
        if not Settings.objectName():
            Settings.setObjectName(u"Settings")
        Settings.resize(500, 438)
        icon = QIcon()
        icon.addFile(u"icon.ico", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        Settings.setWindowIcon(icon)
        self.verticalLayout_6 = QVBoxLayout(Settings)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.groupBox = QGroupBox(Settings)
        self.groupBox.setObjectName(u"groupBox")
        self.verticalLayout_3 = QVBoxLayout(self.groupBox)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.groupBox_2 = QGroupBox(self.groupBox)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.horizontalLayout_8 = QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label = QLabel(self.groupBox_2)
        self.label.setObjectName(u"label")

        self.horizontalLayout_2.addWidget(self.label)

        self.comboBox_classification_model = QComboBox(self.groupBox_2)
        self.comboBox_classification_model.addItem("")
        self.comboBox_classification_model.setObjectName(u"comboBox_classification_model")

        self.horizontalLayout_2.addWidget(self.comboBox_classification_model)


        self.horizontalLayout_8.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_4 = QLabel(self.groupBox_2)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_3.addWidget(self.label_4)

        self.doubleSpinBox_classification_threshold = QDoubleSpinBox(self.groupBox_2)
        self.doubleSpinBox_classification_threshold.setObjectName(u"doubleSpinBox_classification_threshold")
        self.doubleSpinBox_classification_threshold.setMaximum(1.000000000000000)
        self.doubleSpinBox_classification_threshold.setSingleStep(0.010000000000000)
        self.doubleSpinBox_classification_threshold.setValue(0.700000000000000)

        self.horizontalLayout_3.addWidget(self.doubleSpinBox_classification_threshold)


        self.horizontalLayout_8.addLayout(self.horizontalLayout_3)


        self.verticalLayout_3.addWidget(self.groupBox_2)

        self.groupBox_3 = QGroupBox(self.groupBox)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.verticalLayout_2 = QVBoxLayout(self.groupBox_3)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_2 = QLabel(self.groupBox_3)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_4.addWidget(self.label_2)

        self.comboBox_object_detection_model = QComboBox(self.groupBox_3)
        self.comboBox_object_detection_model.addItem("")
        self.comboBox_object_detection_model.setObjectName(u"comboBox_object_detection_model")

        self.horizontalLayout_4.addWidget(self.comboBox_object_detection_model)


        self.verticalLayout_2.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.label_5 = QLabel(self.groupBox_3)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_7.addWidget(self.label_5)

        self.doubleSpinBox_object_detection_confidence = QDoubleSpinBox(self.groupBox_3)
        self.doubleSpinBox_object_detection_confidence.setObjectName(u"doubleSpinBox_object_detection_confidence")
        self.doubleSpinBox_object_detection_confidence.setMaximum(1.000000000000000)
        self.doubleSpinBox_object_detection_confidence.setSingleStep(0.010000000000000)
        self.doubleSpinBox_object_detection_confidence.setValue(0.700000000000000)

        self.horizontalLayout_7.addWidget(self.doubleSpinBox_object_detection_confidence)


        self.horizontalLayout_9.addLayout(self.horizontalLayout_7)


        self.verticalLayout_2.addLayout(self.horizontalLayout_9)


        self.verticalLayout_3.addWidget(self.groupBox_3)

        self.groupBox_4 = QGroupBox(self.groupBox)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.verticalLayout = QVBoxLayout(self.groupBox_4)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_3 = QLabel(self.groupBox_4)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_5.addWidget(self.label_3)

        self.comboBox_OCR_model = QComboBox(self.groupBox_4)
        self.comboBox_OCR_model.addItem("")
        self.comboBox_OCR_model.addItem("")
        self.comboBox_OCR_model.setObjectName(u"comboBox_OCR_model")

        self.horizontalLayout_5.addWidget(self.comboBox_OCR_model)

        self.label_6 = QLabel(self.groupBox_4)
        self.label_6.setObjectName(u"label_6")

        self.horizontalLayout_5.addWidget(self.label_6)

        self.spinBox_parallel = QSpinBox(self.groupBox_4)
        self.spinBox_parallel.setObjectName(u"spinBox_parallel")
        self.spinBox_parallel.setMinimum(1)
        self.spinBox_parallel.setMaximum(256)
        self.spinBox_parallel.setValue(3)

        self.horizontalLayout_5.addWidget(self.spinBox_parallel)


        self.verticalLayout.addLayout(self.horizontalLayout_5)


        self.verticalLayout_3.addWidget(self.groupBox_4)


        self.verticalLayout_6.addWidget(self.groupBox)

        self.groupBox_5 = QGroupBox(Settings)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.verticalLayout_5 = QVBoxLayout(self.groupBox_5)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.checkBox_update = QCheckBox(self.groupBox_5)
        self.checkBox_update.setObjectName(u"checkBox_update")

        self.horizontalLayout_10.addWidget(self.checkBox_update)


        self.verticalLayout_5.addLayout(self.horizontalLayout_10)


        self.verticalLayout_6.addWidget(self.groupBox_5)

        self.groupBox_6 = QGroupBox(Settings)
        self.groupBox_6.setObjectName(u"groupBox_6")
        self.verticalLayout_4 = QVBoxLayout(self.groupBox_6)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.checkBox_load_all = QCheckBox(self.groupBox_6)
        self.checkBox_load_all.setObjectName(u"checkBox_load_all")

        self.verticalLayout_4.addWidget(self.checkBox_load_all)


        self.verticalLayout_6.addWidget(self.groupBox_6)

        self.pushButton_save = QPushButton(Settings)
        self.pushButton_save.setObjectName(u"pushButton_save")

        self.verticalLayout_6.addWidget(self.pushButton_save)


        self.retranslateUi(Settings)

        QMetaObject.connectSlotsByName(Settings)
    # setupUi

    def retranslateUi(self, Settings):
        Settings.setWindowTitle(QCoreApplication.translate("Settings", u"Form", None))
        self.groupBox.setTitle(QCoreApplication.translate("Settings", u"Model Settings", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("Settings", u"Classification", None))
        self.label.setText(QCoreApplication.translate("Settings", u"Model:", None))
        self.comboBox_classification_model.setItemText(0, QCoreApplication.translate("Settings", u"None", None))

        self.label_4.setText(QCoreApplication.translate("Settings", u"Threshold:", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("Settings", u"Object Detection:", None))
        self.label_2.setText(QCoreApplication.translate("Settings", u"Model:", None))
        self.comboBox_object_detection_model.setItemText(0, QCoreApplication.translate("Settings", u"None", None))

        self.label_5.setText(QCoreApplication.translate("Settings", u"Confidence:", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("Settings", u"OCR", None))
        self.label_3.setText(QCoreApplication.translate("Settings", u"Model:", None))
        self.comboBox_OCR_model.setItemText(0, QCoreApplication.translate("Settings", u"RapidOCR", None))
        self.comboBox_OCR_model.setItemText(1, QCoreApplication.translate("Settings", u"None", None))

        self.label_6.setText(QCoreApplication.translate("Settings", u"Parallel:", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("Settings", u"Index Setting", None))
        self.checkBox_update.setText(QCoreApplication.translate("Settings", u"Fully Update Database", None))
        self.groupBox_6.setTitle(QCoreApplication.translate("Settings", u"Search Setting", None))
        self.checkBox_load_all.setText(QCoreApplication.translate("Settings", u"Load Images when start up", None))
        self.pushButton_save.setText(QCoreApplication.translate("Settings", u"Save", None))
    # retranslateUi

