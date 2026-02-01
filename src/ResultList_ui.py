# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ResultList.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QSizePolicy, QToolButton,
    QVBoxLayout, QWidget)

class Ui_ResultListWidget(object):
    def setupUi(self, ResultListWidget):
        if not ResultListWidget.objectName():
            ResultListWidget.setObjectName(u"ResultListWidget")
        ResultListWidget.resize(600, 400)
        self.verticalLayout = QVBoxLayout(ResultListWidget)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.listWidget_results = QListWidget(ResultListWidget)
        self.listWidget_results.setObjectName(u"listWidget_results")

        self.verticalLayout.addWidget(self.listWidget_results)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label_3 = QLabel(ResultListWidget)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout.addWidget(self.label_3)

        self.comboBox_items_per_page = QComboBox(ResultListWidget)
        self.comboBox_items_per_page.addItem("")
        self.comboBox_items_per_page.addItem("")
        self.comboBox_items_per_page.addItem("")
        self.comboBox_items_per_page.addItem("")
        self.comboBox_items_per_page.setObjectName(u"comboBox_items_per_page")

        self.horizontalLayout.addWidget(self.comboBox_items_per_page)

        self.toolButton_toFirst = QToolButton(ResultListWidget)
        self.toolButton_toFirst.setObjectName(u"toolButton_toFirst")

        self.horizontalLayout.addWidget(self.toolButton_toFirst)

        self.toolButton_toPrev = QToolButton(ResultListWidget)
        self.toolButton_toPrev.setObjectName(u"toolButton_toPrev")

        self.horizontalLayout.addWidget(self.toolButton_toPrev)

        self.label_page_count = QLabel(ResultListWidget)
        self.label_page_count.setObjectName(u"label_page_count")
        self.label_page_count.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout.addWidget(self.label_page_count)

        self.toolButton_toNext = QToolButton(ResultListWidget)
        self.toolButton_toNext.setObjectName(u"toolButton_toNext")

        self.horizontalLayout.addWidget(self.toolButton_toNext)

        self.toolButton_toLast = QToolButton(ResultListWidget)
        self.toolButton_toLast.setObjectName(u"toolButton_toLast")

        self.horizontalLayout.addWidget(self.toolButton_toLast)


        self.verticalLayout.addLayout(self.horizontalLayout)


        self.retranslateUi(ResultListWidget)

        self.comboBox_items_per_page.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(ResultListWidget)
    # setupUi

    def retranslateUi(self, ResultListWidget):
        ResultListWidget.setWindowTitle(QCoreApplication.translate("ResultListWidget", u"Results", None))
        self.label_3.setText(QCoreApplication.translate("ResultListWidget", u"Items per page:", None))
        self.comboBox_items_per_page.setItemText(0, QCoreApplication.translate("ResultListWidget", u"10", None))
        self.comboBox_items_per_page.setItemText(1, QCoreApplication.translate("ResultListWidget", u"50", None))
        self.comboBox_items_per_page.setItemText(2, QCoreApplication.translate("ResultListWidget", u"100", None))
        self.comboBox_items_per_page.setItemText(3, QCoreApplication.translate("ResultListWidget", u"200", None))

        self.toolButton_toFirst.setText(QCoreApplication.translate("ResultListWidget", u"\u21e4", None))
        self.toolButton_toPrev.setText(QCoreApplication.translate("ResultListWidget", u"\u2190", None))
        self.label_page_count.setText(QCoreApplication.translate("ResultListWidget", u"0/0", None))
        self.toolButton_toNext.setText(QCoreApplication.translate("ResultListWidget", u"\u2192", None))
        self.toolButton_toLast.setText(QCoreApplication.translate("ResultListWidget", u"\u21e5", None))
    # retranslateUi

