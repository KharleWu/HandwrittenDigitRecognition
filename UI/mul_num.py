# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'single_num.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Mul_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(90, 0, 481, 181))
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(90, 200, 71, 21))
        self.label_2.setObjectName("label_2")

        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(90, 230, 171, 31))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")

        self.comboBox_2 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_2.setGeometry(QtCore.QRect(340, 230, 171, 31))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(340, 200, 71, 21))
        self.label_3.setObjectName("label_3")

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(587, 310, 81, 21))
        self.label_4.setObjectName("label_4")

        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(50, 290, 121, 21))
        self.label_6.setObjectName("label_6")

        self.pbtPredict = QtWidgets.QPushButton(self.centralwidget)
        self.pbtPredict.setGeometry(QtCore.QRect(90, 530, 101, 31))
        self.pbtPredict.setStyleSheet("")
        self.pbtPredict.setObjectName("pbtPredict")

        self.pbtClear = QtWidgets.QPushButton(self.centralwidget)
        self.pbtClear.setGeometry(QtCore.QRect(220, 530, 101, 31))
        self.pbtClear.setStyleSheet("")
        self.pbtClear.setCheckable(False)
        self.pbtClear.setChecked(False)
        self.pbtClear.setObjectName("pbtClear")

        self.pbtUpLoad = QtWidgets.QPushButton(self.centralwidget)
        self.pbtUpLoad.setGeometry(QtCore.QRect(350, 530, 101, 31))
        self.pbtUpLoad.setCheckable(False)
        self.pbtUpLoad.setObjectName("pbtUpLoad")

        self.lbDataArea = QtWidgets.QLabel(MainWindow)
        self.lbDataArea.setGeometry(QtCore.QRect(50, 320, 500, 180))
        self.lbDataArea.setMouseTracking(False)
        self.lbDataArea.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lbDataArea.setFrameShape(QtWidgets.QFrame.Box)
        self.lbDataArea.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.lbDataArea.setLineWidth(4)
        self.lbDataArea.setMidLineWidth(0)
        self.lbDataArea.setText("")
        self.lbDataArea.setObjectName("lbDataArea")

        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(0, 0, 75, 23))
        self.pushButton_5.setObjectName("pushButton_5")

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.verticalLayoutWidget = QtWidgets.QWidget(MainWindow)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(50, 320, 496, 176))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")

        self.dArea_Layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.dArea_Layout.setContentsMargins(0, 0, 0, 0)
        self.dArea_Layout.setSpacing(0)
        self.dArea_Layout.setObjectName("dArea_Layout")

        self.lbResult = QtWidgets.QLabel(MainWindow)
        self.lbResult.setGeometry(QtCore.QRect(600, 320, 91, 131))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.lbResult.setFont(font)
        self.lbResult.setObjectName("lbResult")

        self.retranslateUi(MainWindow)
        self.comboBox.activated['QString'].connect(MainWindow.cbBox_Mode_Callback)
        self.comboBox_2.activated['QString'].connect(MainWindow.cbBox_Mode_2_Callback)
        self.pbtClear.clicked.connect(MainWindow.pbtClear_Callback)
        self.pbtPredict.clicked.connect(MainWindow.pbtPredict_Callback)
        self.pbtUpLoad.clicked.connect(MainWindow.pbtUpLoad_Callback)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "多个数字识别"))

        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">使用说明</span></p><p><span style=\" font-size:12pt;\">1、根据所需功能选择模式</span></p><p><span style=\" font-size:12pt;\">2、选择想要使用的算法</span></p><p><span style=\" font-size:12pt;\">3、输入待识别数据后，点击“识别”，得到识别结果</span></p><p><span style=\" font-size:12pt;\">4、点击“清除数据”后可重新输入待识别数据</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">模式选择</span></p></body></html>"))

        self.comboBox.setItemText(0, _translate("MainWindow", "请选择您所需要的功能"))
        self.comboBox.setItemText(1, _translate("MainWindow", "鼠标手写输入"))
        self.comboBox.setItemText(2, _translate("MainWindow", "上传本地图片"))

        self.comboBox_2.setItemText(0, _translate("MainWindow", "卷积神经网络(CNN)"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "深度卷积神经网络(深度CNN)"))
        self.comboBox_2.setItemText(2, _translate("MainWindow", "最近邻算法(KNN)"))
        self.comboBox_2.setItemText(3, _translate("MainWindow", "支持向量机(SVM)"))

        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">算法选择</span></p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">识别结果:</span></p></body></html>"))
        self.label_6.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">数据输入区域：</span></p></body></html>"))
        self.pbtPredict.setText(_translate("MainWindow", "识别"))
        self.pbtClear.setText(_translate("MainWindow", "清除数据"))
        self.pbtUpLoad.setText(_translate("MainWindow", "上传本地图片"))
        self.pushButton_5.setText(_translate("MainWindow", "返回主页"))
        self.lbResult.setText(_translate("MainWindow", "9"))
