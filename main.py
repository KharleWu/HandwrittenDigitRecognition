# -*- coding: utf-8 -*-

import sys, os
from dataset.mnist import load_mnist
from PIL import Image, ImageQt
import numpy as np
import joblib
import cv2
import shutil

from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QApplication, QLabel, QMessageBox, QPushButton
from PyQt5.QtGui import QPainter, QPen, QPixmap, QColor, QImage
from PyQt5.QtCore import Qt, QPoint, QSize
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from Algorithm.simple_convnet import SimpleConvNet
from common.functions import softmax
from Algorithm.deep_convnet import DeepConvNet

from UI.single_num import sin_MainWindow
from UI.shouye import Ui_MainWindow
from UI.mul_num import Mul_MainWindow
from UI.paintboard import PaintBoard

MODE_SETUP = 0    # 设定初始状态
MODE_MNIST = 1    # MNIST随机抽取
MODE_WRITE = 2    # 手写输入
MODE_UPLOAD = 3   # 上传图片

MODE_CNN = 0
MODE_DCNN = 1
MODE_KNN = 2
MODE_SVM = 3

Thresh = 0.5      # 识别结果置信度阈值

# 读取MNIST数据集
(_, _), (x_test, _) = load_mnist(normalize=True, flatten=False, one_hot_label=False)

# 初始化网络

# 网络1：简单CNN
"""
conv - relu - pool - affine - relu - affine - softmax
"""
network = SimpleConvNet(input_dim=(1,28,28),
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
network.load_params("pkl/params.pkl")

# 网络2：深度CNN
network_2 = DeepConvNet()
network_2.load_params("pkl/deep_convnet_params.pkl")

# 网络3：KNN
knn = joblib.load('pkl/knn_params.pkl')

# 网络4：SVM
svm = joblib.load('pkl/svm_params.pkl')

class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.sin_window = None
        self.mul_window = None

    def show_sin_window(self):
        self.sin_window.show()
        self.setVisible(False)

    def show_mul_window(self):
        self.mul_window.show()
        self.setVisible(False)

    # 窗口居中
    def center(self):
        # 获得窗口
        framePos = self.frameGeometry()
        # 获得屏幕中心点
        scPos = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        framePos.moveCenter(scPos)
        self.move(framePos.topLeft())

    # 窗口关闭事件
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.Yes)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

class Sin_MainWindow(QMainWindow, sin_MainWindow):
    def __init__(self):
        super(Sin_MainWindow, self).__init__()
        # 初始化UI
        self.setupUi(self)
        self.center()

        self.mode = MODE_SETUP
        self.mode_al = MODE_CNN
        self.result = [0, 0]
        self.main_win = None

        # 初始化画板
        self.paintBoard = PaintBoard(self, Size = QSize(224, 224), Fill = QColor(0,0,0,0))
        self.paintBoard.setPenColor(QColor(0,0,0,0))
        self.dArea_Layout.addWidget(self.paintBoard)

        self.clearDataArea()
        self.pbtGetMnist.setEnabled(False)
        self.pbtUpLoad.setEnabled(False)
        self.pbtPredict.setEnabled(False)

    def show_main_window(self):
        self.main_win.show()
        self.setVisible(False)

    # 窗口居中
    def center(self):
        # 获得窗口
        framePos = self.frameGeometry()
        # 获得屏幕中心点
        scPos = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        framePos.moveCenter(scPos)
        self.move(framePos.topLeft())

    # 窗口关闭事件
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.Yes)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    # 清除数据待输入区
    def clearDataArea(self):
        self.paintBoard.Clear()
        self.lbDataArea.clear()
        self.lbResult.clear()
        self.lbCofidence.clear()
        self.result = [0, 0]

    """
    回调函数
    """
    # 模式下拉列表回调
    def cbBox_Mode_Callback(self, text):
        if text == '0：请选择您所需要的功能':
            self.mode = MODE_SETUP
            self.clearDataArea()
            self.pbtGetMnist.setEnabled(False)
            self.pbtUpLoad.setEnabled(False)
            self.pbtPredict.setEnabled(False)

            self.paintBoard.setBoardFill(QColor(0,0,0,0))
            self.paintBoard.setPenColor(QColor(0,0,0,0))

        elif text == 'MINIST随机抽取':
            self.mode = MODE_MNIST
            self.clearDataArea()
            self.pbtGetMnist.setEnabled(True)
            self.pbtUpLoad.setEnabled(False)

            self.paintBoard.setBoardFill(QColor(0,0,0,0))
            self.paintBoard.setPenColor(QColor(0,0,0,0))

        elif text == '鼠标手写输入':
            self.mode = MODE_WRITE
            self.clearDataArea()
            self.pbtGetMnist.setEnabled(False)
            self.pbtUpLoad.setEnabled(False)
            self.pbtPredict.setEnabled(True)

            # 更改背景
            self.paintBoard.setBoardFill(QColor(0,0,0,255))
            self.paintBoard.setPenColor(QColor(255,255,255,255))

        elif text == '上传本地图片':
            self.mode = MODE_UPLOAD
            self.clearDataArea()
            self.pbtGetMnist.setEnabled(False)
            self.pbtUpLoad.setEnabled(True)

            self.paintBoard.setBoardFill(QColor(0,0,0,0))
            self.paintBoard.setPenColor(QColor(0,0,0,0))

    # 选择分类所用的算法
    def cbBox_Mode_2_Callback(self, text):
        if(text == "卷积神经网络(CNN)"):
            self.mode_al = MODE_CNN
        elif(text == "深度卷积神经网络(深度CNN)"):
            self.mode_al = MODE_DCNN
        elif(text == "最近邻算法(KNN)"):
            self.mode_al = MODE_KNN
        elif(text == "支持向量机(SVM)"):
            self.mode_al = MODE_SVM

    # 初始化
    def pbtSetUp_Callback(self):
        self.clearDataArea()

    # 数据清除
    def pbtClear_Callback(self):
        self.clearDataArea()

    # 识别
    def pbtPredict_Callback(self):
        __img, img_array =[],[]      # 将图像统一从qimage->pil image -> np.array [1, 1, 28, 28]

        # 获取qimage格式图像
        if self.mode == MODE_MNIST:
            __img = self.lbDataArea.pixmap()  # label内若无图像返回None
            if __img == None:   # 无图像则用纯黑代替
                # __img = QImage(224, 224, QImage.Format_Grayscale8)
                __img = ImageQt.ImageQt(Image.fromarray(np.uint8(np.zeros([224,224]))))
            else: __img = __img.toImage()
        elif self.mode == MODE_WRITE:
            __img = self.paintBoard.getContentAsQImage()
        elif self.mode == MODE_UPLOAD:
            #print(self.f_in)
            __img = ImageQt.toqimage(self.f_in)

        # 转换成pil image类型处理
        pil_img = ImageQt.fromqimage(__img)
        pil_img = pil_img.resize((28, 28), Image.ANTIALIAS)

        # pil_img.save('test.png')

        img_array = np.array(pil_img.convert('L')).reshape(1,1,28, 28) / 255.0
        img_array_tmp = img_array.reshape(1, -1)
        # img_array = np.where(img_array>0.5, 1, 0)

        # reshape成网络输入类型
        if(self.mode_al == MODE_CNN):
            __result = network.predict(img_array)      # shape:[1, 10]
            __result = softmax(__result)
            self.result[0] = np.argmax(__result)          # 预测的数字
            self.result[1] = __result[0, self.result[0]]     # 置信度
        elif(self.mode_al == MODE_DCNN):
            __result = network_2.predict(img_array)
            __result = softmax(__result)
            self.result[0] = np.argmax(__result)          # 预测的数字
            self.result[1] = __result[0, self.result[0]]     # 置信度
        elif(self.mode_al == MODE_KNN):
            self.result[0] = knn.predict(img_array_tmp)
            # self.result[1] = acc_knn[self.result[0]] * 100
            self.result[1] = 95
        elif(self.mode_al == MODE_SVM):
            self.result[0] = svm.predict(img_array_tmp)
            # self.result[1] = acc_svm[self.result[0]] * 100
            self.result[1] = 95


        self.lbResult.setText("%d" % (self.result[0]))
        self.lbCofidence.setText("%.8f" % (self.result[1]))

    # 随机抽取
    def pbtGetMnist_Callback(self):
        self.clearDataArea()
        self.pbtPredict.setEnabled(True)

        # 随机抽取一张测试集图片，放大后显示
        img = x_test[np.random.randint(0, 9999)]    # shape:[1,28,28]
        img = img.reshape(28, 28)                   # shape:[28,28]

        img = img * 0xff      # 恢复灰度值大小
        pil_img = Image.fromarray(np.uint8(img))
        pil_img = pil_img.resize((224, 224))        # 图像放大显示

        # 将pil图像转换成qimage类型
        qimage = ImageQt.ImageQt(pil_img)

        # 将qimage类型图像显示在label
        pix = QPixmap.fromImage(qimage)
        self.lbDataArea.setPixmap(pix)

    #上传文件
    def pbtUpLoad_Callback(self):
        self.clearDataArea()
        #获取本地图片路径
        f_name = ''
        openfile_name = QFileDialog.getOpenFileName(self, '选择文件', '', "Image File(*.jpg *.jpeg)")
        f_name = openfile_name[0]
        print(f_name)
        if(f_name == ''):
            return
        self.pbtPredict.setEnabled(True)
        #print(f_name)

        #file_in存源文件
        file_in = Image.open(f_name)
        #将源文件进行存储
        num = len(os.listdir(r'dataset/input/'))
        file_in.save('dataset/input/'+ str(num + 1)+ ".jpg")

        self.f_in = file_in
        pil_img = file_in.resize((224, 224))# 图像放大显示
        # 将pil图像转换成qimage类型
        qimage = ImageQt.ImageQt(pil_img)

        # 将qimage类型图像显示在label
        pix = QPixmap.fromImage(qimage)
        self.lbDataArea.setPixmap(pix)

class mul_window(QMainWindow,Mul_MainWindow):
    def __init__(self):
        super(mul_window, self).__init__()

        # 初始化UI
        self.setupUi(self)
        self.center()

        self.mode = MODE_SETUP
        self.mode_al = MODE_CNN
        self.result = [0, 0]
        self.main_win = None

        # 初始化画板
        self.paintBoard = PaintBoard(self, Size = QSize(500, 180), Fill = QColor(0,0,0,0))
        self.paintBoard.setPenColor(QColor(0,0,0,0))
        self.dArea_Layout.addWidget(self.paintBoard)

        self.clearDataArea()
        self.pbtUpLoad.setEnabled(False)
        self.pbtPredict.setEnabled(False)

    def show_main_window(self):
        self.main_win.show()
        self.setVisible(False)

    # 窗口居中
    def center(self):
        # 获得窗口
        framePos = self.frameGeometry()
        # 获得屏幕中心点
        scPos = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        framePos.moveCenter(scPos)
        self.move(framePos.topLeft())

    # 窗口关闭事件
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.Yes)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    # 清除数据待输入区
    def clearDataArea(self):
        self.paintBoard.Clear()
        self.lbDataArea.clear()
        self.lbResult.clear()
        self.result = [0, 0]

    """
    回调函数
    """
    # 模式下拉列表回调
    def cbBox_Mode_Callback(self, text):
        if text == '请选择您所需要的功能':
            self.mode = MODE_SETUP
            self.clearDataArea()
            self.pbtUpLoad.setEnabled(False)
            self.pbtPredict.setEnabled(False)

            self.paintBoard.setBoardFill(QColor(0,0,0,0))
            self.paintBoard.setPenColor(QColor(0,0,0,0))

        elif text == '鼠标手写输入':
            self.mode = MODE_WRITE
            self.clearDataArea()
            self.pbtUpLoad.setEnabled(False)
            self.pbtPredict.setEnabled(True)

            # 更改背景
            self.paintBoard.setPenColor(QColor(0,0,0,255))
            self.paintBoard.setBoardFill(QColor(255,255,255,255))

        elif text == '上传本地图片':
            self.mode = MODE_UPLOAD
            self.clearDataArea()
            self.pbtUpLoad.setEnabled(True)

            self.paintBoard.setBoardFill(QColor(0,0,0,0))
            self.paintBoard.setPenColor(QColor(0,0,0,0))

    # 选择分类所用的算法
    def cbBox_Mode_2_Callback(self, text):
        if(text == "卷积神经网络(CNN)"):
            self.mode_al = MODE_CNN
        elif(text == "深度卷积神经网络(深度CNN)"):
            self.mode_al = MODE_DCNN
        elif(text == "最近邻算法(KNN)"):
            self.mode_al = MODE_KNN
        elif(text == "支持向量机(SVM)"):
            self.mode_al = MODE_SVM

    # 初始化
    def pbtSetUp_Callback(self):
        self.clearDataArea()

    # 数据清除
    def pbtClear_Callback(self):
        self.clearDataArea()

    # 获取分割线
    def get_split_line(self, img, projection_row):
        split_line_list = []
        flag = False
        start = 0
        end = 0
        for i in range(0, len(projection_row)):
            if flag == False and projection_row[i] > 0:
                flag = True
                start = i
            elif flag and (projection_row[i] == 0 or i == len(projection_row) - 1):
                flag = False
                end = i
                if end - start < 15:  # need specify or rewrite
                    flag = True
                    continue
                else:
                    split_line_list.append((start, end))
        return split_line_list

    # 获取轮廓
    def get_contours(self, img):
        contour_list = []
        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(0, len(contours[0])):
            x, y, w, h = cv2.boundingRect(contours[0][i])
            contour_list.append((x, y, w, h))
            # cv2.rectangle(img_input, (x,y), (x+w, y+h), (0,0,255))
        return contour_list

    # 排序，合并
    def sort_merge(self, contour_row):
        # 排序
        contour_row = sorted(contour_row, key=lambda x: x[0])  # sort by x
        # print(contour_row)
        i = 0
        # 合并
        for _ in contour_row:
            if i == len(contour_row) - 1 or contour_row[i][0] == -1:
                break
            # print(contour_row[i])
            rectR = contour_row[i + 1]
            rectL = contour_row[i]
            ovlp = rectL[0] + rectL[2] - rectR[0]
            dist = abs((rectR[0] + rectR[2] / 2) - (rectL[0] - rectL[2] / 2))
            w_L = rectL[0] + rectL[2]
            w_R = rectR[0] + rectR[2]
            span = (w_R if w_R > w_L else w_L) - rectL[0]
            nmovlp = (ovlp / rectL[2] + ovlp / rectR[2]) / 2 - dist / span / 8
            if nmovlp > 0:
                x = rectL[0]
                y = (rectL[1] if rectL[1] < rectR[1] else rectR[1])
                w_L = rectL[0] + rectL[2]
                w_R = rectR[0] + rectR[2]
                w = (w_R if w_R > w_L else w_L) - x
                h_L = rectL[1] + rectL[3]
                h_R = rectR[1] + rectR[3]
                h = (h_R if h_R > h_L else h_L) - y
                contour_row[i] = (x, y, w, h)
                contour_row.pop(i + 1)  # after pop , index at i
                contour_row.append((-1, -1, -1, -1))  # add to fix bug(the better way is use iterator)
                i -= 1
            i += 1
        # print(contour_row)
        return contour_row

    # 组合垂直线
    def combine_verticalLine(self, contour_row):
        i = 0
        pop_num = 0
        for _ in contour_row:
            rect = contour_row[i]
            if rect[0] == -1:
                break
            if rect[2] == 0:
                i += 1
                continue
            if rect[3] * 1.0 / rect[2] > 4:
                if i != 0 and i != len(contour_row) - 1:
                    rect_left = contour_row[i - 1]
                    rect_right = contour_row[i + 1]
                    left_dis = rect[0] - rect_left[0] - rect_left[2]
                    right_dis = rect_right[0] - rect[0] - rect[2]
                    if left_dis <= right_dis and rect_left[2] < rect_right[2]:
                        x = rect_left[0]
                        y = (rect_left[1] if rect_left[1] < rect[1] else rect[1])
                        w = rect[0] + rect[2] - rect_left[0]
                        h_1 = rect_left[1] + rect_left[3]
                        h_2 = rect[1] + rect[3]
                        h_ = (h_1 if h_1 > h_2 else h_2)
                        h = h_ - y
                        contour_row[i - 1] = (x, y, w, h)
                        contour_row.pop(i)
                        contour_row.append((-1, -1, -1, -1))
                        pop_num += 1
                    else:
                        x = rect[0]
                        y = (rect[1] if rect[1] < rect_right[1] else rect_right[1])
                        w = rect_right[0] + rect_right[2] - rect[0]
                        h_1 = rect_right[1] + rect_right[3]
                        h_2 = rect[1] + rect[3]
                        h_ = (h_1 if h_1 > h_2 else h_2)
                        h = h_ - y
                        contour_row[i] = (x, y, w, h)
                        contour_row.pop(i + 1)
                        contour_row.append((-1, -1, -1, -1))
                        pop_num += 1
            i += 1
        for i in range(0, pop_num):
            contour_row.pop()
        return contour_row

    # 拆分超大宽度
    def split_oversizeWidth(self, contour_row):
        i = 0
        for _ in contour_row:
            rect = contour_row[i]
            if rect[2] * 1.0 / rect[3] > 1.2:  # height/width>1.2 -> split
                x_new = int(rect[0] + rect[2] / 2 + 1)
                y_new = rect[1]
                w_new = rect[0] + rect[2] - x_new
                h_new = rect[3]
                contour_row[i] = (rect[0], rect[1], int(rect[2] / 2), rect[3])
                contour_row.insert(i + 1, (x_new, y_new, w_new, h_new))
            i += 1
        return contour_row

    # 图像预处理
    def image_preprocess(self, img_input):
        gray_img = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.GaussianBlur(gray_img, (3, 3), 3)
        _, img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)  # 将一幅灰度图二值化 input-one channel
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img = cv2.erode(img, kernel)

        return img

    # 获取分割结果
    def get_segmentation_result(self, img, img_input):  # has been eroded
        projection_row = cv2.reduce(img, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)  # projection
        split_line_list = self.get_split_line(img, projection_row)  # split image as row
        segmentation_result = []
        result = []
        for i in split_line_list:
            img_row = img[i[0]:i[1], :]
            contour_row = self.get_contours(img_row)
            contour_row = self.sort_merge(contour_row)
            contour_row = self.split_oversizeWidth(contour_row)
            contour_row = self.combine_verticalLine(contour_row)
            segmentation_result.append(contour_row)
            tmp = 0
            for (x, y, w, h) in contour_row:  # draw
                sizeof = 0;
                if(w > h):
                    sizeof = w;
                else:
                    sizeof = h;
                img_name = 'dataset/mul_tmp/' + str(tmp) + ".jpg"
                tmp += 1
                y += i[0]
                cv2.rectangle(img_input, (x,y), (x+w,y+h), (0, 0, 255))
                if w>0 and h>0:
                    target = np.zeros((sizeof, sizeof), dtype = np.uint8)
                    bgr_img = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
                    for k in range(int((sizeof-h)/2), int((sizeof+h)/2)):
                        for j in range(int((sizeof - w)/2), int((sizeof + w)/2)):
                            bgr_img[k, j] = img[y + k - int((sizeof-h)/2), x + j - int((sizeof-w)/2)]
                    cv2.imwrite(img_name,bgr_img)
        return segmentation_result

    # 识别
    def pbtPredict_Callback(self):
        __img, img_array =[],[]      # 将图像统一从qimage->pil image -> np.array [1, 1, 28, 28]
        img_num = []
        img_num_tmp = []
        self.result = []
        s = ''
        # 获取qimage格式图像
        if self.mode == MODE_WRITE:
            # 删除文件夹
            shutil.rmtree("dataset/mul_tmp")
            # 重新创建文件夹
            os.mkdir("dataset/mul_tmp")
            __img = self.paintBoard.getContentAsQImage()
            #将源文件进行存储
            num = len(os.listdir(r'dataset/input/'))
            __img.save('dataset/input/'+ str(num + 1)+ ".jpg")
            f_name = 'dataset/input/'+ str(num + 1)+ ".jpg"
            img_input = cv2.imread(f_name, 1)  # (2975, 1787, 3)   但是图片查看器显示的是  1787 * 2975
            img = self.image_preprocess(img_input)  # 预处理
            segmentation_result = self.get_segmentation_result(img, img_input)
        # elif self.mode == MODE_UPLOAD:
        path = "dataset/mul_tmp/"
        num = len(os.listdir(r'dataset/mul_tmp/'))
        for i in range (0, num):
            path = "dataset/mul_tmp/"
            pil_img = Image.open(path + str(i) + '.jpg')
            pil_img = pil_img.resize((28, 28), Image.ANTIALIAS)
            img_array = np.array(pil_img.convert('L')).reshape(1,1,28, 28) / 255.0
            img_num.append(img_array)
            img_array_tmp = img_array.reshape(1, -1)
            img_num_tmp.append(img_array_tmp)

            #得到结果
            if(self.mode_al == MODE_CNN):
                __result = network.predict(img_num[i])      # shape:[1, 10]
                __result = softmax(__result)
                self.result.append(np.argmax(__result))
                s = s + str(self.result[i])
            elif(self.mode_al == MODE_DCNN):
                __result = network_2.predict(img_num[i])
                __result = softmax(__result)
                self.result.append(np.argmax(__result))
                s = s + str(self.result[i])
            elif(self.mode_al == MODE_KNN):
                tmp = knn.predict(img_num_tmp[i])
                self.result.append(tmp[0])
                s = s + str(tmp[0])
            elif(self.mode_al == MODE_SVM):
                tmp = svm.predict(img_num_tmp[i])
                self.result.append(tmp[0])
                s = s + str(tmp[0])

        self.lbResult.setText(s)

    #上传文件
    def pbtUpLoad_Callback(self):
        self.clearDataArea()
        # 删除文件夹
        shutil.rmtree("dataset/mul_tmp")
        # 重新创建文件夹
        os.mkdir("dataset/mul_tmp")

        #获取本地图片路径
        f_name = ''
        openfile_name = QFileDialog.getOpenFileName(self, '选择文件', '', "Image File(*.jpg *.jpeg)")
        f_name = openfile_name[0]

        if(f_name == ''):
            return

        self.pbtPredict.setEnabled(True)
        #print(f_name)
        img_input = cv2.imread(f_name, 1)  # (2975, 1787, 3)   但是图片查看器显示的是  1787 * 2975
        img = self.image_preprocess(img_input)  # 预处理
        segmentation_result = self.get_segmentation_result(img, img_input)
        file_in = Image.open(f_name)
        #按照要求调整像素
        resized_image = file_in.resize((500, 180), Image.ANTIALIAS)
        #将源文件和调整后的文件均进行存储
        num = len(os.listdir(r'dataset/input/'))
        file_in.save('dataset/input/'+ str(num + 1)+ ".jpg")
        # 将pil图像转换成qimage类型
        qimage = ImageQt.toqimage(resized_image)

        # 将qimage类型图像显示在label
        pix = QPixmap.fromImage(qimage)
        self.lbDataArea.setPixmap(pix)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    main = MainWindow()
    sin = Sin_MainWindow()
    mul = mul_window()

    main.sin_window = sin
    main.mul_window = mul
    sin.main_win = main
    mul.main_win = main

    main.show()

    # 点击进入单个数字识别
    main.pushButton_1.clicked.connect(main.show_sin_window)
    # 点击进入多个数字识别
    main.pushButton_2.clicked.connect(main.show_mul_window)
    # 点击从单个数字识别跳转回首页
    sin.pushButton_5.clicked.connect(sin.show_main_window)
    # 点击从多个数字识别跳转回首页
    mul.pushButton_5.clicked.connect(mul.show_main_window)
    sys.exit(app.exec_())
