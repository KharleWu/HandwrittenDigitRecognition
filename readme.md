我们是小组2，本项目是手写数字识别项目，总共有以下几个文件夹:
1. Algorithm  
deep_convnet为深度cnn的训练模型  
simple_convnet为cnn的训练模型  
接下来三个分别为模型训练的代码文件
2. common  
该文件主要存储的是深度cnn和cnn模型定义时所用到的一些函数定义
3. dataset  
input存放了单或多数字上传或者多数字手写板书写的数字图像  
mul_tmp存放了多数字识别前图像分割后的图像结果  
pic为我们存得的一些数字图片源文件
4. pkl  
存放训练好的几个模型(注意，模型中因为knn_params太大导致无法上传，故运行前请先运行Algorithm/train_knn_svm）
5. UI  
存放了各个页面以及手写画板的ui文件和对应的py文件
6. main.py  
运行代码的文件