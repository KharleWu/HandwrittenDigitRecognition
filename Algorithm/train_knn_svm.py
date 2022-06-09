import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier # Sklearn中kNN算法
from sklearn.svm import SVC # SKlearn中SVM算法
import joblib


# 数据处理
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255.0
test_images = test_images/ 255.0
train_images = np.reshape(train_images, (len(train_images), 28 * 28))
test_images = np.reshape(test_images, (len(test_images), 28 * 28))
rand_arr = np.arange(test_images.shape[0])
np.random.shuffle(rand_arr)
test_images = test_images[rand_arr[0:1500]]
test_labels = test_labels[rand_arr[0:1500]]

knn = KNeighborsClassifier(algorithm='kd_tree', n_neighbors = 3)
knn.fit(train_images, train_labels) # 训练模型
joblib.dump(knn, '../pkl/knn_params.pkl')

svm = SVC(C=1.0,kernel='linear')
svm.fit(train_images, train_labels)
joblib.dump(svm, '../pkl/svm_params.pkl')
'''
(train_dataset, train_labels), (test_dataset, test_labels) = tf.keras.datasets.mnist.load_data()

SIZE_IMAGE = train_dataset.shape[1]

train_labels = np.array(train_labels, dtype=np.int32)
def get_accuracy(predictions, labels):
    acc = (np.squeeze(predictions) == labels).mean()
    return acc * 100

def raw_pixels(img):
    return img.flatten()

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SIZE_IMAGE * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SIZE_IMAGE, SIZE_IMAGE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

    return img

shuffle = np.random.permutation(len(train_dataset))
train_dataset, train_labels = train_dataset[shuffle], train_labels[shuffle]

descriptors = []
for img in train_dataset:
    descriptors.append(np.float32(raw_pixels(deskew(img))))
print(len(descriptors[0]))
descriptors = np.squeeze(descriptors)

# 根据我们的测试，我们得到当split_values为0.9的时候，各个部分预测准确度更高，因此，我们选择k = 0.9
knn = cv2.ml.KNearest_create()
partition = int(0.9 * len(descriptors))
descriptors_train, descriptors_test = np.split(descriptors, [partition])
labels_train, labels_test = np.split(train_labels, [partition])
knn.train(descriptors_train, cv2.ml.ROW_SAMPLE, labels_train)
print(type(descriptors_test[0]))
print(type(descriptors_test[0][0]))
ret, result, neighbours, dist = knn.findNearest(descriptors_test, 3)
print(result)

# 数据划分
split_values = np.arange(0.1, 1, 0.2)
# 创建字典用于存储准确率
results = defaultdict(list)
# 创建 KNN 模型
knn = cv2.ml.KNearest_create()
for split_value in split_values:
    partition = int(split_value * len(descriptors))
    descriptors_train, descriptors_test = np.split(descriptors, [partition])
    labels_train, labels_test = np.split(train_labels, [partition])
    print('Training KNN model')
    knn.train(descriptors_train, cv2.ml.ROW_SAMPLE, labels_train)

    # 存储准确率
    for k in np.arange(1, 10):
        ret, result, neighbours, dist = knn.findNearest(descriptors_test, k)
        acc = get_accuracy(result, labels_test)
        print(" {}".format("%.2f" % acc))
        results[int(split_value * 100)].append(acc)

fig = plt.figure(figsize=(12, 5))
plt.suptitle("k-NN handwritten digits recognition", fontsize=14, fontweight='bold')

ax = plt.subplot(1, 1, 1)
ax.set_xlim(0, 10)
dim = np.arange(1, 10)

for key in results:
    ax.plot(dim, results[key], linestyle='--', marker='o', label=str(key) + "%")

plt.legend(loc='upper left', title="% training")
plt.title('Accuracy of the k-NN model varying both k and the percentage of images to train/test with pre-processing '
          'and HoG features')
plt.xlabel("number of k")
plt.ylabel("accuracy")
'''