import time

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics, preprocessing, svm

RANDOM_STATE = 40  # 随机种子
TEST_SIZE = 0.2  # 测试集比例

# '''
# 	采用sklearn的svm函数
#
# '''
with open("../chn_mnist", "rb") as f:
    data = pickle.load(f)
images = data["images"]
targets = data["targets"]
for img in images:
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  #OTSU二值化
    img = img/255

images = np.reshape(data["images"], (data["images"].shape[0], -1))
X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                                    stratify=targets)  # 训练集与测试集划分
print('Begin:' + time.strftime('%Y-%m-%d %H:%M:%S'))
model = svm.SVC(kernel = 'sigmoid',gamma='scale')  # 使用sigmoid核
model.fit(X_train, pd.DataFrame(y_train).values.ravel())
print('End:' + time.strftime('%Y-%m-%d %H:%M:%S'))

ypre = model.predict(X_test)
print('准确率:%f' % model.score(X_test, y_test))

print(metrics.classification_report(ypre, y_test))
print(y_test)
C=confusion_matrix(y_test, ypre, labels=[0,1,2,3,4,5,6,7,8,9,10,100,1000,10000,100000000])
df=pd.DataFrame(C,index=[0,1,2,3,4,5,6,7,8,9,10,100,1000,10000,100000000],columns=[0,1,2,3,4,5,6,7,8,9,10,100,1000,10000,100000000])
ax = sns.heatmap(df, fmt='.20g',annot=True)
plt.show()
