import pandas as pd
import numpy as np
import cv2
import os
import seaborn as sns
import skimage
import skimage.io
import skimage.transform
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from keras.preprocessing import image
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization, LeakyReLU
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import warnings

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 常量定义
IMAGE_PATH = '../Chinese MNIST/data/data/'
CSV_PATH = '../Chinese MNIST/chinese_mnist.csv'
IMAGE_WIDTH = 64  # 数据图片宽度
IMAGE_HEIGHT = 64  # 数据图片高度
IMAGE_CHANNELS = 1  # 图片输入通道
RANDOM_STATE = 42  # 随机种子
TEST_SIZE = 0.2  # 测试集比例
VAL_SIZE = 0.2  # 验证集比例
CONV_2D_DIM_1 = 16  # 第一个卷积层输入通道数
CONV_2D_DIM_2 = 16  # 第二个卷积层输入通道数
CONV_2D_DIM_3 = 32  # 第三个卷积层输入通道数
CONV_2D_DIM_4 = 64  # 第四个卷积层输入通道数
MAX_POOL_DIM = 2  # 池化窗口大小
KERNEL_SIZE = 3  # 卷积核大小
BATCH_SIZE = 32  # 每批训练数据集大小
NO_EPOCHS = 50  # 定型周期
DROPOUT_RATIO = 0.5  # 神经元舍弃率
PATIENCE = 5  # 容忍准确率无增长数量
VERBOSE = 1  # 进度条输出


#############################################################
# 数据集加载部分
#############################################################


def read_image(file_name):  # 读取图片
    image = cv2.imread(IMAGE_PATH + file_name, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(image.shape)
    # plt.imshow(image)
    # plt.show()
    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image = image/255 # OTSU二值化
    image = skimage.transform.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT, 1), mode='reflect')
    return image[:, :, :]


def categories_encoder(dataset, var='character'):  # 对应图片与标签
    X = np.stack(dataset['file'].apply(read_image))
    y = pd.get_dummies(dataset[var], drop_first=False)
    return X, y


def create_file_name(x):  # 返回标准格式图片名
    file_name = f"input_{x[0]}_{x[1]}_{x[2]}.jpg"
    return file_name


def read_image_sizes(file_name):  # 返回图片大小
    image = skimage.io.imread(IMAGE_PATH + file_name)
    return list(image.shape)


data_df = pd.read_csv(CSV_PATH)  # 读取csv
image_files = list(os.listdir(IMAGE_PATH))  # 列出所有图片
data_df["file"] = data_df.apply(create_file_name, axis=1)  # 图片载入
file_names = list(data_df['file'])

m = np.stack(data_df['file'].apply(read_image_sizes))
df = pd.DataFrame(m, columns=['w', 'h'])  # 读入数据长宽 (64,64)
data_df = pd.concat([data_df, df], axis=1, sort=False)  # 完整数据集
#print(data_df.head())
# plt.imshow(read_image('input_1_1_1.jpg'),cmap = 'gray')
# plt.show()
train_df, test_df = train_test_split(data_df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                     stratify=data_df["code"].values)  # 训练集(训练+验证)与测试集划分
train_df, val_df = train_test_split(train_df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                    stratify=train_df["code"].values)  # 训练集与验证集划分

X_train, y_train = categories_encoder(train_df)  # 连接训练集图片与标签
X_val, y_val = categories_encoder(val_df)  # 连接验证集图片与标签
X_test, y_test = categories_encoder(test_df)  # 连接测试集图片与标签

#################################################################################
# 创建模型
##################################################################################

generator = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    rescale=1. / 225,
)

model = Sequential()  # 搭建CNN卷积神经网络
model.add(Conv2D(CONV_2D_DIM_1, kernel_size=KERNEL_SIZE, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
                 activation='relu', padding='same'))  # 第一个卷积层
model.add(Conv2D(CONV_2D_DIM_2, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))  # 第二个卷积层
model.add(MaxPool2D(MAX_POOL_DIM))  # 最大值池化
model.add(Dropout(DROPOUT_RATIO))  # 随机失活
model.add(Conv2D(CONV_2D_DIM_2, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))  # 第三个卷积层
model.add(Conv2D(CONV_2D_DIM_2, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))  # 第四个卷积层
model.add(Dropout(DROPOUT_RATIO))  # 随机失活
model.add(Flatten())  # 扁平化
model.add(Dense(y_train.columns.size, activation='softmax'))  # 全连接层
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# 配置网络学习过程 自适应矩估计优化器 交叉熵损失函数 准确率

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.99 ** (x + NO_EPOCHS))  # 自适应学习率
earlystopper = EarlyStopping(monitor='loss', patience=PATIENCE, verbose=VERBOSE)  # 提前结束训练
checkpointer = ModelCheckpoint('best_model.h5',  # 保存最佳模型
                               monitor='val_accuracy',
                               verbose=VERBOSE,
                               save_best_only=True,
                               save_weights_only=True)

train_model = model.fit(X_train, y_train,  # 模型训练
                        batch_size=BATCH_SIZE,
                        epochs=NO_EPOCHS,
                        verbose=1,
                        validation_data=(X_val, y_val),
                        callbacks=[earlystopper, checkpointer, annealer])

#########################################################################
#  结果图像显示
#########################################################################
def create_trace(x, y, ylabel, color):
    trace = go.Scatter(
        x=x, y=y,
        name=ylabel,
        marker=dict(color=color),
        mode="markers+lines",
        text=x
    )
    return trace


def plot_accuracy_and_loss(train_model):   #准确率&损失值图像
    hist = train_model.history
    acc = hist['accuracy']
    val_acc = hist['val_accuracy']
    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = list(range(1, len(acc) + 1))
    # 定义traces
    trace_ta = create_trace(epochs, acc, "Training accuracy", "Green")
    trace_va = create_trace(epochs, val_acc, "Validation accuracy", "Red")
    trace_tl = create_trace(epochs, loss, "Training loss", "Blue")
    trace_vl = create_trace(epochs, val_loss, "Validation loss", "Magenta")
    fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Training and validation accuracy',
                                                              'Training and validation loss'))
    # a向图形添加traces
    fig.append_trace(trace_ta, 1, 1)
    fig.append_trace(trace_va, 1, 1)
    fig.append_trace(trace_tl, 1, 2)
    fig.append_trace(trace_vl, 1, 2)
    # 设置图形的layout
    fig['layout']['xaxis'].update(title='Epoch')
    fig['layout']['xaxis2'].update(title='Epoch')
    fig['layout']['yaxis'].update(title='Accuracy', range=[0, 1])
    fig['layout']['yaxis2'].update(title='Loss', range=[0, 1])
    # 图
    plot(fig, filename='accuracy-loss.html')


plot_accuracy_and_loss(train_model)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


def test_accuracy_report(model):
    predicted = model.predict(X_test)
    test_predicted = np.argmax(predicted, axis=1)
    test_truth = np.argmax(y_test.values, axis=1)
    print(metrics.classification_report(test_truth, test_predicted, target_names=y_test.columns))
    test_res = model.evaluate(X_test, y_test.values, verbose=0)
    print('Loss function: %s, accuracy:' % test_res[0], test_res[1])


model_optimal = model
model_optimal.load_weights('best_model.h5')
ypre = model.predict(X_test)                #CNN模型对网络进行预测
test_predicted = np.argmax(ypre, axis=1)
test_truth = np.argmax(y_test.values, axis=1)
print(test_predicted)
print(ypre.shape)
C = confusion_matrix(test_truth, test_predicted, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
df = pd.DataFrame(C, index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000, 10000, 100000000],
                  columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000, 10000, 100000000])
ax = sns.heatmap(df, fmt='.20g', annot=True)
plt.show()
score = model_optimal.evaluate(X_test, y_test, verbose=0)
print(f'Best validation loss: {score[0]}, accuracy: {score[1]}')

test_accuracy_report(model_optimal)

model.save('mnist.h5')

plt.figure(figsize=(15, 10))


def resultPrint(pre):
    # result=["一","七","万","三","九","二","五","亿","八","六","十","千","四","百","零"]
    result = ["1", "7", "10000", "3", "9", "2", "5", "100000000", "8", "6", "10", "1000", "4", "100", "0"]
    return result[pre]


for image_index in range(10):
    # 预测
    # pil_im = image.img_to_array(X_test[image_index])
    # pil_im = np.expand_dims(pil_im, axis=0)
    # pred = model.predict(X_test[image_index])

    # 显示
    ax = plt.subplot(int(np.ceil(10 / 5)), 5, image_index + 1)

    ax.set_title('predict: {}'.format(resultPrint(test_predicted[image_index])), font='Consolas')

    plt.imshow(X_test[image_index].reshape(64, 64))
    # plt.savefig("predict_num.jpg")

plt.subplots_adjust(hspace=1.5)
plt.show()
