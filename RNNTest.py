#用来画图形的
import pandas as pd
#用来处理数组，向量的
import numpy as np
#models处理模型的编译，提供模型训练，
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
#图形的勾画
import matplotlib.pyplot as plt
# %matplotlib inline

if __name__ == "__main__":

    # 生成数据
    t = np.arange(0, 1500)
    # 生成一个正弦波
    x = np.sin(0.02 * t) + np.random.rand(1500) * 2
    plt.plot(x)
    plt.show()
    # 数据集合的前1000个左右训练集合，后面的500个作为测试集合
    train, test = x[0:1000], x[1000:]

    step = 10
    # 通过step的方式切分X（观测值）和 Y（预测值）
    def convertToDataset(data, step):
        # data = np.append(data,np.repeat(data[-1,],step))
        X, Y = [], []
        startIndex = len(data) - step
        for i in range(startIndex):
            endIndex = i + step
            X.append(data[i:endIndex, ])
            Y.append(data[endIndex,])
        return np.array(X), np.array(Y)


    # 生成训练集合，测试集合 的X 和 Y
    trainX, trainY = convertToDataset(train, step)
    testX, testY = convertToDataset(test, step)

    #shape返回包含元素的元组
    #例如(n, )，其中n是数组的长度
    #二维数组，shape将返回(m, n)，其中m是数组的行数，n是数组的列数。
    print(trainX.shape)  #(990, 10) 990 行，10列
    print(trainY.shape)  #(990,) 990 行
    print(testX.shape)   #(490, 10) 490 行 10列
    print(testY.shape)   #(490,)   490 行

    #训练
    model = Sequential()
    model.add(SimpleRNN(units=64, activation="tanh"))
    model.add(Dense(1))
    # Mean Squared Error（均方误差）
    # optimizer='rmsprop'优化器，每次梯度运算之后都会更新最优的学习率
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    # %%
    # epochs=100 整个数据集将被训练100次。每一轮都包括前向传播和反向传播。
    # batch_size=16 模型每取16个样本进行训练，并更新一次权重。
    # verbose=2 每个epoch结束时打印一行日志
    history = model.fit(trainX, trainY, epochs=100, batch_size=16, verbose=2)

    #执行
    loss = history.history['loss']
    plt.plot(loss, label='Training loss')
    plt.legend()

    plt.show()