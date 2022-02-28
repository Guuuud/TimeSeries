import pandas as pd
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM
from keras.layers import GRU
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.models import Model, load_model
import warnings
from keras.layers import LSTM, Dense, Activation, TimeDistributed, Dropout, Lambda, RepeatVector, Input, Reshape, Concatenate, Dot
warnings.filterwarnings("ignore")
from keras.layers import Dense, LSTM, Dropout, Bidirectional,Conv1D,SimpleRNN,Conv1D,Flatten,GRU

# 1、load data
df = pd.read_csv('/Users/lee/Desktop/杜:时序预测/zbcj.csv')
# print(df)
print(df)
# # 2、process data
# pyplot.plot(df['推力平均值'])
# pyplot.show()
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(np.reshape(df['zdcj'].values, (df.shape[0], 1)))
print("The shape of dataset is:", dataset.shape)
print(dataset)

def create_dataset(dataset, step):
    dataX, dataY = [], []
    for i in range(len(dataset) - step - 1):
        a = dataset[i:(i + step), 0]
        dataX.append(a)
        dataY.append(dataset[i + step, 0])
    return np.array(dataX), np.array(dataY)


def create_dataset_2(dataset, step):
    dataX, dataY = [], []
    for i in range(len(dataset) - step - 1):
        a = dataset[i:(i + step), 0]
        dataX.append(a)
        dataY.append(dataset[i + step, 0])
        dataY.append(dataset[i + step + 1, 0])
    return np.array(dataX), np.array(dataY)


step = 25
dataX, dataY = create_dataset(dataset, step)
print("the shape of dataX is",dataX.shape)
print("the shape of dataY is",dataY.shape)
print(dataY)
# dataY是真实数据，X是用于测试的数据
test_size = int(dataX.shape[0] * 0.3)
print("test size的大小是", test_size)
train_size = dataX.shape[0] - test_size
# val_size = test_size
print("train size的大小是", train_size)
X_train = dataX[0:train_size, ]
X_test = dataX[train_size:train_size+test_size,]
# Y_train = dataY[0:train_size, ]
# X_val = dataX[train_size+test_size:train_size+test_size*2]

def Y_train_mul_reshape(dataY,size):
    Y_train = []
    for i in range(size):
        a = []
        a.append(dataY[i])
        a.append(dataY[i+1])
        a.append(dataY[i+2])
        a.append(dataY[i+3])
        a.append(dataY[i+4])
        Y_train.append(a)
    return Y_train

Y_train = Y_train_mul_reshape(dataY,train_size)



Y_test = dataY[train_size:dataX.shape[0], ]


# print("Before reshape, X_train's shape is", X_train.shape, " Y_train's shape is", Y_train.shape)
print(Y_train)
real_data = dataY[step:dataX.shape[0], ]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
# X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))
Y_train = np.array(Y_train)
print(Y_train.shape)

print("After reshape, X_train's shape is", X_train.shape, " Y_train's shape is", Y_train.shape)

Y_test = np.reshape(Y_test, (Y_test.shape[0], 1))
real_data = np.reshape(real_data, (real_data.shape[0], 1))

print("X_train's shape is", X_train.shape, " Y_train's shape is", Y_train.shape)
print("X_test's shape is", X_test.shape, " Y_test's shape is", Y_test.shape)

# print(Y_train)
# 3、model train
# ********************
model = Sequential()
ip = Input(shape=(X_train.shape[1], X_train.shape[2]))
print("-----------！！！！！！！！",ip.shape)
conv = Conv1D(32, 4, padding='same')(ip)
print("---------conv shape!!!",conv.shape)
conv = Conv1D(64, 4, padding='same')(conv)
print("---------conv shape!!!",conv.shape)
conv = Conv1D(128, 4, padding='same')(conv)
print("---------conv shape!!!",conv.shape)
model.add(GRU(25))
model.add(Dropout(0.5))
model.add(Dense(10))  # dropout层防止过拟合
model.add(Dense(5))  # 全连接层
model.compile(optimizer=RMSprop(), loss='mse', metrics=['accuracy'], )
# ************
# model = Sequential()
# model.add(SimpleRNN(24))
# model.add(Dropout(0.5))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
# model.compile(optimizer=RMSprop(), loss='mse')
# ********************

print("----", len(model.layers))
model.fit(X_train, Y_train, nb_epoch=300, batch_size=10, verbose=2)

# test size not train
Y_predict = model.predict(X_test)
print("最后产生的结果")
print(Y_predict.shape)
# 由normalization变为原来的数值
Y_predict = scaler.inverse_transform(Y_predict)
print(Y_predict)


np.reshape(Y_test, (Y_test.shape[0], 1))
Y_test = scaler.inverse_transform(Y_test)
Y_train = scaler.inverse_transform(Y_train)

# Y_predict = np.reshape(Y_predict, (Y_predict.shape[0],))

Y_test = np.reshape(Y_test, (Y_test.shape[0],))
real_data = np.reshape(real_data, (real_data.shape[0],))
# Y_train = np.reshape(Y_train, (Y_train.shape[0],))

Y_predict_2 = []
for i in range(Y_predict.shape[0]):
    Y_predict_2.append(Y_predict[i][3])

# real data
pyplot.xlabel("last 30% Huanhao")
pyplot.ylabel("ZSPJZ")
print("在画图之前Y_predict和Y_test的形状，Y_predict的是", Y_predict.shape, "Y_test的是", Y_test.shape)
pyplot.plot(Y_predict_2, label='Predict')
pyplot.plot(Y_test, label='Reality')
pyplot.legend(["Predict", "Reality"], loc="upper left")
pyplot.title("ZSPJZ")
pyplot.show()
