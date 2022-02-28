import pandas as pd
import numpy as np
from sklearn import metrics
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

filepath = '/Users/lee/Desktop/Project/PRSA_data_2010.1.1-2014.12.31.csv'  # 文件路径
data = pd.read_csv(filepath, index_col=0)
data.shape
data.head(10)

# 查看缺失值信息
data.isnull().sum()
data["pm2.5"] = data.groupby(['year', 'month'])['pm2.5'].transform(lambda x: x.fillna(x.mean()))
data.isnull().sum()
# 查看填充后数据缺失情况
cbwd_category = data['cbwd'].astype('category')
# 使用标签的编码作为真正的数据
data['cbwd'] = cbwd_category.cat.codes
X_data = data[['pm2.5', 'DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir']]  # 提取特征数据
Y_data = data[['pm2.5']]  # 提取pm2.5数据

X = np.zeros((X_data.shape[0] // 24 - 1,
             24,
              X_data.shape[-1]))
Y = np.zeros((Y_data.shape[0] // 24 - 1,
              2))
rows = range(0, X_data.shape[0] - 24, 24)
for i, row in enumerate(rows):
    X[i, :, :] = X_data.iloc[row : row + 24]
    Y[i, :] = [Y_data.iloc[row + 24],  Y_data.iloc[row + 25]]
print(X.shape)
print(Y.shape)

# 这里将80%数据作为训练集，20%数据作为测试集，则训练集有1460个样本，测试集有365个样本
X_train = data[:int(X.shape[0] * 0.8)]
Y_train = data[:int(X.shape[0] * 0.8)]
print(X_train.shape)
X_val = X[int(X.shape[0] * 0.8)::]
Y_val = Y[int(X.shape[0] * 0.8)::]

X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
Y_mean = Y_train.mean(axis=0)
Y_std = Y_train.std(axis=0)

X_train_norm = (X_train - X_mean) / X_std
Y_train_norm = (Y_train - Y_mean) / Y_std
X_val_norm = (X_val - X_mean) / X_std
Y_val_norm = (Y_val - Y_mean) / Y_std
print(X_train_norm.shape)
# 使用3层LSTM，输出层为2输出的Dense层
model = Sequential()
model.add(LSTM(32,
               input_shape=(X_train_norm.shape[1], X_train_norm.shape[-1]),
               return_sequences=True))
model.add(Dropout(0.2))  # 防止过拟合
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))

model.add(Dense(2))

model.compile(loss='mse',
              optimizer='rmsprop')
model.summary()

history = model.fit(X_train_norm, Y_train_norm,
                    epochs=60,
                    batch_size=128,
                    validation_data=(X_val_norm, Y_val_norm))

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(range(len(loss)), loss, 'b-', label='训练集损失')
plt.plot(range(len(loss)), val_loss, 'r-', label='测试集损失')
plt.legend(loc='best')
plt.show();

model_pred = model.predict(X_val_norm)
val_pred = model_pred * Y_std + Y_mean  # 别忘了，数据进行了标准化处理，因此预测值需要处理，再计算R方

# 计算R2
R_2_0 = metrics.r2_score(Y_val[:,0], val_pred[:, 0])  # 计算0时预测的R方
R_2_1 = metrics.r2_score(Y_val[:,1], val_pred[:, 1])  # 计算1时预测的R方

# 实际值与预测值对比图
fig = plt.subplots(figsize=(12, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0)

ax1 = plt.subplot(gs[0])
plt.plot(range(Y_val.shape[0]), Y_val[:, 0], 'b-', label='0时PM2.5实际图')
plt.plot(range(Y_val.shape[0]), val_pred[:, 0], 'r-', label='0时PM2.5预测图')
plt.legend(loc='best')
plt.text(150, 400, '拟合R2：{0}%'.format(round(R_2_0 * 100, 2)))

ax2 = plt.subplot(gs[1], sharex=ax1)
plt.plot(range(Y_val.shape[0]), Y_val[:, 1], 'b-', label='1时PM2.5实际图')
plt.plot(range(Y_val.shape[0]), val_pred[:, 1], 'r-', label='1时PM2.5预测图')
ax2.set_xlabel(' ')
plt.legend(loc='best')
plt.text(150, 400, '拟合R2：{0}%'.format(round(R_2_1 * 100, 2)))
plt.show();
