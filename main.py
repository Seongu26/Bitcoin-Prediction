import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error

csv_path = "https://raw.githubusercontent.com/curiousily/Deep-Learning-For-Hackers/master/data/3.stock-prediction/BTC-USD.csv"
df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
print(df)

df["Close"]['2010':].plot(figsize=(12,6))

plt.legend(['Bitcoin price all time'])
plt.title('Bitcoin price')
plt.show()

# scale data
scaler = MinMaxScaler()
close_price = df.Close.values.reshape(-1, 1)
datas = scaler.fit_transform(close_price)

SEQ_LEN = 100

def return_rmse(test,predicted):

    mse = mean_squared_error(test, predicted)

    print("The mean squared error is {}.".format(mse))


def to_sequences(data, seq_len):
    d = []

    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])

    return np.array(d)

def preprocess(data_raw, seq_len, train_split):

    data = to_sequences(data_raw, seq_len)

    num_train = int(train_split * data.shape[0])

    X_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]

    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = preprocess(datas, SEQ_LEN, train_split = 0.9)

print(X_train.shape)

print(X_test.shape)

# model building
DROPOUT = 0.2
WINDOW_SIZE = SEQ_LEN - 1

model = Sequential()

model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=True),
                        input_shape=(WINDOW_SIZE, X_train.shape[-1])))
model.add(Dropout(rate=DROPOUT))

model.add(Bidirectional(LSTM((WINDOW_SIZE*2), return_sequences=True)))
model.add(Dropout(rate=DROPOUT))

model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=False)))
model.add(Dense(units=1))
model.add(Activation('linear'))

# training
model.compile(
    loss='mean_squared_error',
    optimizer='adam'
)


history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=512,
    shuffle=False,
    validation_split=0.1
)

model.evaluate(X_test, y_test)

y_hat = model.predict(X_test)

y_test_inverse = scaler.inverse_transform(y_test)
y_hat_inverse = scaler.inverse_transform(y_hat)

plt.plot(y_test_inverse, label="Actual Price", color='green')
plt.plot(y_hat_inverse, label="Predicted Price", color='red')

plt.title('Bitcoin price prediction')
plt.xlabel('Time [days]')
plt.ylabel('Price')
plt.legend(loc='best')

plt.show()

return_rmse(y_test_inverse,y_hat_inverse)

# RMSE = 1021