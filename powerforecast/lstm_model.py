"""
@Project   : MachineLearningNoteBooks
@Module    : ml_model.py
@Author    : HjwGivenLyy [1752929469@qq.com]
@Created   : 1/17/19 1:43 PM
@Desc      : LSTM algorithm
"""

import time
import typing
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from base import FILE_PATH
from base import evaluation_criteria_model, split_train_test_data

DATA_PATH = FILE_PATH + "data.csv"

np.random.seed(5)


def load_data():
    """get date from file library"""
    data = pd.read_csv(DATA_PATH)
    rtn = data.values.astype('float32')
    return rtn


def get_key_lst_by_value(aim_value, data_dct: dict) -> list:
    """根据字典value值获取key值"""
    key_lst = []

    for key, value in data_dct.items():
        if value == aim_value:
            key_lst.append(key)

    return key_lst


def load_data_feature_lstm():
    """get date from file library"""
    data = pd.read_csv(FILE_PATH)
    data.set_index(["year_month"], inplace=True)
    data.index.name = 'year_month'
    rtn = data.values.astype('float32')
    return rtn


def deal_data(data: np.array,
              look_back: int = 1) -> typing.Tuple[np.array, np.array]:
    """convert an array of values into a data set matrix"""

    data_x, data_y = [], []

    for i in range(len(data) - look_back):
        data_x.append(data[i:(i + look_back), 0])
        data_y.append(data[i + look_back, 0])

    return np.array(data_x), np.array(data_y)


def min_max_scale_data(data: np.array):
    """normalize the data"""
    scale_model = MinMaxScaler(feature_range=(0, 1))
    return scale_model.fit_transform(data), scale_model


def series_to_supervised(data, n_in: int = 1, n_out: int = 1,
                         drop_nan: bool = True):

    n_vars = 1 if isinstance(data, list) else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []

    # input sequence (t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if drop_nan:
        agg.dropna(inplace=True)

    return agg


def create_lstm_model(look_back: int=1, eval_type: str='mape'):
    """
    create a lstm model without feature
    :param look_back:
    :param eval_type: model evaluation method
    :return:
    """

    # get data from file library
    origin_data = load_data()

    # Z-score standardization of origin data
    scale_data, scale_model = min_max_scale_data(data=origin_data)

    # Divide the data set into a training set and a test set
    # (not considering the validation set)
    train, test = split_train_test_data(feature_x=scale_data, test_size=0.8)

    # convert an array of values into a data set matrix
    x_train, y_train = deal_data(data=train, look_back=look_back)
    x_test, y_test = deal_data(data=test, look_back=look_back)

    # lstm X: [samples, time steps, features]，so transform data
    train_x = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    test_x = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    # create lstm network model
    lstm_model = Sequential()
    lstm_model.add(LSTM(4, input_shape=(1, look_back)))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')

    # model train
    lstm_model.fit(train_x, y_train, epochs=1000, batch_size=64, verbose=2)

    # model predict
    predict_y = lstm_model.predict(test_x)

    # model evaluation
    true_y = scale_model.inverse_transform([y_test])[0]
    predict_y = scale_model.inverse_transform(predict_y)
    predict_y = predict_y.reshape(predict_y.shape[0])

    eval_value = evaluation_criteria_model(true_y=true_y, predict_y=predict_y,
                                           eval_type=eval_type)
    print(eval_value)

    return eval_value


def create_feature_lstm_model():
    """train a lstm model with features"""

    # get data from file library
    origin_data = load_data_feature_lstm()

    # Z-score standardization of origin data
    scale_data, scale_model = min_max_scale_data(data=origin_data)

    # Constructed into a supervised learning problem
    reframe_data = series_to_supervised(data=scale_data, n_in=1, n_out=1)

    # Discard columns we don't want to predict
    delete_cols = [i for i in range(19, 36)]
    reframe_data.drop(reframe_data.columns[delete_cols], axis=1, inplace=True)

    # Divide data into training and test sets
    values = reframe_data.values
    n_train_month = 59
    train = values[:n_train_month, :]
    test = values[n_train_month:, :]
    train_x, train_y = train[:, :-1], train[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]

    # LSTM X: [samples, time steps, features]，so transform data
    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

    # Design network structure
    # 20, (64, 128)
    model = Sequential()
    model.add(LSTM(20, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # Fitting network
    history = model.fit(train_x, train_y, epochs=850, batch_size=128,
                        validation_data=(test_x, test_y), verbose=2,
                        shuffle=False)
    # Plot history loss
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    # Give prediction
    y_predict = model.predict(test_x)
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))

    # Reverse scaling predict value
    inv_y_predict = np.concatenate((y_predict, test_x[:, 1:]), axis=1)
    inv_y_predict = scale_model.inverse_transform(inv_y_predict)
    inv_y_predict = inv_y_predict[:, 0]
    print("inv_y_predict = ", inv_y_predict)

    # Reverse scaling actual value
    test_y_true = test_y.reshape((len(test_y), 1))
    inv_y_true = np.concatenate((test_y_true, test_x[:, 1:]), axis=1)
    inv_y_true = scale_model.inverse_transform(inv_y_true)
    inv_y_true = inv_y_true[:, 0]
    print("inv_y_true = ", inv_y_true)

    # Calculate rmse and mape
    rmse = sqrt(mean_squared_error(inv_y_true, inv_y_predict))
    print('Test RMSE: %.3f' % rmse)
    mape = evaluation_criteria_model(inv_y_true, inv_y_predict)
    print('Test MAPE: %.3f' % mape)


def train_parameters_model(param_str, train_x, train_y, test_x, test_y,
                           n_step, n_features, scale_model, is_plot: bool=False):
    """

    :param param_str: params consists str "_"
    :param train_x:
    :param train_y:
    :param test_x:
    :param test_y:
    :param n_step: time step
    :param n_features: features number
    :param scale_model:
    :param is_plot:
    :return:
    """

    print("start train params = {}".format(param_str))
    time.sleep(1)
    param_lst = param_str.split("_")

    # Design network structure
    # 20, (64, 128)
    model = Sequential()
    model.add(LSTM(
        int(param_lst[0]), input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # Fitting network
    history = model.fit(
        train_x, train_y, epochs=int(param_lst[1]), batch_size=int(param_lst[2]),
        verbose=2, validation_data=(test_x, test_y), shuffle=False)

    if is_plot:
        # Plot history loss
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

    # Give prediction
    y_predict = model.predict(test_x)
    test_x = test_x.reshape((test_x.shape[0], n_step*n_features))

    # Reverse scaling predict value
    inv_y_predict = np.concatenate(
        (y_predict, test_x[:, -(n_features - 1):]), axis=1)
    inv_y_predict = scale_model.inverse_transform(inv_y_predict)
    inv_y_predict = inv_y_predict[:, 0]

    # Reverse scaling actual value
    test_y_true = test_y.reshape((len(test_y), 1))
    inv_y_true = np.concatenate(
        (test_y_true, test_x[:, -(n_features - 1):]), axis=1)
    inv_y_true = scale_model.inverse_transform(inv_y_true)
    inv_y_true = inv_y_true[:, 0]

    return evaluation_criteria_model(inv_y_true, inv_y_predict)


def create_feature_lstm_model_lag_n(neurons_num: int, epochs: int,
                                    batch_size: int, n_step: int=1):

    # step1: prepare data
    # get data from file library
    origin_data = load_data_feature_lstm()

    # Z-score standardization of origin data
    scale_data, scale_model = min_max_scale_data(data=origin_data)

    # Constructed into a supervised learning problem
    reframe_data = series_to_supervised(data=scale_data, n_in=n_step, n_out=1)

    # Divide data into training and test sets
    values = reframe_data.values
    n_train_month = 59
    train = values[:n_train_month, :]
    test = values[n_train_month:, :]

    # split data to input and output
    n_features = int(reframe_data.values.shape[1] / (n_step + 1))
    n_obs = n_features * n_step
    train_x, train_y = train[:, :n_obs], train[:, -n_features]
    test_x, test_y = test[:, :n_obs], test[:, -n_features]

    # LSTM X: [samples, time steps, features]，so transform data
    train_x = np.reshape(train_x, (train_x.shape[0], n_step, n_features))
    test_x = np.reshape(test_x, (test_x.shape[0], n_step, n_features))

    # Design network structure
    # 20, (64, 128)
    model = Sequential()
    model.add(LSTM(neurons_num, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # Fitting network
    history = model.fit(
        train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=2,
        validation_data=(test_x, test_y), shuffle=False)

    # Plot history loss
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    # Give prediction
    y_predict = model.predict(test_x)
    test_x = test_x.reshape((test_x.shape[0], n_step * n_features))

    # Reverse scaling predict value
    inv_y_predict = np.concatenate(
        (y_predict, test_x[:, -(n_features - 1):]), axis=1)
    inv_y_predict = scale_model.inverse_transform(inv_y_predict)
    inv_y_predict = inv_y_predict[:, 0]
    print("inv_y_predict = ", inv_y_predict)

    # Reverse scaling actual value
    test_y_true = test_y.reshape((len(test_y), 1))
    inv_y_true = np.concatenate(
        (test_y_true, test_x[:, -(n_features - 1):]), axis=1)
    inv_y_true = scale_model.inverse_transform(inv_y_true)
    inv_y_true = inv_y_true[:, 0]
    print("inv_y_true = ", inv_y_true)

    # Calculate rmse and mape
    rmse = sqrt(mean_squared_error(inv_y_true, inv_y_predict))
    print('Test RMSE: %.3f' % rmse)
    mape = evaluation_criteria_model(inv_y_true, inv_y_predict)
    print('Test MAPE: %.3f' % mape)


def create_feature_lstm_model_lag_n_with_tune_params(n_step: int=1):

    # step1: prepare data
    # get data from file library
    origin_data = load_data_feature_lstm()

    # Z-score standardization of origin data
    scale_data, scale_model = min_max_scale_data(data=origin_data)

    # Constructed into a supervised learning problem
    reframe_data = series_to_supervised(data=scale_data, n_in=n_step, n_out=1)

    # Divide data into training and test sets
    values = reframe_data.values
    n_train_month = 59
    train = values[:n_train_month, :]
    test = values[n_train_month:, :]

    # split data to input and output
    n_features = int(reframe_data.values.shape[1] / (n_step + 1))
    n_obs = n_features * n_step
    train_x, train_y = train[:, :n_obs], train[:, -n_features]
    test_x, test_y = test[:, :n_obs], test[:, -n_features]

    # LSTM X: [samples, time steps, features]，so transform data
    train_model_x = np.reshape(train_x, (train_x.shape[0], n_step, n_features))
    test_model_x = np.reshape(test_x, (test_x.shape[0], n_step, n_features))

    # step2: choose the best model
    # Select best parameters for network
    neurons_num_lst = [i for i in range(10, 60, 10)]
    epochs_lst = [i for i in range(100, 1100, 200)]
    batch_size_lst = [64, 128]

    mape_dict = {}

    for neurons_num in neurons_num_lst:
        for epochs in epochs_lst:
            for batch_size in batch_size_lst:

                # start parameters selection
                params_lst = [str(neurons_num), str(epochs), str(batch_size)]
                mape_key = '_'.join(params_lst)

                # Calculate mape
                mape_dict[mape_key] = train_parameters_model(
                    param_str=mape_key, train_x=train_model_x, train_y=train_y,
                    test_x=test_model_x, test_y=test_y, n_step=n_step,
                    n_features=n_features, scale_model=scale_model)

    sorted_mape_dict = sorted(mape_dict.items(), key=lambda x: x[1],
                              reverse=False)

    print("min mape = {0}".format(sorted_mape_dict[0][1]))
    best_params_lst = get_key_lst_by_value(
        aim_value=sorted_mape_dict[0][1], data_dct=mape_dict)

    if len(best_params_lst) == 1:
        best_params_str = best_params_lst[0]
        print("best_params_str = {0}".format(best_params_str))
    else:
        best_params_strs = ','.join(best_params_lst)
        best_params_str = np.random.choice(best_params_lst)
        print("best_params_str include {0}".format(best_params_strs))

    # reshape data to lstm
    train_x = np.reshape(train_x, (train_x.shape[0], n_step, n_features))
    test_x = np.reshape(test_x, (test_x.shape[0], n_step, n_features))

    mape = train_parameters_model(
        param_str=best_params_str, train_x=train_x, train_y=train_y,
        test_x=test_x, test_y=test_y, n_step=n_step, n_features=n_features,
        scale_model=scale_model, is_plot=True)

    print('Test MAPE: %.3f' % mape)


def plot_origin_data():
    """plot origin data"""
    data = load_data()
    plt.plot(data)
    plt.show()


if __name__ == "__main__":
    # create_lstm_model(look_back=10)
    # create_feature_lstm_model_lag_n_with_tune_params(n_step=3)
    # n_step = 3 -> best parameters: neurons_num=35, epochs=100, batch_size=64
    # create_feature_lstm_model_lag_n(
    #     neurons_num=35, epochs=100, batch_size=128, n_step=3)

    create_feature_lstm_model_lag_n(20, 115, 64, 2)
