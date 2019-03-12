"""
@Project   : PowerForecast
@Module    : ml_model.py
@Author    : HjwGivenLyy [1752929469@qq.com]
@Created   : 1/16/19 10:59 AM
@Desc      : Building a machine learning model for power data
"""

import typing

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import svm
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from base import FILE_PATH
from base import evaluation_criteria_model, split_train_test_data

ELE_DATA = FILE_PATH + "data.csv"
LOG_ELE_DATA = FILE_PATH + "lag_data.csv"

np.random.seed(5)


def load_data():
    """get date from file library"""

    # read data
    ele_data = pd.read_csv(ELE_DATA)
    column_name_lst = list(ele_data.columns)
    feature_cols = column_name_lst[2:]
    label_col = column_name_lst[1]

    # 将数据分为 feature 和 label
    x, y = ele_data[feature_cols], np.array(ele_data[label_col], dtype=float)

    return x, y


def standardization_df(df: pd.DataFrame,
                       columns_lst: list=list()) -> pd.DataFrame:
    """
    对df进行z-score标准化
    :param df: 原始数据
    :param columns_lst: 需要进行 z-score 的列字段名
    :return: pd.DataFrame
    """

    def center_series(x: pd.Series):

        return (x - np.mean(x)) / (np.std(x))

    if columns_lst:
        for columns_name in columns_lst:
            df[columns_name] = center_series(df[columns_name])
        return df
    else:
        return df.apply(lambda x: (x - np.mean(x)) / (np.std(x)))


def standard_scale(x: np.array, y: np.array=None):
    """对数据进行z-score标准化"""
    scale_x_model = StandardScaler()
    scale_x = scale_x_model.fit_transform(x)
    if y is not None:
        scale_y_model = StandardScaler()
        scale_y = scale_y_model.fit_transform(y.reshape(-1, 1))
        return scale_x, scale_x_model, scale_y, scale_y_model
    return scale_x, scale_x_model


def data_reducer(x: np.array, reduce_type: str="pca"):
    """
    对训练数据进行降维
    :param x: [[5.8, 3.2, 1.6, 0.6], [4.8, 2.7, 1.6, 0.4], ... ]
    :param reduce_type: pca --> 默认，svd --> truncated svd
    :return: [[-2.09156169, 0.50223545, -0.37300618],
               [-2.48378571, -0.50421932, -0.20003013], ... ]
    """

    reducer_model = PCA(0.95)
    # reducer_model = PCA(n_components=10)  # 保留前10个主成分
    reduced_x = reducer_model.fit_transform(x)

    if reduce_type == "svd":
        n_components = min(int(x.shape[1] * 0.80), reduced_x.shape[1])
        svd = TruncatedSVD(n_components=n_components)
        reduced_x = svd.fit_transform(x)

    return reduced_x, reducer_model


def select_ridge_params(features, label) -> float:
    """select ridge regression parameters: alpha"""

    ridge_params = {'alpha': [0.05, 0.1, 0.3, 1., 3., 5., 10., 15., 30., 50.]}

    ridge_search = GridSearchCV(
        estimator=Ridge(), param_grid=ridge_params,
        scoring='neg_mean_squared_error', cv=5, n_jobs=4, iid=False)

    ridge_search.fit(features, label)
    ridge_best_params = ridge_search.best_params_

    return ridge_best_params["alpha"]


def select_lasso_params(features, label) -> float:
    """select lasso regression parameters: alpha"""

    lasso_params = {'alpha': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]}

    lasso_search = GridSearchCV(
        estimator=Lasso(), param_grid=lasso_params,
        scoring='neg_mean_squared_error', cv=5, n_jobs=4, iid=False)

    lasso_search.fit(features, label)
    lasso_best_params = lasso_search.best_params_

    return lasso_best_params["alpha"]


def select_svm_params(features, label) -> typing.Tuple[int, float]:
    """select svm parameters: C and gamma"""

    svm_params = {
        'C': [i + 1 for i in range(0, 10, 1)],
        'gamma': [1e-3*i for i in range(10, 1010, 10)]}

    svm_search = GridSearchCV(
        estimator=svm.SVR(kernel='rbf'), param_grid=svm_params,
        scoring='neg_mean_squared_error', cv=5, n_jobs=4, iid=False)

    svm_search.fit(features, label)
    svm_best_params = svm_search.best_params_

    return svm_best_params["C"], svm_best_params["gamma"]


def select_mlp_params(features, label) -> typing.Tuple[tuple, int]:
    """select mlp parameters: hidden_layer_sizes and batch_size"""

    mlp_params = {
        'hidden_layer_sizes': [(2**(i+2), ) for i in range(0, 4)],
        'batch_size': [2**(i+2) for i in range(0, 3)]}

    mlp_search = GridSearchCV(
        estimator=MLPRegressor(), param_grid=mlp_params,
        scoring='neg_mean_squared_error', cv=5, n_jobs=4, iid=False)

    mlp_search.fit(features, label)
    mlp_best_params = mlp_search.best_params_

    return mlp_best_params["hidden_layer_sizes"], mlp_best_params["batch_size"]


def select_xgboost_params(features, label) -> typing.Tuple[float, int, int]:
    """
    select xgboost parameters:
        learning_rate, n_estimators, max_depth
    """

    xgboost_params = {
        'learning_rate': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
        'n_estimators': [i for i in range(10, 110, 10)],
        'max_depth': [i for i in range(2, 10, 1)]}

    xgboost_search = GridSearchCV(
        estimator=xgb.XGBRegressor(
            min_child_weight=1, gamma=0, colsample_bytree=0.8,
            objective='reg:linear', nthread=4, scale_pos_weight=1, seed=123),
        param_grid=xgboost_params,
        scoring='neg_mean_squared_error', cv=5, n_jobs=4, iid=False)

    xgboost_search.fit(features, label)
    mlp_best_params = xgboost_search.best_params_

    return mlp_best_params["learning_rate"], mlp_best_params["n_estimators"], \
           mlp_best_params["max_depth"]


def select_model(features_data, label_data, model_type: str="svm"):
    """
    use suitable model
    :param features_data: feature data
    :param label_data: label data
    :param model_type: model type
    :return:
    """

    model = None

    if model_type == "svm":
        c, gamma = select_svm_params(features=features_data, label=label_data)
        model = svm.SVR(kernel='rbf', C=c, gamma=gamma)
    elif model_type == "mlp":
        hidden_layer_sizes, batch_size = select_mlp_params(
            features=features_data, label=label_data)
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                             batch_size=batch_size)
    elif model_type == "xgboost":
        learning_rate, n_estimators, max_depth = select_xgboost_params(
            features=features_data, label=label_data)
        model = xgb.XGBRegressor(
            learning_rate=learning_rate, n_estimators=n_estimators,
            max_depth=max_depth, min_child_weight=1, gamma=0,
            colsample_bytree=0.8, objective='reg:linear', nthread=4,
            scale_pos_weight=1, seed=123)
    elif model_type == "ridge":
        model = Ridge(alpha=select_ridge_params(
            features=features_data, label=label_data))
    elif model_type == "lasso":
        model = Lasso(alpha=select_lasso_params(
            features=features_data, label=label_data))

    return model


def create_model(model_type: str='svm', reduce_type: str='pca',
                 eval_type: str='mape') -> float:
    """
    create a ml model
    :param model_type: select model
    :param reduce_type: data dimension reduction method
    :param eval_type: model evaluation method
    :return:
    """

    # get data from file library
    x, y = load_data()

    # Divide the data set into a training set and a test set
    # (not considering the validation set)
    x_train, x_test, y_train, y_test = split_train_test_data(
        feature_x=x, label_y=y)

    # Z-score standardization of origin data
    x_train_scale, scale_x_model, y_train_scale, scale_y_model = standard_scale(
        x=x_train, y=y_train)
    x_test_scale = scale_x_model.transform(x_test)
    y_test_scale = scale_y_model.transform(y_test.reshape(-1, 1))

    # Dimension reduction of standardized data sets： PCA or TruncatedSVD
    reduced_train_x, reduced_model = data_reducer(
        x=x_train_scale, reduce_type=reduce_type)
    reduced_test_x = reduced_model.transform(x_test_scale)

    # create model
    ele_model = select_model(features_data=reduced_train_x,
                             label_data=y_train_scale.ravel(),
                             model_type=model_type)

    # model train
    ele_model.fit(reduced_train_x, y_train_scale.ravel())
    rbf_svr_predict = ele_model.predict(reduced_test_x)

    # model evaluation
    y_true = scale_y_model.inverse_transform(y_test_scale)
    y_true = y_true.reshape(y_true.shape[0])
    y_predict = scale_y_model.inverse_transform(rbf_svr_predict)

    eval_value = evaluation_criteria_model(true_y=y_true, predict_y=y_predict,
                                           eval_type=eval_type)

    return eval_value


if __name__ == "__main__":

    eval_value1 = create_model(model_type="xgboost", eval_type="mape")
    print(eval_value1)
