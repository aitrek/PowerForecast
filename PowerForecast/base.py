"""
@Project   : PowerForecast
@Module    : base.py
@Author    : HjwGivenLyy [1752929469@qq.com]
@Created   : 3/12/19 1:47 PM
@Desc      : basic module of entire project
"""

import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


FILE_PATH = "/home/pauleta/pauleta-gauss/PowerForecast/PowerForecast/data/"


def evaluation_criteria_model(true_y: np.array, predict_y: np.array,
                              eval_type: str="mape") -> float:
    """模型评估准则"""
    rtn = 0.0

    if eval_type == "mape":
        rtn = np.mean(np.abs(true_y - predict_y) / true_y)
    elif eval_type == "mse":
        rtn = mean_squared_error(true_y, predict_y)
    elif eval_type == "mae":
        rtn = mean_absolute_error(true_y, predict_y)

    return rtn


def split_train_test_data(feature_x: np.array, label_y: np.array=None,
                          test_size: float=0.2):
    """将数据集分为 train 和 test """
    if label_y is not None:
        x_train, x_test, y_train, y_test = train_test_split(
            feature_x, label_y, test_size=test_size, random_state=123456)
        return x_train, x_test, y_train, y_test
    else:
        x_train, x_test = train_test_split(
            feature_x, test_size=test_size, random_state=123456)
        return x_train, x_test
