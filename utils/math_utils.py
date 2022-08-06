import numpy as np
import torch

import datetime
from chinese_calendar import is_workday


def extract_time_feature(str_time):
    # 时间特征：月份12 + 星期7 + 第几天366 + 当天节假日否2 + 当前的小时数据24 = 411
    # 修改为：
    # 时间特征：月份12 + 星期7 + 月份的第几天31 + 当天节假日否2 + 当前的小时数据24 = 76
    # 修改为：
    # 时间特征：月份12 + 星期7 + 月份的第几天31 + 当天节假日否2 + 当前的小时数据24 = 5个特征

    # 2020-04-15 00:00:00
    cur_time = datetime.datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S')
    month_feature = cur_time.month - 1

    week_feature = cur_time.weekday()  # 星期几，从0开始计数

    day_feature = cur_time.day - 1

    holiday_feature = 1
    if is_workday(cur_time):
        holiday_feature = 0

    hour_feature = cur_time.hour

    feature = [month_feature, week_feature, day_feature, holiday_feature, hour_feature]

    return np.array(feature)


def z_score(x, mean, std):
    """Z-score normalization function: $z = (X - \mu) / \sigma $,
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: np.ndarray, input array to be normalized.
    :param mean:  float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score normalized array.
    """

    return (x - mean) / std


def z_inverse(x, mean, std):
    # The inverse of function z_score().
    # x: np.ndarray, input to be recovered.
    # mean: float, the value of mean.
    # std: float, the value of standard deviation.
    # return: np.ndarray, z-score inverse array.

    return x * std + mean


def mape(v, v_):
    """Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, mape averages on all elements of input.
    """
    if torch.is_tensor(v):
        return torch.mean(torch.abs((v_ - v) / (v + 1e-5))) * 100
    else:
        return np.mean(np.abs(v_ - v) / (v + 1e-5)) * 100

def smape(v, v_):
    """对称平均绝对百分比误差（Symmetric Mean Absolute Percentage Error）.
    注意点：当真实值有数据等于0，而预测值也等于0时，存在分母0除问题，该公式不可用！
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, mape averages on all elements of input.
    """
    if torch.is_tensor(v):
        return torch.mean(2.0 * torch.abs(v_ - v) / (torch.abs(v_) + torch.abs(v))) * 100
    else:
        return np.mean(2.0 * np.abs(v_ - v) / (np.abs(v_) + np.abs(v))) * 100


def mse(v, v_):
    """Mean squared error.
    :param v:  np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, mse averages on all elements of input.
    """
    if torch.is_tensor(v):
        return torch.mean((v_ - v) ** 2)
    else:
        return np.mean((v_ - v) ** 2)

def rmse(v, v_):
    """Root mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, rmse averages on all elements of input.
    """

    if torch.is_tensor(v):
        return torch.sqrt(torch.mean((v_ - v) ** 2))
    else:
        return np.sqrt(np.mean((v_ - v) ** 2))


def mae(v, v_):
    """Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, mae averages on all elements of input."""

    if torch.is_tensor(v):
        return torch.mean(torch.abs(v_ - v))
    else:
        return np.mean(np.abs(v_ - v))


def evaluation(y, y_, x_stats):
    """Evaluation function: interface to calculate MAPE, MAE and RMSE between ground truth and prediction.
    Extended version: multi-step prediction can be calculated by self-calling.
    :param y: np.ndarray or int, ground truth.
    :param y_:  np.ndarray or int, prediction.
    :param x_stats: dict, paras of z-scores (mean & std).
    :return: np.ndarray, averaged metric values.
    """

    dim = len(y_.shape)

    if dim == 3:
        # single_step case
        v = z_inverse(y, x_stats['mean'], x_stats['std'])
        v_ = z_inverse(y_, x_stats['mean'], x_stats['std'])
        return np.array([mape(v, v_), mae(v, v_), mse(v, v_)])
    else:
        # multi_step case
        tmp_list = []
        # recursively call
        for i in range(y_.shape[0]):
            tmp_res = evaluation(y[i], y_[i], x_stats)
            tmp_list.append(tmp_res)
        return np.concatenate(tmp_list, axis=-1)
