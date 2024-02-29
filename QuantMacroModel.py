import os
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

from QuantDataProcess import winsor_percentile
from sklearn.linear_model import LinearRegression

from statsmodels.tsa.seasonal import STL

from statsmodels.tsa.x13 import x13_arima_analysis

import matplotlib.pyplot as plt

# 设置pandas的显示选项
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.width', None)

# 用于季节性调整程序
path_sa = os.getcwd()  # 获取当前文件夹地址


def linear_regression_model(data, winsor=True):
    """
    使用前N-1或者前N-2期样本，对最后1期或者2期因变量做预测，即只预测1-2期
    只预测1个因变量
    只有最后1-2期因变量是预测值，前期因变量是实际值，所以可能会出现方向不一致的情况
    Args:
        data: DataFrame数据
        winsor: 是否截尾

    Returns: 拟合后数据

    """
    # 如果最新的样本期，有高频数据缺失，则丢弃
    if data.iloc[-1, :-1].isnull().values.any():
        # 如果有缺失值，丢弃最后一行数据
        data.drop(data.tail(1).index, inplace=True)

    # 对原始数据进行0-1标准化
    _mean = data.mean()
    _std = data.std()
    _data = (data - _mean) / _std
    _data.iplot(title='0-1标准化')

    print('原始数据均值：', _mean)
    print('原始数据标准差：', _std)

    # 增加一个对是否对数据进行异常值截尾的条件判断
    if winsor:

        # 异常值截尾，低: 2%，高: 98%
        _data = winsor_percentile(_data)
        _data.iplot(title='异常值截尾')

    else:

        print("本数据未进行异常值截尾")

    # 判断因变量是缺失1期还是2期
    if pd.isnull(_data.iloc[-2, -1]):

        # 选择前N-2行和前N-1列作为样本
        _x = _data.iloc[:-2, :-1]
        _y = _data.iloc[:-2, -1]

        # 创建线性回归模型，设定截距项为0
        model = LinearRegression(fit_intercept=False)

        # 拟合数据
        model.fit(_x, _y)

        # 打印系数和截距项
        print('线性回归系数：', model.coef_)
        print('线性回归截距项：', model.intercept_)
        print('线性回归拟合优度：', model.score(_x, _y))

        # 获取最后两行数据，作为预测样本
        _x_test = _data.iloc[-2:, :-1]

        print('预测样本：', _x_test)

        # 进行预测
        _y_pred = model.predict(_x_test)

        print('预测值：', _y_pred)

        # 填补预测值
        _data.iloc[-2:, -1] = _y_pred

        # 还原均值和标准差
        _data = _data * _std + _mean

        # 还原原始数据
        data.iloc[-2:, -1] = _data.iloc[-2:, -1]

        print('全部预测值：', model.predict(_data.iloc[:, :-1])[-6:])

    else:

        # 选择前N-1行和前N-1列作为样本
        _x = _data.iloc[:-1, :-1]
        _y = _data.iloc[:-1, -1]

        # 创建线性回归模型，设定截距项为0
        model = LinearRegression(fit_intercept=False)

        # 拟合数据
        model.fit(_x, _y)

        # 打印系数和截距项
        print('线性回归系数：', model.coef_)
        print('线性回归截距项：', model.intercept_)
        print('线性回归拟合优度：', model.score(_x, _y))

        # 获取最后一行数据，作为预测样本，要转换成DataFrame格式
        _x_test = _data.iloc[[-1], :-1]

        print('预测样本：', _x_test)

        # 进行预测
        _y_pred = model.predict(_x_test)

        print('预测值：', _y_pred)

        # 填补预测值
        _data.iloc[-1, -1] = _y_pred

        # 还原均值和标准差
        _data = _data * _std + _mean

        # 还原原始数据
        data.iloc[-1, -1] = _data.iloc[-1, -1]

        # 打印全部预测值
        print('全部预测值：', model.predict(_data.iloc[:, :-1])[-6:])

    print(data.tail(6))

    # 打印一条横线
    print('-' * 40)

    data.iplot(title='预测后结果', secondary_y=data.columns.values[-1])


def bry_boschan(y, h, c):
    """
    Bry-Boschan算法，对宏观经济指标的拐点  进行自动化识别的算法
    拐点被定义为同时满足以下条件的局部极值：
      1、峰和谷必须交替出现；
      2、相邻峰和谷的间隔（半周期）不小于h个月；
      3、相邻峰或相邻谷间隔（周期）不小于c个月；
    Args:
        y: 原始数据只包括1个宏观经济指标的DataFrame数据
        h: h和c为自定义参数，满足c≥2h，建议h和c分别取6和15
        c: h和c为自定义参数，满足c≥2h，建议h和c分别取6和15

    Returns: 拐点

    """
    # y是只包括1个宏观经济指标的DataFrame数据
    # 1. 找到所有局部最大值和最小值
    local_max = argrelextrema(np.array(y), np.greater)[0]
    local_min = argrelextrema(np.array(y), np.less)[0]

    # 2. 峰和谷必须交替出现
    turning_points = sorted(np.concatenate((local_max, local_min)))
    turning_points_values = [1 if i in local_max else -1 for i in turning_points]

    # 3. 相邻峰谷间隔（半周期）不小于h个月
    turning_points, turning_points_values = zip(
        *[(turning_points[i], turning_points_values[i]) for i in range(1, len(turning_points)) if
          turning_points[i] - turning_points[i - 1] >= h])

    # 4. 相邻峰或相邻谷间隔（周期）不小于c个月
    maxima = [tp for tp, tpv in zip(turning_points, turning_points_values) if tpv == 1]
    minima = [tp for tp, tpv in zip(turning_points, turning_points_values) if tpv == -1]
    maxima = [maxima[i] for i in range(1, len(maxima)) if maxima[i] - maxima[i - 1] >= c]
    minima = [minima[i] for i in range(1, len(minima)) if minima[i] - minima[i - 1] >= c]

    # 5. 返回结果
    turning_points_final = sorted(maxima + minima)
    turning_points_values_final = [1 if i in maxima else -1 for i in turning_points_final]

    return turning_points_final, turning_points_values_final


def bry_boschan_analysis(y, h=1, c=3):
    """
    Bry-Boschan算法应用
    Args:
        y: 只包括1个宏观经济指标的DataFrame数据
        h: 默认值为1
        c: 默认值为2

    Returns: 用不同颜色区分上行期和下行期的图

    """
    # 使用bry_boschan函数
    turning_points, turning_points_values = bry_boschan(y=y, h=h, c=c)

    y['turning_points'] = 0
    y['turning_points'].iloc[turning_points] = turning_points_values

    # 解决最新的值是缺失的问题，并赋值为与最近状态相反的状态（upswing or downswing）
    if y['turning_points'].iloc[-1] == 0:
        y['turning_points'].iloc[-1] = turning_points_values[-2]

    # 新建变量，命名为Phase，取值为upswing or downswing，代表上行期或者下行期
    y[y.columns[0] + 'Phase'] = y['turning_points'].apply(
        lambda x: 'upswing' if x == 1 else ('downswing' if x == -1 else np.nan))

    # 填补缺失值的状态
    y.bfill(inplace=True)

    y.drop(['turning_points'], axis=1, inplace=True)

    # 绘制宏观经济数据拐点的折线图
    plt.plot(y.iloc[:, 0], color='blue')
    # 根据上行期和下行期的值选择不同的颜色
    colors = {'upswing': 'red', 'downswing': 'green'}
    for phase, color in colors.items():
        plt.fill_between(y.index, 0, y.iloc[:, 0], where=y[y.columns[0] + 'Phase'] == phase,
                         color=color, alpha=0.3)
    # 作图字体设定
    plt.rcParams["font.family"] = "Microsoft YaHei"
    # 添加图例和标签
    plt.title(y.columns[0] + f', HalfCycle h={h}' + f', Cycle c={c}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    # 显示图形
    plt.show()


# def stl_sa(data, seasonal=13, period=52):
#     """
#     STL 是“使用 Loess 进行季节和趋势分解”的首字母缩写词，而 Loess 是一种估计非线性关系的方法
#     STL 方法由RB Cleveland、Cleveland、McRae 和 Terpenning ( 1990 )开发
#     STL 的原始数据不能出现Nan缺失值
#     在STL方法中，seasonal参数是用于设置季节性平滑器长度的
#     seasonal这个参数必须是奇数，通常应该大于等于7
#     seasonal这个参数的大小决定了季节性成分的平滑程度，数值越大，平滑程度越高，反之则更贴近原始数据
#     Args:
#         data: 包括宏观经济指标的DataFrame数据
#         seasonal: 必须是奇数，通常应该大于等于7
#         period: 默认周期性为52（周度数据）
#
#     Returns: 季调后的趋势项，也可以输出季调后序列（需改原始代码）
#
#     """
#     for i in range(len(data.columns)):
#         stl = STL(data.iloc[:, [i]], seasonal=seasonal, period=period)
#         res = stl.fit()
#         seasonal_sa, trend_sa, resid_sa = res.seasonal, res.trend, res.resid
#         # seasonal_sa = seasonal_sa.to_frame()
#         # seasonal_sa.columns.values[0] = _name
#         trend_sa = trend_sa.to_frame()
#         _name = data.iloc[:, [i]].columns.values[0]
#
#         trend_sa.columns.values[0] = _name
#         # trend_sa.name = _name
#         data.iloc[:, [i]] = trend_sa
#         return data


def stl_sa(data, seasonal=13, period=52):
    """
    针对周度数据

    STL 是“使用 Loess 进行季节和趋势分解”的首字母缩写词，而 Loess 是一种估计非线性关系的方法
    STL 方法由RB Cleveland、Cleveland、McRae 和 Terpenning ( 1990 )开发

    STL 的原始数据不能出现Nan缺失值
    在STL方法中，seasonal参数是用于设置季节性平滑器长度的
    seasonal这个参数必须是奇数，通常应该大于等于7
    seasonal这个参数的大小决定了季节性成分的平滑程度，数值越大，平滑程度越高，反之则更贴近原始数据

    Args:
        data: 包括宏观经济指标的DataFrame数据
        seasonal: 必须是奇数，通常应该大于等于7
        period: 默认周期性为52（周度数据）

    Returns: 季调后的趋势项，也可以输出季调后序列（需改原始代码）
    """
    # Ensure the input is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a DataFrame")

    # Create a new DataFrame to store the results
    result_df = data.copy()

    for col in data.columns:
        # Seasonal Adjustment Using STL
        # Correctly initialize STL with the series directly as the first positional argument
        stl = STL(result_df[col], seasonal=seasonal, period=period)
        res = stl.fit()
        trend_sa = res.trend

        # Update the DataFrame column with the trend component
        result_df[col] = trend_sa

    return result_df


# 把生成的春节数据dat文件转换成DataFrame格式
def generate_spring_festival_dates(name):
    with open('{}\\hol_files\\{}.dat'.format(path_sa, name), 'r') as f:
        _data = f.readlines()

    _data = pd.DataFrame(_data)
    _holiday = pd.DataFrame()
    _data = _data[0].str.split(' ', expand=True)
    _holiday['hol'] = _data[4]
    _holiday.index = pd.date_range('1901-01-31', '2050-12-31', freq='M')

    return _holiday


def seasonal_adj(data, log=False, fq='M', mode='hol_save_7_14'):
    """
    针对月度或者季度数据

    调用Python的X-13模块做季节性调整
    Args:
        data: 包括宏观经济指标的DataFrame数据
        log: 是否进行对数变换
        fq: fq='M'是月度序列，fq='Q'是季度序列
        mode: mode=是春节调整的模式

    Returns: 季调后的趋势项，也可以输出季调后序列（需改原始代码）

    """
    path = '{}\\x13as.exe'.format(path_sa)
    hol = generate_spring_festival_dates(mode)

    for i in data.columns:
        _sa = None
        if fq == 'M':
            _sa = x13_arima_analysis(data[[i]], log=log, x12path=path, freq=fq, exog=hol)
        elif fq == 'Q':
            _sa = x13_arima_analysis(data[[i]], log=log, x12path=path, freq=fq)
        # _result_seasonal = _sa.seasadj.to_frame()
        _result_trend = _sa.trend.to_frame()
        _name = data[[i]].columns[0]
        # _result_seasonal.rename(columns={'seasadj': _name}, inplace=True)
        _result_trend.rename(columns={'trend': _name}, inplace=True)
        return _result_trend


def calculate_yearly_week_counts(df_weekly):
    """

    Args:
        df_weekly: 周度数据，日期为周日

    Returns: 每一年包含的周数

    """
    df_processed = df_weekly.copy()
    df_processed = df_processed.resample('W-SUN').asfreq()
    _year_weeks = df_processed.groupby(df_processed.index.isocalendar().year).size()
    print(_year_weeks)


def redistribute_53rd_week(df_weekly):
    """
    对于包含53周的年份，将第53周的数据平均分配到前52周，并删除第53周的数据
    Args:
        df_weekly: 周度数据，日期为周日

    Returns: 删除第53周后的数据

    """
    # 复制一份数据，防止直接修改原始DataFrame
    df_processed = df_weekly.copy()

    # 计算每年的周数
    _year_weeks = df_processed.groupby(df_processed.index.isocalendar().year).size()

    # 过滤出有53周的年份
    _extra_weeks_years = _year_weeks[_year_weeks == 53].index

    for _year in _extra_weeks_years:
        # 找到该年第53周的数据
        _extra_week_data = df_processed[df_processed.index.isocalendar().year == _year]
        _extra_week_data = _extra_week_data[_extra_week_data.index.isocalendar().week == 53]

        # 如果_extra_week_data不为空，则进行处理
        if not _extra_week_data.empty:
            # 计算第53周的数据平均值
            _extra_data_average = _extra_week_data.mean()

            # 将第53周的数据平均分配到前52周
            _mask = (df_processed.index.year == _year) & (df_processed.index.isocalendar().week < 53)
            df_processed.loc[_mask, :] += _extra_data_average / 52

            # 删除第53周的数据
            df_processed = df_processed.drop(_extra_week_data.index)

    return df_processed


def weekly_log_diff_52(weekly_data):
    """
    输入：周度数据，确保数据无缺失值或者缺失值可以用0填充

    返回：52周对数差分
    """
    # 填充缺失的日期，结束日为周日
    weekly_data_processed = weekly_data.resample('W-SUN').mean()
    # 缺失值填充为0
    weekly_data_processed.fillna(0, inplace=True)
    # 对于包含53周的年份，将第53周的数据平均分配到前52周，并删除第53周的数据
    weekly_data_processed = redistribute_53rd_week(weekly_data_processed)
    # 季节性调整
    weekly_data_processed = stl_sa(weekly_data_processed)
    # 做52周的对数差分
    weekly_data_processed = np.log(weekly_data_processed).diff(52).dropna()

    return weekly_data_processed
