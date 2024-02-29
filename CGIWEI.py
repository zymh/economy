import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from QuantDataProcess import get_data, config, winsor_percentile

from QuantMacroModel import weekly_log_diff_52, stl_sa, redistribute_53rd_week

# 设置matplotlib的字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei'是一个常用的支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 消费

# 中国: 电影票房收入: 当周值，周频
weekly_movie_box_office = get_data('S6623764', '2007-01-14', '')
weekly_movie_box_office = weekly_log_diff_52(weekly_movie_box_office)

# 30大中城市商品房成交面积，日频降至周频，取均值
top30_housing_sales = get_data('S2707380', '2010-01-02', '')
top30_housing_sales = weekly_log_diff_52(top30_housing_sales)

# 乘用车零售数据，不标准的周频
car_retail = get_data('S6126413', '2015-03-06', '')
car_retail = stl_sa(car_retail)
car_retail = car_retail.resample('D').asfreq().interpolate('linear')
car_retail = car_retail.resample('W-SUN').mean()
car_retail = redistribute_53rd_week(car_retail)
car_retail = np.log(car_retail).diff(52).dropna()

# 中国: 互联网搜索指数: 失业金领取条件，日频降至周频
# “ui”是Unemployment Insurance（失业保险）的缩写，“search_index”指的是搜索指数
china_ui_search_index = get_data('M6600003', '2011-01-01', '')
# 异常值截取，上截0.98
china_ui_search_index = winsor_percentile(china_ui_search_index, 0, 0.98)
china_ui_search_index = weekly_log_diff_52(china_ui_search_index)

# 中国: 日均产量: 粗钢，旬度，先升频至日频，再降至周频
steel_production_daily = get_data('S5704502', '2009-01-10', '')
steel_production_daily = steel_production_daily.resample('D').asfreq().interpolate('linear')
steel_production_daily = weekly_log_diff_52(steel_production_daily)

# 中国: 开工率: 汽车轮胎(半钢胎)，周频填充缺失值，线性插值，计算同比
semi_steel_tire_utilization_rate = get_data('S6124651', '2013-07-18', '')
semi_steel_tire_utilization_rate = semi_steel_tire_utilization_rate.resample('W-SUN').mean().interpolate('linear')
semi_steel_tire_utilization_rate = weekly_log_diff_52(semi_steel_tire_utilization_rate)

# 中国: 开工率: 石油沥青装置，周频填充缺失值，线性插值，计算同比
asphalt_plant_utilization_rate = get_data('S5449386', '2015-06-05', '')
asphalt_plant_utilization_rate = asphalt_plant_utilization_rate.resample('W-SUN').mean().interpolate('linear')
asphalt_plant_utilization_rate = weekly_log_diff_52(asphalt_plant_utilization_rate)

# 波罗的海干散货指数(BDI)，日频降至周频
bdi = get_data('S0031550', '1988-10-19', '')
bdi = weekly_log_diff_52(bdi)

# 秦皇岛港: 煤炭调度: 港: 煤炭调度口吞吐量，日频降至周频
qhd_port_coal_throughput = get_data('S5104483', '2008-09-01', '')
qhd_port_coal_throughput = weekly_log_diff_52(qhd_port_coal_throughput)

combined_df = pd.concat(
    [weekly_movie_box_office, top30_housing_sales, car_retail, china_ui_search_index, steel_production_daily,
     semi_steel_tire_utilization_rate, asphalt_plant_utilization_rate, bdi, qhd_port_coal_throughput], axis=1,
    sort=True)

combined_df.columns = config(combined_df)

print(combined_df)


def pca_contribution_analysis_remove_nan_and_reverse(df):
    """
    对给定的DataFrame进行删除含缺失值的行，0-1标准化，然后进行PCA分析，降维至1个主成分，并展示每个变量的贡献。
    此版本将主成分的符号反转。

    参数:
    - df: DataFrame, 包含需要进行PCA分析的数据。

    返回:
    - 无, 直接绘制图表展示每个变量对主成分的贡献度。
    """
    # 删除含有缺失值的行
    df_dropped = df.dropna()

    # 0-1标准化
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_dropped)

    # 应用PCA，降维至1个主成分
    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(df_scaled)

    # 反转主成分的符号
    principal_components = -1 * principal_components
    loadings = -1 * pca.components_.T * np.sqrt(pca.explained_variance_)

    # 计算每个原始变量对主成分的贡献度
    contribution_df = pd.DataFrame(loadings, columns=['Contribution'], index=df_dropped.columns)

    # 作图展示每个变量的贡献
    plt.figure(figsize=(10, 6))
    plt.bar(contribution_df.index, contribution_df['Contribution'], color='skyblue')
    plt.xlabel('Variable')
    plt.ylabel('Contribution to Principal Component')
    plt.title('Contribution of Each Variable to the Principal Component')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


pca_contribution_analysis_remove_nan_and_reverse(combined_df)


def preprocess_and_pca(df):
    # 删除含NaN的行
    df_clean = df.dropna()

    # 0-1标准化
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_clean)

    # PCA分析并反转主成分方向
    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(df_scaled) * -1  # 反转方向
    principal_component_series = pd.Series(principal_components.flatten(), index=df_clean.index)
    print(principal_component_series)
    # 计算每个变量对主成分的贡献，并反转方向
    contributions = pca.components_.T * np.sqrt(pca.explained_variance_) * -1
    contributions_df = pd.DataFrame(contributions, index=df_clean.columns, columns=['Contribution'])

    # 绘制时间序列因子变化图
    plt.figure(figsize=(12, 4))
    plt.plot(principal_component_series, label='Principal Component')
    plt.xlabel('Time')
    plt.ylabel('Principal Component Value')
    plt.title('Principal Component Over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()


preprocess_and_pca(combined_df)


def analyze_and_visualize_pca_contributions(df):
    """
    对给定的DataFrame进行预处理（删除含缺失值的行和0-1标准化），执行PCA分析，提取共同因子，
    反转主成分和其贡献的方向，并绘制共同因子的时间序列图以及每个变量对主成分贡献的时间序列堆积柱状图。

    参数:
    - df: DataFrame, 包含需要进行PCA分析的数据。

    返回:
    - 无, 直接绘制图表展示共同因子的变化以及每个变量对主成分的贡献度。
    """

    # 删除含有缺失值的行
    df_clean = df.dropna()

    # 0-1标准化
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_clean)

    # PCA分析并反转主成分方向
    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(df_scaled) * -1  # 反转方向
    principal_component_series = pd.Series(principal_components.flatten(), index=df_clean.index)

    # 计算并反转每个变量对主成分的贡献方向
    contributions = pca.components_[0].flatten() * np.sqrt(pca.explained_variance_)[0] * -1  # 正确地处理维度

    # 绘制共同因子的时间序列图
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)  # 两行一列的图中的第一张
    plt.plot(principal_component_series, color='tab:blue')
    plt.title('Principal Component Over Time')
    plt.xlabel('Time')
    plt.ylabel('Principal Component Value')

    # 绘制每个变量对主成分的贡献的时间序列堆积柱状图
    plt.subplot(2, 1, 2)  # 两行一列的图中的第二张

    # 计算每个变量的贡献，并反转方向
    cumulative_contributions = np.cumsum(df_scaled * contributions, axis=1) * -1  # 反转贡献的方向

    # 绘制堆积柱状图
    time_points = range(len(df_clean))  # 使用简化的时间点索引
    for i, col in enumerate(df_clean.columns):
        if i == 0:
            plt.bar(time_points, cumulative_contributions[:, i], label=col, width=0.4)
        else:
            plt.bar(time_points, cumulative_contributions[:, i] - cumulative_contributions[:, i - 1],
                    bottom=cumulative_contributions[:, i - 1], label=col, width=0.4)

    plt.title('Stacked Contribution of Each Variable to the Principal Component')
    plt.xlabel('Time Index')
    plt.ylabel('Contribution Value')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    plt.show()


analyze_and_visualize_pca_contributions(combined_df)


def analyze_and_visualize_pca_gdp_correlation(df, gdp_series):
    """
    对给定的主成分序列进行季度均值处理，与GDP序列绘制折线图，并计算相关系数。

    参数:
    - df: 包含原始数据的DataFrame，用于PCA分析。
    - gdp_series: 包含GDP时间序列数据的Series。

    返回:
    - 相关系数
    """

    # 删除含有缺失值的行
    df_clean = df.dropna()

    # 0-1标准化
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_clean)

    # PCA分析并反转主成分方向
    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(df_scaled) * -1  # 反转方向
    principal_component_series = pd.Series(principal_components.flatten(), name='PCA', index=df_clean.index)

    # 假设df已经是PCA处理后的主成分序列，这里直接按季度取均值
    principal_component_series_quarterly = principal_component_series.resample('Q').mean()

    # 找到两个序列时间索引的交集
    pca_gdp = pd.concat([principal_component_series_quarterly, gdp_series], axis=1, join='inner', sort=True)

    pca_gdp.iplot(secondary_y='M0039354')

    print(pca_gdp.corr(), 'Correlation coefficient between PCA Principal Component and GDP')

    # # 仅保留交集中的时间点
    # principal_component_series_quarterly = principal_component_series_quarterly.loc[common_index]
    # aligned_gdp = gdp_series.loc[common_index]
    #
    # # 绘制折线图
    # plt.figure(figsize=(14, 7))
    # plt.plot(principal_component_series_quarterly.index, principal_component_series_quarterly,
    #          label='Principal Component', marker='o')
    # plt.plot(aligned_gdp.index, aligned_gdp, label='GDP', marker='x')
    # plt.title('PCA Principal Component and GDP Over Time')
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    #
    # # 计算并返回相关系数
    # correlation = principal_component_series_quarterly.corr(aligned_gdp)
    # print(f'Correlation coefficient between PCA Principal Component and GDP: {correlation}')
    # return correlation


gdp = get_data('M0039354', '1992-03-31', '')

analyze_and_visualize_pca_gdp_correlation(combined_df, gdp)
