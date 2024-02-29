# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:20:06 2024

@author: zhong
"""

import numpy as np
import pandas as pd
import math
import scipy.optimize as sco
import warnings
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm



warnings.resetwarnings()
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', PendingDeprecationWarning)

def RiskParity(dataframe_rendimenti):
    bound = 1.0
    rendimenti = np.array(dataframe_rendimenti)
    nn = np.shape(rendimenti)[0]
    m = np.shape(rendimenti)[1]
    sigma = np.zeros([m, m])
    for i in range(0, m, 1):
        for j in range(0, m, 1):
            sigma[i][j] = 252 * np.cov(rendimenti[:, i], rendimenti[:, j])[0][1]

    def riskparity_(x):
        n = len(sigma)
        w = np.mat(x).T
        port_var = np.sqrt(w.T * np.mat(sigma) * w)
        port_vec = np.mat(np.repeat(port_var / n, n)).T
        diag = np.mat(np.diag(x) / port_var)
        partial = np.mat(sigma) * w
        return np.square(port_vec - diag * partial).sum()

    cons = ({'type': 'eq', 'fun': lambda w: sum(w) - 1})
    bnds = ((0, bound),) * sigma.shape[0]
    w_ini = np.repeat(1, np.shape(sigma)[0])
    w_ini = w_ini / sum(w_ini)
    res = sco.minimize(riskparity_, w_ini, bounds=bnds, constraints=cons, options={'disp': False, 'ftol': 10 ** -10})
    return res['x']

# 从excel的sheet2中读取数据
df = pd.read_excel('C:/Users/zhong/Desktop/中国银河/风险平价等策略/风险平价模型简易练习.xlsx', sheet_name='Sheet2').iloc[1:188,:]

df.head()
# 将第一列转换为日期格式
df['日期'] = pd.to_datetime(df['日期'])

# 设置日期列为索引
df.set_index('日期', inplace=True)

# 将数据按月份和年份分组，然后求和
df = df.groupby([df.index.year, df.index.month]).sum()

# 将索引转换回日期格式
df.index = pd.to_datetime(df.index.map(lambda x: f'{x[0]}-{x[1]}'))

# 如果你想要取每个月中最晚的一天作为索引，你可以这样做：
df.index = df.index + pd.offsets.MonthEnd(1)


# 获取奇数列和偶数列
odd_columns = df.iloc[:, ::2]
even_columns = df.iloc[:, 1::2]

# 计算（奇数列-偶数列）/奇数列
new_df = (odd_columns.values - even_columns.values) / odd_columns.values

# 将结果转换为数据框
dfratio= pd.DataFrame(new_df, columns=odd_columns.columns,index=df.index)



drawi=6
weightnp=np.zeros((df.shape[0]-12,drawi))

for i in range(df.shape[0]-12):
    wdata=dfratio.iloc[i:(i+11),0:drawi]
    weights = RiskParity(wdata)
    weightnp[i,:]=weights
    
weightratio=(weightnp*dfratio.iloc[12:169,0:drawi]).sum(axis=1)
df_draw=dfratio.iloc[12:169,0:drawi]
df_draw['weightratio']=weightratio


print(df_draw.sum(axis=0))

# 设置字体路径
font_path = 'C:/Windows/Fonts/simsun.ttc'

# 创建字体属性对象
font_prop = fm.FontProperties(fname=font_path)


# 画出每一列数据随时间变化的图像
plt.figure(figsize=(10, 6))
for column in df_draw.columns:
    plt.plot(df_draw.index, df_draw[column], label=column)
plt.xlabel('时间', fontproperties=font_prop)
plt.ylabel('值', fontproperties=font_prop)
plt.title('随时间变化的列数据', fontproperties=font_prop)
plt.legend(prop=font_prop)
plt.show()
