import numpy as np
import pandas as pd
from os.path import join as pjoin
import geopandas as gp
from shapely.geometry import Point
import pylab as plt
import os
from utils import com_uses as cu
str='2021/8/21 10:10'
time=cu.extract_time_feature(str)
print(time)

train_file = pjoin('./data/jn_data_2021', 'lhb_train_data_20190202_20201231.csv')
usecols = ['F_AIR_PRESSURE','F_HUMIDITY','F_TEMPERATURE','F_WIND_DIRECTION','F_WIND_SPEED']
df_data = pd.read_csv(train_file, usecols=usecols, error_bad_lines=False, low_memory=False,
                       encoding='utf-8', keep_default_na=False)
usecols2 = ['F_STATION_NAME','F_AIR_PRESSURE','F_TEMPERATURE','F_WIND_SPEED','F_MONITOR_TIME']
df_data2 = pd.read_csv(train_file, usecols=usecols2, error_bad_lines=False, low_memory=False,
                        encoding='utf-8',keep_default_na=False)
sort_df_data2=df_data2.sort_values(by='F_AIR_PRESSURE')
temp=sort_df_data2.values
np.set_printoptions(suppress=True)
w_min = df_data.min(axis='rows').values
print('the weather min:',w_min)
w_max = df_data.max(axis='rows').values
print('the weather max:',w_max)
w_mean = (df_data.to_numpy()).mean(axis=0)
print('the weather mean: ', w_mean)
w_std = df_data.std(axis='rows').values
print('the weather max2: ', w_std)

print('the test finishing')







#画图代码
'''
# 计算出各站点的经纬度
# 经度 差0.1度  10km
# 纬度 差0.1度
# 
'''
# china_map=gp.GeoDataFrame.from_file("./china_city_shp/dijishi_2004.shp",
#                                     encoding='gb18030')
#
# #map_data = pd.read_csv("./data/jn_data_2021/t_maw_airmonitoring_info.csv", encoding='gb18030')
# test_file = pjoin('./data/jn_data_2021', 'test_data_20210101_20210506.csv')
# usecols = ['F_STATION_NAME','ZDJD','ZDWD']
# map_data=pd.read_csv(test_file,usecols=usecols,
#                      error_bad_lines=False, low_memory=False,  keep_default_na=False)
#
# lng = pd.to_numeric(map_data['ZDJD'],errors='coerce')
# lng_y = list(set(lng.tolist()))#set() 函数创建一个无序不重复元素集
#
# lat = pd.to_numeric(map_data['ZDWD'],errors='coerce')
# lat_x = list(set(lat.tolist()))
#
# text = list(set(map_data['F_STATION_NAME'].tolist()))
#
# print("len(text)",len(text))
# pts = gp.GeoSeries([Point(x, y) for x, y in zip(lat, lng)])
#
# fig, ax = plt.subplots(figsize=(12,18))
# for i in range(len(lat_x)):
#     #plt.text(lng_x[i],lat_y[i],text[i],fontsize=5,bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k',lw=1 ,alpha=0.5))
#     plt.text(lat_x[i],lng_y[i],text[i],fontsize=6, verticalalignment='center', horizontalalignment='right')
# #plt.text(lng_x,lat_y,text)
# ax.set_aspect('equal')
# #pts.plot(figsize=(12,18))
# pts.plot(ax=ax, marker= '*', color='blue', markersize=40)
# plt.rcParams['font.sans-serif']=['SimHei']#显示中文标签
# plt.rcParams['axes.unicode_minus']=False
# plt.show()


# train_file = pjoin('./data/jn_data_2021/', 'lhb_train_data_20190202_20201231.csv')
# info_file = pjoin('./data/jn_data_2021/', 't_maw_airmonitoring_info.csv')
# test_file = pjoin('./data/jn_data_2021', 'test_data_20210101_20210506.csv')
# usecols=['F_STATION_NAME','F_DIV_NAME']
# usecols2=['F_STATION_NAME','ZDJD','ZDWD']
# df_data = pd.read_csv(info_file, usecols=usecols, error_bad_lines=False, low_memory=False,
#                       encoding='gb18030', keep_default_na=False)
# df_data = pd.read_csv(info_file, usecols=usecols, error_bad_lines=False, low_memory=False,
#                       encoding='gb18030', keep_default_na=False)
# group_data=df_data.groupby('F_DIV_NAME')
# groupvale=group_data.groups
# DIV_name=list()#区名称(没有“其它”）
# DIV_station=list()#区下的站点名称，顺序与DIV_name对应一致（没有机场站）
# for name, groups in group_data:
#     print(name)
#     DIV_name.append(name)
#     print(groups.loc[:, 'F_STATION_NAME'])
#     DIV_station.append(groups.loc[:,'F_STATION_NAME'].values)
#
# df_data = pd.read_csv(train_file, error_bad_lines=False, low_memory=False,
#                       encoding='gb18030', keep_default_na=False)
#
# ###--按区生成每一个区的训练数据--------
# for i in range(len(DIV_name)):
#     div_data = pd.DataFrame()
#     for k in range(len(DIV_station[i])):
#         temp = df_data.loc[df_data['F_STATION_NAME'] == DIV_station[i][k]]
#         div_data = div_data.append(temp)

print('datageneration finish!')