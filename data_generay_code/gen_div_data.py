import pandas as pd
import numpy as np
import csv
import json
from utils import com_uses as cu
import os

def get_sim_by_num_array(start_num,end_num,station_conns,station_dic,DIV_lable,step):

    station_localtion_dic=get_station_localtion_dic(DIV_lable)
    sim_frames=[]
    for station_conn in station_conns:
        start_station=station_conn[0]
        end_station=station_conn[1]
        start_station_localtion=station_localtion_dic[start_station]
        end_station_localtion=station_localtion_dic[end_station]
        wind_directions=station_dic[start_station]
        wind_directions_A=cu.get_sim_wind_array(wind_directions,start_station_localtion,end_station_localtion)
        wind_directions_A=np.around(wind_directions_A, decimals=4)

        sim_location=cu.get_sim_location(start_station_localtion,end_station_localtion)
        sim_location_A=np.array(sim_location)
        sim_location_A = np.around(sim_location_A, decimals=4)

        sim_location_B=np.tile(sim_location_A, [len(wind_directions),1])

        wind_directions_A=wind_directions_A.reshape(sim_location_B.shape)
        per_sim_frame=np.concatenate((sim_location_B,wind_directions_A), axis=1)
        sim_frames.append(per_sim_frame)
        
    sim_frames=np.array(sim_frames).transpose(1,0,2)

    conn_sim=[]
    for i in range(start_num,end_num,step):
        x_frames=sim_frames[i:i+24]
        conn_sim.append(x_frames.tolist())
    return conn_sim

def select_PM25_dataset_by_num(start_num,end_num,F_PM25_A,step):
    batchsize=48
    station_F_PM25_list=[]
    for i in range(start_num,end_num,step):
        X_F_PM25=F_PM25_A[i:i+batchsize]
        x_frames=X_F_PM25.tolist()
        x=np.array(x_frames).astype(float)
        x=np.around(x, decimals=4)
        station_F_PM25_list.append(x.tolist())
    return station_F_PM25_list

def select_train_dataset_by_num(start_num,end_num,sum_F_AIR_PRESSURE_B,sum_F_HUMIDITY_B,sum_F_TEMPERATURE_B,
                                sum_F_WIND_DIRECTION_B,sum_F_WIND_SPEED_B,F_MONITOR_TIME_B):
    batchsize=48
    weather_list=[]
    weather_for_list=[]
    
    for i in range(start_num,end_num):
        x_frames = []
        y_frames = []
        time1_frames = []
        time2_frames = []
        
        X_F_AIR_PRESSURE=sum_F_AIR_PRESSURE_B[i:i+24]
        Y_F_AIR_PRESSURE=sum_F_AIR_PRESSURE_B[i+24:i+batchsize]
            
        X_F_HUMIDITY=sum_F_HUMIDITY_B[i:i+24]
        Y_F_HUMIDITY=sum_F_HUMIDITY_B[i+24:i+batchsize]
            
        X_F_TEMPERATURE=sum_F_TEMPERATURE_B[i:i+24]
        Y_F_TEMPERATURE=sum_F_TEMPERATURE_B[i+24:i+batchsize]
            
        X_F_WIND_DIRECTION=sum_F_WIND_DIRECTION_B[i:i+24]
        Y_F_WIND_DIRECTION=sum_F_WIND_DIRECTION_B[i+24:i+batchsize]
            
        X_F_WIND_SPEED=sum_F_WIND_SPEED_B[i:i+24]
        Y_F_WIND_SPEED=sum_F_WIND_SPEED_B[i+24:i+batchsize]
        time1_frames.append(F_MONITOR_TIME_B[i:i+24])
        time2_frames.append(F_MONITOR_TIME_B[i+24:i+batchsize])
        x_frames.append(X_F_AIR_PRESSURE)
        x_frames.append(X_F_HUMIDITY)
        x_frames.append(X_F_TEMPERATURE)
        x_frames.append(X_F_WIND_DIRECTION)
        x_frames.append(X_F_WIND_SPEED)
        
        y_frames.append(Y_F_AIR_PRESSURE)
        y_frames.append(Y_F_HUMIDITY)
        y_frames.append(Y_F_TEMPERATURE)
        y_frames.append(Y_F_WIND_DIRECTION)
        y_frames.append(Y_F_WIND_SPEED)
        
        x1=np.array(x_frames).T
        y1=np.array(y_frames).T
        x2=np.array(time1_frames)
        y2=np.array(time2_frames)
        x=np.concatenate((x1,x2),axis=2)
        y=np.concatenate((y1,y2),axis=2)
        weather_list.append(x.tolist())
        weather_for_list.append(y.tolist())

    return weather_list,weather_for_list


def select_test_dataset_by_num(start_num, end_num, sum_F_AIR_PRESSURE_B, sum_F_HUMIDITY_B, sum_F_TEMPERATURE_B,
                                sum_F_WIND_DIRECTION_B, sum_F_WIND_SPEED_B,F_MONITOR_TIME_B):
    batchsize = 48
    weather_list = []
    weather_for_list = []

    for i in range(start_num, end_num,24):
        x_frames = []
        y_frames = []
        time1_frames = []
        time2_frames = []

        X_F_AIR_PRESSURE = sum_F_AIR_PRESSURE_B[i:i + 24]
        Y_F_AIR_PRESSURE = sum_F_AIR_PRESSURE_B[i + 24:i + batchsize]
        X_F_HUMIDITY = sum_F_HUMIDITY_B[i:i + 24]
        Y_F_HUMIDITY = sum_F_HUMIDITY_B[i + 24:i + batchsize]
        X_F_TEMPERATURE = sum_F_TEMPERATURE_B[i:i + 24]
        Y_F_TEMPERATURE = sum_F_TEMPERATURE_B[i + 24:i + batchsize]
        X_F_WIND_DIRECTION = sum_F_WIND_DIRECTION_B[i:i + 24]
        Y_F_WIND_DIRECTION = sum_F_WIND_DIRECTION_B[i + 24:i + batchsize]
        X_F_WIND_SPEED = sum_F_WIND_SPEED_B[i:i + 24]
        Y_F_WIND_SPEED = sum_F_WIND_SPEED_B[i + 24:i + batchsize]
        time1_frames.append(F_MONITOR_TIME_B[i:i + 24])
        time2_frames.append(F_MONITOR_TIME_B[i + 24:i + batchsize])

        x_frames.append(X_F_AIR_PRESSURE)
        x_frames.append(X_F_HUMIDITY)
        x_frames.append(X_F_TEMPERATURE)
        x_frames.append(X_F_WIND_DIRECTION)
        x_frames.append(X_F_WIND_SPEED)

        y_frames.append(Y_F_AIR_PRESSURE)
        y_frames.append(Y_F_HUMIDITY)
        y_frames.append(Y_F_TEMPERATURE)
        y_frames.append(Y_F_WIND_DIRECTION)
        y_frames.append(Y_F_WIND_SPEED)

        x1 = np.array(x_frames).T
        y1 = np.array(y_frames).T
        x2 = np.array(time1_frames)
        y2 = np.array(time2_frames)
        x = np.concatenate((x1,x2),axis=2)
        y = np.concatenate((y1,y2),axis=2)
        weather_list.append(x.tolist())
        weather_for_list.append(y.tolist())

    return weather_list, weather_for_list
 
def get_station_localtion_dic(DIV_lable):
    data_path = 'D:/Projects/myProject/data_generation/air_forcast_data/stations13.csv'
    df=pd.read_csv(data_path)
    all_data = df.loc[:, :].values
    station_localtion_dic={}
    for data in all_data:
        F_STATION_NAME=str(data[0])
        ZDJD = data[1]
        ZDWD = data[2]
        STATION_CLUSTER=str(data[3])
        if STATION_CLUSTER==DIV_lable:
            station_localtion_dic[F_STATION_NAME]=[ZDJD,ZDWD]
    return station_localtion_dic


def get_poi(div_station_names):
    
    station_poi_path="D:/Projects/myProject/data_generation/air_forcast_data/air_poi.json"
    poi_dict=[]
    poi=[]
    for poi_json in open(station_poi_path, encoding='utf-8'):
        poi_dict=eval(poi_json)
    for div_station_name in div_station_names:
        if div_station_name in poi_dict.keys():
            poi.append(eval(poi_dict[div_station_name]))
    return poi
    
def gen_conn(div_num,div_station_names):
    div_station_names=list(div_station_names)
    conn=[]
    station_conns=[]
    for i in range(div_num):
        for j in range(div_num):
            cell=[]
            station_cell=[]
            if i!=j:
                cell.append(i)
                cell.append(j)
                conn.append(cell)
                station_cell.append(div_station_names[i])
                station_cell.append(div_station_names[j])
                station_conns.append(station_cell)
    return conn,station_conns

def split_div_station_feature_train(DIV_lable):

    out_base_dir="D:/Projects/myProject/data_generation/v2_air_data/"
    train_path=out_base_dir+DIV_lable+"_train.json"
    val_path=out_base_dir+DIV_lable+"_val.json"
    
    train_fout = open(train_path, 'w')
    val_fout = open(val_path, 'w')

    div_station_data_path="D:/Projects/myProject/data_generation/v2_air_data/stations.json"
    div_station_dic={}
    for data in open(div_station_data_path, encoding='utf-8'):
        div_station_dic=eval(data)
    div_station_names=div_station_dic[DIV_lable]
    DIV_num=len(div_station_names)
    print("len_div_station_names: ",len(div_station_names))
    
    DIV_train_dict={}
    DIV_val_dict={}
    
    DIV_train_dict["name"]=DIV_lable
    DIV_val_dict["name"]=DIV_lable

    train_num=15000
    val_num=1729
    in_base_dir = "D:/Projects/myProject/data_generation/air_forcast_data/data/jn_data_2021/"
    station_feature_path=in_base_dir+"all_"+DIV_lable+"_station_final_train.csv"
    select_station_df = pd.read_csv(station_feature_path)
    
    F_AIR_PRESSURE_list=[]
    F_HUMIDITY_list=[]
    F_TEMPERATURE_list=[]
    F_WIND_DIRECTION_list=[]
    F_WIND_SPEED_list=[]
    F_PM25_list=[]
    F_MONITOR_TIME_array = []
    station_dic={}
    for station_name in div_station_names:
        select_station= select_station_df[select_station_df['F_STATION_NAME']==station_name]
        F_AIR_PRESSURE= select_station[['F_AIR_PRESSURE']]
        F_HUMIDITY=select_station[['F_HUMIDITY']]
        F_TEMPERATURE=select_station[['F_TEMPERATURE']]
        F_WIND_DIRECTION=select_station[['F_WIND_DIRECTION']]
        F_WIND_SPEED=select_station[['F_WIND_SPEED']]
        F_PM25=select_station[['F_PM25']]
        F_MONITOR_TIME = cu.batch_time_feature(np.array(select_station['F_MONITOR_TIME']))
        F_AIR_PRESSURE_list.append(np.array(F_AIR_PRESSURE).tolist())
        F_HUMIDITY_list.append(np.array(F_HUMIDITY).tolist())
        F_TEMPERATURE_list.append(np.array(F_TEMPERATURE).tolist())
        F_WIND_DIRECTION_list.append(np.array(F_WIND_DIRECTION).tolist())
        F_WIND_SPEED_list.append(np.array(F_WIND_SPEED).tolist())
        F_PM25_list.append(np.array(F_PM25).tolist())
        
        F_PM25_A=np.array(F_PM25)
        start_num=0
        end_num=train_num
        train_station_F_PM25=select_PM25_dataset_by_num(start_num,end_num,F_PM25_A,1)
        val_station_F_PM25=select_PM25_dataset_by_num(train_num,train_num+val_num,F_PM25_A,1)
        
        train_station_F_PM25=np.array(train_station_F_PM25).reshape(train_num,48)
        val_station_F_PM25=np.array(val_station_F_PM25).reshape(val_num,48)
        DIV_train_dict[station_name]=train_station_F_PM25.tolist()
        DIV_val_dict[station_name]=val_station_F_PM25.tolist()
        F_MONITOR_TIME_array = np.array(F_MONITOR_TIME)
        
        station_dic[station_name]=np.array(F_WIND_DIRECTION).tolist()
        
    F_AIR_PRESSURE_A=np.array(F_AIR_PRESSURE_list)
    F_HUMIDITY_A=np.array(F_HUMIDITY_list)
    F_TEMPERATURE_A=np.array(F_TEMPERATURE_list)
    F_WIND_DIRECTION_A=np.array(F_WIND_DIRECTION_list)
    F_WIND_SPEED_A=np.array(F_WIND_SPEED_list)
    F_PM25_A=np.array(F_PM25_list)

    station_num=F_AIR_PRESSURE_A.shape[0]

    sum_F_AIR_PRESSURE_B=np.sum(F_AIR_PRESSURE_A.astype(float), axis=0)/station_num
    sum_F_AIR_PRESSURE_B=np.around(sum_F_AIR_PRESSURE_B, decimals=4)
    sum_F_HUMIDITY_B=np.sum(F_HUMIDITY_A.astype(float), axis=0)/station_num
    sum_F_HUMIDITY_B=np.around(sum_F_HUMIDITY_B, decimals=4)
    sum_F_TEMPERATURE_B=np.sum(F_TEMPERATURE_A.astype(float), axis=0)/station_num
    sum_F_TEMPERATURE_B=np.around(sum_F_TEMPERATURE_B, decimals=4)
    sum_F_WIND_DIRECTION_B=np.sum(F_WIND_DIRECTION_A.astype(float), axis=0)/station_num
    sum_F_WIND_DIRECTION_B=np.around(sum_F_WIND_DIRECTION_B, decimals=4)
    sum_F_WIND_SPEED_B=np.sum(F_WIND_SPEED_A.astype(float), axis=0)/station_num
    sum_F_WIND_SPEED_B=np.around(sum_F_WIND_SPEED_B, decimals=4)
    sum_F_PM25_B=np.sum(F_PM25_A.astype(float), axis=0)/station_num
    sum_F_PM25_B=np.around(sum_F_PM25_B, decimals=4)
    F_MONITOR_TIME_B = F_MONITOR_TIME_array

    train_weather_list,train_weather_for_list=select_train_dataset_by_num(0,train_num,sum_F_AIR_PRESSURE_B,sum_F_HUMIDITY_B,\
    sum_F_TEMPERATURE_B,sum_F_WIND_DIRECTION_B,sum_F_WIND_SPEED_B,F_MONITOR_TIME_B)
    
    print("386 train_weather_list: ",np.array(train_weather_list).shape)
    
    train_weather_list=np.array(train_weather_list).reshape(train_num,24, 10)
    train_weather_for_list=np.array(train_weather_for_list).reshape(train_num,24,10)

    print("train_weather_list: ",np.array(train_weather_list).shape)
    print("train_weather_for_list: ",np.array(train_weather_for_list).shape)
    
    val_weather_list,val_weather_for_list=select_train_dataset_by_num(train_num,train_num+val_num,sum_F_AIR_PRESSURE_B,\
        sum_F_HUMIDITY_B,sum_F_TEMPERATURE_B,sum_F_WIND_DIRECTION_B,sum_F_WIND_SPEED_B,F_MONITOR_TIME_B)
    
    val_weather_list=np.array(val_weather_list).reshape(val_num,24, 10)
    val_weather_for_list=np.array(val_weather_for_list).reshape(val_num,24,10)

    print("val_weather_list: ",np.array(val_weather_list).shape)
    print("val_weather_for_list: ",np.array(val_weather_for_list).shape)

    DIV_train_dict["weather"]=train_weather_list.tolist()
    DIV_train_dict["weather_for"]=train_weather_for_list.tolist()
    
    DIV_val_dict["weather"]=val_weather_list.tolist()
    DIV_val_dict["weather_for"]=val_weather_for_list.tolist()

    poi=get_poi(div_station_names)
        
    DIV_train_dict["poi"]=poi
    DIV_val_dict["poi"]=poi
        
    conn,station_conns=gen_conn(DIV_num,div_station_names)
        
    DIV_train_dict["conn"]=conn
    DIV_val_dict["conn"]=conn
    print("station_conns_len: ",len(station_conns))
    step=1
    train_conn_sim=get_sim_by_num_array(0,train_num,station_conns,station_dic,DIV_lable,step)
    print("train_conn_sim_shape",np.array(train_conn_sim).shape)
    
    val_conn_sim=get_sim_by_num_array(train_num,train_num+val_num,station_conns,station_dic,DIV_lable,step)
    print("val_conn_sim_shape",np.array(val_conn_sim).shape)

    DIV_train_dict["sim"]=train_conn_sim
    DIV_val_dict["sim"]=val_conn_sim

    json.dump(DIV_train_dict,train_fout)

    json.dump(DIV_val_dict,val_fout)
        
    train_fout.close()
    val_fout.close()

    print("OK1")
        
def split_div_station_feature_test(DIV_lable):
    out_base_dir="D:/Projects/myProject/data_generation/v2_air_data/"
    test_path=out_base_dir+DIV_lable+"_test.json"
    with open(test_path, 'w') as test_fout:
        DIV_test_dict={}

        div_station_data_path = "D:/Projects/myProject/data_generation/v2_air_data/stations.json"
        div_station_dic={}
        for data in open(div_station_data_path, encoding='utf-8'):
            div_station_dic=eval(data)
        div_station_names=div_station_dic[DIV_lable]
        DIV_num=len(div_station_names)
        DIV_test_dict["name"]=DIV_lable
        test_num=125
        in_base_dir = "D:/Projects/myProject/data_generation/air_forcast_data/data/jn_data_2021/"
        station_feature_path=in_base_dir+"all_"+DIV_lable+"_station_final_test.csv"
        select_station_df = pd.read_csv(station_feature_path)

        F_AIR_PRESSURE_list=[]
        F_HUMIDITY_list=[]
        F_TEMPERATURE_list=[]
        F_WIND_DIRECTION_list=[]
        F_WIND_SPEED_list=[]
        F_MONITOR_TIME_array=[]
        F_PM25_list=[]
        station_dic={}
        for station_name in div_station_names:
            select_station= select_station_df[select_station_df['F_STATION_NAME']==station_name]
            F_AIR_PRESSURE= select_station[['F_AIR_PRESSURE']]
            F_HUMIDITY=select_station[['F_HUMIDITY']]
            F_TEMPERATURE=select_station[['F_TEMPERATURE']]
            F_WIND_DIRECTION=select_station[['F_WIND_DIRECTION']]
            F_WIND_SPEED=select_station[['F_WIND_SPEED']]
            F_PM25=select_station[['F_PM25']]
            F_MONITOR_TIME=cu.batch_time_feature(np.array(select_station['F_MONITOR_TIME']))
            F_AIR_PRESSURE_list.append(np.array(F_AIR_PRESSURE).tolist())
            F_HUMIDITY_list.append(np.array(F_HUMIDITY).tolist())
            F_TEMPERATURE_list.append(np.array(F_TEMPERATURE).tolist())
            F_WIND_DIRECTION_list.append(np.array(F_WIND_DIRECTION).tolist())
            F_WIND_SPEED_list.append(np.array(F_WIND_SPEED).tolist())
            F_PM25_list.append(np.array(F_PM25).tolist())
            station_dic[station_name]=np.array(F_WIND_DIRECTION).tolist()
            F_MONITOR_TIME_array=np.array(F_MONITOR_TIME)

            F_PM25_A=np.array(F_PM25)
            start_num = 0
            end_num = F_PM25_A.shape[0]-24
            test_station_F_PM25=select_PM25_dataset_by_num(start_num,end_num,F_PM25_A,24)#逐24小时取测试样本
            test_station_F_PM25=np.array(test_station_F_PM25).reshape(test_num,48)
            DIV_test_dict[station_name]=test_station_F_PM25.tolist()
            
        F_AIR_PRESSURE_A=np.array(F_AIR_PRESSURE_list)
        F_HUMIDITY_A=np.array(F_HUMIDITY_list)
        F_TEMPERATURE_A=np.array(F_TEMPERATURE_list)
        F_WIND_DIRECTION_A=np.array(F_WIND_DIRECTION_list)
        F_WIND_SPEED_A=np.array(F_WIND_SPEED_list)
        F_PM25_A=np.array(F_PM25_list)
        
        station_num=F_AIR_PRESSURE_A.shape[0]

        sum_F_AIR_PRESSURE_B=np.sum(F_AIR_PRESSURE_A.astype(float), axis=0)/station_num
        sum_F_AIR_PRESSURE_B=np.around(sum_F_AIR_PRESSURE_B, decimals=4)
        sum_F_HUMIDITY_B=np.sum(F_HUMIDITY_A.astype(float), axis=0)/station_num
        sum_F_HUMIDITY_B=np.around(sum_F_HUMIDITY_B, decimals=4)
        sum_F_TEMPERATURE_B=np.sum(F_TEMPERATURE_A.astype(float), axis=0)/station_num
        sum_F_TEMPERATURE_B=np.around(sum_F_TEMPERATURE_B, decimals=4)
        sum_F_WIND_DIRECTION_B=np.sum(F_WIND_DIRECTION_A.astype(float), axis=0)/station_num
        sum_F_WIND_DIRECTION_B=np.around(sum_F_WIND_DIRECTION_B, decimals=4)
        sum_F_WIND_SPEED_B=np.sum(F_WIND_SPEED_A.astype(float), axis=0)/station_num
        sum_F_WIND_SPEED_B=np.around(sum_F_WIND_SPEED_B, decimals=4)
        sum_F_PM25_B=np.sum(F_PM25_A.astype(float), axis=0)/station_num
        sum_F_PM25_B=np.around(sum_F_PM25_B, decimals=4)
        F_MONITOR_TIME_B=F_MONITOR_TIME_array

        start_num=0
        end_num=sum_F_PM25_B.shape[0]-24
        test_weather_list,test_weather_for_list=select_test_dataset_by_num(start_num,end_num,sum_F_AIR_PRESSURE_B,\
        sum_F_HUMIDITY_B,sum_F_TEMPERATURE_B,sum_F_WIND_DIRECTION_B,sum_F_WIND_SPEED_B,F_MONITOR_TIME_B)#逐24小时取气象样本
        test_weather_list=np.array(test_weather_list).reshape(125,24, 10)
        test_weather_for_list=np.array(test_weather_for_list).reshape(125,24,10)
    
        print("test_weather_list_shape: ",np.array(test_weather_list).shape)
        print("test_weather_for_list: ",np.array(test_weather_for_list).shape)
    
        DIV_test_dict["weather"]=test_weather_list.tolist()
        DIV_test_dict["weather_for"]=test_weather_for_list.tolist()
    
        poi=get_poi(div_station_names)
        
        DIV_test_dict["poi"]=poi
        
        conn,station_conns=gen_conn(DIV_num,div_station_names)
        
        DIV_test_dict["conn"]=conn
        print("station_conns_len: ",len(station_conns))
        step=24
        test_conn_sim=get_sim_by_num_array(start_num,end_num,station_conns,station_dic,DIV_lable,step)
        print("test_conn_sim_shape",np.array(test_conn_sim).shape)
        DIV_test_dict["sim"]=test_conn_sim
        
        json.dump(DIV_test_dict,test_fout)
        print("548 finished")

    test_fout.close()
 
    print("OK")            

    
def split_div_station_feature():
    split_div_station_feature_train()
    print("train_OK")
    split_div_station_feature_test()
    print("finished")
    
    
def select_div_station_feature(DIV_lable):
    base_dir = "D:/Projects/myProject/data_generation/air_forcast_data/data/jn_data_2021/"
    train_data_path = "D:/Projects/myProject/data_generation/air_forcast_data/data/jn_data_2021/lhb_train_data_20190202_20201231.csv"
    test_data_path = "D:/Projects/myProject/data_generation/air_forcast_data/data/jn_data_2021/test_data_20210101_20210506.csv"
    
    station_data_path="D:/Projects/myProject/data_generation/final_air_data/stations.json"
    DIV=[]
    for data in open(station_data_path,encoding='utf-8'):
        data_json=eval(data)
        DIV = data_json[DIV_lable]
    f_train_path=base_dir+"all_"+DIV_lable + "_station_final_train.csv"
    f_test_path=base_dir+"all_"+DIV_lable + "_station_final_test.csv"

    if os.path.exists(f_train_path):
        print('station_final_train.csv is exist!')
    else:
        f_train = open(f_train_path, 'w', encoding='utf-8', newline='')
        csv_writer_train = csv.writer(f_train)
        csv_writer_train.writerow(
            ["F_MONITOR_TIME", "F_STATION_NAME", "F_AIR_PRESSURE", "F_HUMIDITY", "F_TEMPERATURE", "F_WIND_DIRECTION",
             "F_WIND_SPEED", "F_PM25", "ZDJD", "ZDWD"])
        train_all_data = pd.read_csv(train_data_path, encoding='utf-8').loc[:, :].values
        for train_data in train_all_data:
            new_line = []
            F_STATION_NAME = str(train_data[10])
            if F_STATION_NAME in DIV:
                F_MONITOR_TIME = str(train_data[4])
                F_AIR_PRESSURE = str(train_data[0])
                F_HUMIDITY = str(train_data[2])
                F_TEMPERATURE = str(train_data[11])
                F_WIND_DIRECTION = str(train_data[12])
                F_WIND_SPEED = str(train_data[13])
                F_PM25 = str(train_data[8])
                ZDJD = str(train_data[14])
                ZDWD = str(train_data[15])
                new_line.append(F_MONITOR_TIME)
                new_line.append(F_STATION_NAME)
                new_line.append(F_AIR_PRESSURE)
                new_line.append(F_HUMIDITY)
                new_line.append(F_TEMPERATURE)
                new_line.append(F_WIND_DIRECTION)
                new_line.append(F_WIND_SPEED)
                new_line.append(F_PM25)
                new_line.append(ZDJD)
                new_line.append(ZDWD)
                csv_writer_train.writerow(new_line)

        f_train.close()

    if os.path.exists(f_test_path):
        print('the station_final_test.csv is exist!')
    else:
        f_test = open(f_test_path, 'w', encoding='utf-8', newline='')
        csv_writer_test = csv.writer(f_test)
        csv_writer_test.writerow(
            ["F_MONITOR_TIME", "F_STATION_NAME", "F_AIR_PRESSURE", "F_HUMIDITY", "F_TEMPERATURE", "F_WIND_DIRECTION",
             "F_WIND_SPEED", "F_PM25", "ZDJD", "ZDWD"])
        test_all_data = pd.read_csv(test_data_path, encoding='utf-8').loc[:, :].values
        for test_data in test_all_data:
            test_new_line = []
            F_STATION_NAME = str(test_data[10])
            if F_STATION_NAME in DIV:
                F_MONITOR_TIME = str(test_data[4])
                # F_MONITOR_TIME = datetime.datetime.strptime(F_MONITOR_TIME,'%Y-%m-%d %H:%M:%S')
                F_AIR_PRESSURE = str(test_data[0])
                F_HUMIDITY = str(test_data[2])
                F_TEMPERATURE = str(test_data[11])
                F_WIND_DIRECTION = str(test_data[12])
                F_WIND_SPEED = str(test_data[13])
                F_PM25 = str(test_data[8])
                ZDJD = str(test_data[14])
                ZDWD = str(test_data[15])
                test_new_line.append(F_MONITOR_TIME)
                test_new_line.append(F_STATION_NAME)
                test_new_line.append(F_AIR_PRESSURE)
                test_new_line.append(F_HUMIDITY)
                test_new_line.append(F_TEMPERATURE)
                test_new_line.append(F_WIND_DIRECTION)
                test_new_line.append(F_WIND_SPEED)
                test_new_line.append(F_PM25)
                test_new_line.append(ZDJD)
                test_new_line.append(ZDWD)
                csv_writer_test.writerow(test_new_line)

        f_test.close()

    print("OK")

def main():
    for i in range(13):
        DIV_lable="DIV"+str(i+1)
        print("DIV_lable: ",DIV_lable)
        select_div_station_feature(DIV_lable)
        print("select_div_station_feature_finished")
        split_div_station_feature_test(DIV_lable)
        print("test_gen_finished")
        split_div_station_feature_train(DIV_lable)
        print("train_gen_finished")
        
if __name__ == '__main__':
    
    #select_div_station_feature()#取当前站点所需要的特征列
    #split_div_station_feature_test()
    #split_div_station_feature()
    #split_div_station_feature_test()
    #split_div_station_feature_train()
    #DIV_lable="DIV7"
    #select_div_station_feature(DIV_lable)
    #split_div_station_feature_test(DIV_lable)
    #split_div_station_feature_train(DIV_lable)
    main()
    
    

