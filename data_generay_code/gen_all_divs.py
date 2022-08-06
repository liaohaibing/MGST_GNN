import pandas as pd
import numpy as np
import csv
import json

from utils import com_uses as cu

#---PM2.5数据汇聚------------
def select_PM25_dataset_by_num(start_num,end_num,F_PM25_A,step):
    batchsize=24
    station_F_PM25_list=[]
    for i in range(start_num,end_num,step):
        X_F_PM25=F_PM25_A[i:i+batchsize]
        x_frames=X_F_PM25.tolist()
        x=np.array(x_frames).astype(float)
        x=np.around(x, decimals=4)
        station_F_PM25_list.append(x.tolist())
        
    return station_F_PM25_list

#-----气象数据汇聚-加入时间特征-----------
def select_train_dataset_by_num(start_num,end_num,F_AIR_PRESSURE_B,
                                F_HUMIDITY_B,F_TEMPERATURE_B,F_WIND_DIRECTION_B,F_WIND_SPEED_B,F_MONITOR_TIME_B):
    #batchsize=24
    weather_list=[]
    
    for i in range(start_num,end_num):
        x_frames = []
        time_frames = []
        X_F_AIR_PRESSURE = F_AIR_PRESSURE_B[i:i+24]
        X_F_HUMIDITY= F_HUMIDITY_B[i:i+24]
        X_F_TEMPERATURE=F_TEMPERATURE_B[i:i+24]
        X_F_WIND_DIRECTION=F_WIND_DIRECTION_B[i:i+24]
        X_F_WIND_SPEED=F_WIND_SPEED_B[i:i+24]
        X_F_MOITOR_TIME=F_MONITOR_TIME_B[i:i+24]

        x_frames.append(X_F_AIR_PRESSURE)
        x_frames.append(X_F_HUMIDITY)
        x_frames.append(X_F_TEMPERATURE)
        x_frames.append(X_F_WIND_DIRECTION)
        x_frames.append(X_F_WIND_SPEED)
        time_frames.append(X_F_MOITOR_TIME)
        
        x1=np.array(x_frames).T
        x2=np.array(time_frames)
        x=np.concatenate((x1,x2),axis=2)
        weather_list.append(x.tolist())

    return weather_list


# -----气象数据汇聚------------
def select_test_dataset_by_num(start_num, end_num, F_AIR_PRESSURE_B,
                                F_HUMIDITY_B, F_TEMPERATURE_B, F_WIND_DIRECTION_B, F_WIND_SPEED_B,F_MONITOR_TIME_B):
    # batchsize=24
    weather_list = []

    for i in range(start_num, end_num,24):
        x_frames = []
        time_frames=[]
        X_F_AIR_PRESSURE = F_AIR_PRESSURE_B[i:i + 24]
        X_F_HUMIDITY = F_HUMIDITY_B[i:i + 24]
        X_F_TEMPERATURE = F_TEMPERATURE_B[i:i + 24]
        X_F_WIND_DIRECTION = F_WIND_DIRECTION_B[i:i + 24]
        X_F_WIND_SPEED = F_WIND_SPEED_B[i:i + 24]
        X_F_TIME = F_MONITOR_TIME_B[i:i + 24]

        x_frames.append(X_F_AIR_PRESSURE)
        x_frames.append(X_F_HUMIDITY)
        x_frames.append(X_F_TEMPERATURE)
        x_frames.append(X_F_WIND_DIRECTION)
        x_frames.append(X_F_WIND_SPEED)
        time_frames.append(X_F_TIME)

        x1 = np.array(x_frames).T
        x2 = np.array(time_frames)
        x=np.concatenate((x1,x2),axis=2)
        weather_list.append(x.tolist())

    return weather_list

 
def get_div_localtion_dic(div_station_names):
    
    station_path='D:/Projects/myProject/data_generation/air_forcast_data/stations13.csv'
    
    station_df = pd.read_csv(station_path)
    
    div_location_dic={}
    for div_name in div_station_names:
        new_station_df=station_df[station_df['STATION_CLUSTER']==div_name]
        station_location= new_station_df[['ZDJD','ZDWD']]
    
        div_location_np = np.array(station_location)
        div_location=np.mean(div_location_np, axis=0)
        #div_location=np.around(div_location, decimals=4)
        div_location_dic[div_name]=div_location.tolist()
    
    return div_location_dic

def get_sim_by_num_array(start_num,end_num,div_station_names,station_conns,station_dic,step):

    div_location_dic=get_div_localtion_dic(div_station_names)
    sim_frames=[]
    for station_conn in station_conns:
        start_station=station_conn[0]
        end_station=station_conn[1]
        start_station_localtion=div_location_dic[start_station]
        end_station_localtion=div_location_dic[end_station]
        wind_directions=station_dic[start_station]
        sim_location=cu.get_sim_location(start_station_localtion,end_station_localtion)
        sim_location_A=np.array(sim_location)
        sim_location_B=np.tile(sim_location_A, [len(wind_directions),1])
        sim_location_B = np.around(sim_location_B, decimals=4)
        wind_directions_A=cu.get_sim_wind_array(wind_directions,start_station_localtion,end_station_localtion)
        wind_directions_A = wind_directions_A.reshape(sim_location_B.shape)
        per_sim_frame = np.concatenate((sim_location_B, wind_directions_A), axis=1)
        sim_frames.append(per_sim_frame)
    sim_frames=np.array(sim_frames).transpose(1,0,2)
    #----逐24小时取样本----------------------
    conn_sim=[]
    for i in range(start_num,end_num,step):
        x_frames=sim_frames[i:i+24]
        conn_sim.append(x_frames.tolist())
    return conn_sim


def get_poi(div_station_names):
    station_poi_path="D:/Projects/myProject/data_generation/air_forcast_data/air_poi.json"
    poi_dict=[]
    poi=[]
    for poi_json in open(station_poi_path):
        poi_dict=eval(poi_json)
    for div_station_name in div_station_names:
        if div_station_name in poi_dict.keys():
            poi.append(eval(poi_dict[div_station_name]))
    return poi
    
def gen_conn(div_station_names):
    div_num=len(div_station_names)
    div_station_names=list(div_station_names)
    conn=[]
    station_conns=[]
    for i in range(div_num): 
        j=i+1
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

def split_div_station_feature_train():
    
    train_fout = open("D:/Projects/myProject/data_generation/v2_air_data/DIV_train.json", 'w')
    val_fout = open("D:/Projects/myProject/data_generation/v2_air_data/DIV_val.json", 'w')
    station_data_path="D:/Projects/myProject/data_generation/v2_air_data/stations.json"

    '''
    { "DIV1":[],"DIV2":[],...,"DIVM":[],
    "conn":[[0,1],[0,2],...[0,M],[1,0],[1,2],...[1,M],..[M,M-1]],
    "sim":[[[[s1,s2],...边数...],...24...]...15000...],
    "weather":[[[[气象特征1，气象特征2，....，气象特征10],.....24.....],...M....],...15000...],
    "pm2.5":[[[浓度1,浓度2,....浓度24],...M...],...15000...]
    }
    '''
    div_station_dic={}
    for data in open(station_data_path,encoding='utf-8'):
        div_station_dic=eval(data)
    
    DIV_train_dict={}
    DIV_val_dict={}
    
    train_num=15000
    val_num=1729

    div_station_names=[]
    div_conns=[]

    for data in open(station_data_path,encoding='utf-8'):
        DIV_data_json=eval(data)
        div_station_names=DIV_data_json.keys()
        for div_name in div_station_names:
            DIV_train_dict[div_name]=[]
            DIV_val_dict[div_name]=[]
            
        conn,station_conns=gen_conn(div_station_names)
        div_conns=station_conns
        DIV_train_dict["conn"]=conn
        DIV_val_dict["conn"]=conn

    station_feature_path="D:/Projects/myProject/data_generation/air_forcast_data/data/jn_data_2021/all_DIVs_station_final_train.csv"
    station_feature_df = pd.read_csv(station_feature_path)#所有数据
    
    train_weather_list=[]
    val_weather_list=[]
    
    train_pm25_list=[]
    val_pm25_list=[]
    
    div_direction_dic={}
    for key,value in div_station_dic.items():#区
        F_AIR_PRESSURE_list=[]
        F_HUMIDITY_list=[]
        F_TEMPERATURE_list=[]
        F_WIND_DIRECTION_list=[]
        F_WIND_SPEED_list=[]
        F_PM25_list=[]
        F_MONITOR_TIME_array = []
        select_station_df=station_feature_df[station_feature_df['DIV_LABLE']==key]#区数据
        for station_name in value:#区里的站点
            select_station= select_station_df[station_feature_df['F_STATION_NAME']==station_name]
            F_AIR_PRESSURE= select_station[['F_AIR_PRESSURE']]
            F_HUMIDITY=select_station[['F_HUMIDITY']]
            F_TEMPERATURE=select_station[['F_TEMPERATURE']]
            F_WIND_DIRECTION=select_station[['F_WIND_DIRECTION']]
            F_WIND_SPEED=select_station[['F_WIND_SPEED']]
            F_PM25=select_station[['F_PM25']]
            F_AIR_PRESSURE_list.append(np.array(F_AIR_PRESSURE).tolist())
            F_HUMIDITY_list.append(np.array(F_HUMIDITY).tolist())
            F_TEMPERATURE_list.append(np.array(F_TEMPERATURE).tolist())
            F_WIND_DIRECTION_list.append(np.array(F_WIND_DIRECTION).tolist())
            F_WIND_SPEED_list.append(np.array(F_WIND_SPEED).tolist())
            F_PM25_list.append(np.array(F_PM25).tolist())
            F_MONITOR_TIME_array = cu.batch_time_feature(np.array(select_station['F_MONITOR_TIME']))
        
        F_AIR_PRESSURE_A=np.array(F_AIR_PRESSURE_list)
        F_HUMIDITY_A=np.array(F_HUMIDITY_list)
        F_TEMPERATURE_A=np.array(F_TEMPERATURE_list)
        F_WIND_DIRECTION_A=np.array(F_WIND_DIRECTION_list)
        F_WIND_SPEED_A=np.array(F_WIND_SPEED_list)
        F_PM25_A=np.array(F_PM25_list)

        F_AIR_PRESSURE_B=np.mean(F_AIR_PRESSURE_A.astype(float), axis=0)
        F_AIR_PRESSURE_B=np.around(F_AIR_PRESSURE_B, decimals=4)
        F_HUMIDITY_B=np.mean(F_HUMIDITY_A.astype(float), axis=0)
        F_HUMIDITY_B=np.around(F_HUMIDITY_B, decimals=4)
        F_TEMPERATURE_B=np.mean(F_TEMPERATURE_A.astype(float), axis=0)
        F_TEMPERATURE_B=np.around(F_TEMPERATURE_B, decimals=4)
        F_WIND_DIRECTION_B=np.mean(F_WIND_DIRECTION_A.astype(float), axis=0)
        F_WIND_DIRECTION_B=np.around(F_WIND_DIRECTION_B, decimals=4)
        F_WIND_SPEED_B=np.mean(F_WIND_SPEED_A.astype(float), axis=0)
        F_WIND_SPEED_B=np.around(F_WIND_SPEED_B, decimals=4)
        F_MONITOR_TIME_B = np.array(F_MONITOR_TIME_array)
        F_PM25_B=np.mean(F_PM25_A.astype(float), axis=0)
        F_PM25_B=np.around(F_PM25_B, decimals=4)
    
        train_weather=select_train_dataset_by_num(0,train_num,F_AIR_PRESSURE_B,F_HUMIDITY_B,\
        F_TEMPERATURE_B,F_WIND_DIRECTION_B,F_WIND_SPEED_B,F_MONITOR_TIME_B)
        train_weather=np.array(train_weather).reshape(train_num,24, 10)
        train_weather_list.append(train_weather)
        
        val_weather=select_train_dataset_by_num(train_num,train_num+val_num,F_AIR_PRESSURE_B,F_HUMIDITY_B,\
        F_TEMPERATURE_B,F_WIND_DIRECTION_B,F_WIND_SPEED_B,F_MONITOR_TIME_B)
        val_weather=np.array(val_weather).reshape(val_num,24, 10)
        val_weather_list.append(val_weather)
        
        train_pm25=select_PM25_dataset_by_num(0,train_num,F_PM25_B,1)
        train_pm25=np.array(train_pm25).reshape(train_num,24)
        train_pm25_list.append(train_pm25)
        
        val_pm25=select_PM25_dataset_by_num(train_num,train_num+val_num,F_PM25_B,1)
        val_pm25=np.array(val_pm25).reshape(val_num,24)
        val_pm25_list.append(val_pm25)

        div_direction_dic[key]=F_WIND_DIRECTION_B
        
    train_weather_list=np.array(train_weather_list).transpose(1,0,2,3)      
    print("np.array(train_weather_list).shape: ",np.array(train_weather_list).shape)
    
    val_weather_list=np.array(val_weather_list).transpose(1,0,2,3)      
    print("np.array(val_weather_list).shape: ",np.array(val_weather_list).shape)
    
    DIV_train_dict["weather"]=train_weather_list.tolist()
    DIV_val_dict["weather"]=val_weather_list.tolist()

    train_pm25_list=np.array(train_pm25_list).transpose(1,0,2)  
    print("np.array(train_pm25_list).shape: ",np.array(train_pm25_list).shape)
    
    val_pm25_list=np.array(val_pm25_list).transpose(1,0,2)  
    print("np.array(val_pm25_list).shape: ",np.array(val_pm25_list).shape)
    
    DIV_train_dict["pm2.5"]=train_pm25_list.tolist()
    DIV_val_dict["pm2.5"]=val_pm25_list.tolist()
    step = 1
    train_conn_sim=get_sim_by_num_array(0,train_num,div_station_names,div_conns,div_direction_dic,step)
    print("np.array(train_conn_sim).shape: ",np.array(train_conn_sim).shape)
    
    val_conn_sim=get_sim_by_num_array(train_num,train_num+val_num,div_station_names,div_conns,div_direction_dic,step)
    print("np.array(val_conn_sim).shape: ",np.array(val_conn_sim).shape)
    
    DIV_train_dict["sim"]=train_conn_sim
    DIV_val_dict["sim"]=val_conn_sim

    json.dump(DIV_train_dict,train_fout)
    json.dump(DIV_val_dict,val_fout)
        
    train_fout.close()
    val_fout.close()
    
    print("OK")  

def split_div_station_feature_test():
    
    test_fout = open("D:/Projects/myProject/data_generation/v2_air_data/DIV_test.json", 'w')
    station_data_path = "D:/Projects/myProject/data_generation/v2_air_data/stations.json"
    '''
    { "DIV1":[],"DIV2":[],...,"DIVM":[],
    "conn":[[0,1],[0,2],...[0,M],[1,0],[1,2],...[1,M],..[M,M-1]],
    "sim":[[[[s1,s2],...边数...],...24...]...125...],
    "weather":[[[[气象特征1，气象特征2，气象特征3，气象特征4，气象特征5],.....24.....],...M....],...15000...],
    "pm2.5":[[[浓度1,浓度2,....浓度24],...M...],...125...]
    }
    '''
    div_station_dic={}
    for data in open(station_data_path,encoding='utf-8'):
        div_station_dic=eval(data)
    DIV_test_dict={}
    test_num=125
    div_station_names=[]
    #div_location_dic={}
    div_conns=[]

    for data in open(station_data_path, encoding='utf-8'):

        DIV_data_json=eval(data)
        div_station_names=DIV_data_json.keys()
        for div_name in div_station_names:
            DIV_test_dict[div_name]=[]

        conn,station_conns=gen_conn(div_station_names)
        div_conns=station_conns
        DIV_test_dict["conn"]=conn
       # div_location_dic=get_div_localtion_dic(div_station_names)
        
    station_feature_path = "D:/Projects/myProject/data_generation/air_forcast_data/data/jn_data_2021/all_DIVs_station_final_test.csv"
    station_feature_df = pd.read_csv(station_feature_path)

    test_weather_list=[]
    pm25_list=[]
    div_direction_dic={}
    for key,value in div_station_dic.items():
        F_AIR_PRESSURE_list=[]
        F_HUMIDITY_list=[]
        F_TEMPERATURE_list=[]
        F_WIND_DIRECTION_list=[]
        F_WIND_SPEED_list=[]
        F_MONITOR_TIME_list=[]
        F_PM25_list=[]
        select_station_df=station_feature_df[station_feature_df['DIV_LABLE']==key]
        for station_name in value:
            select_station= select_station_df[station_feature_df['F_STATION_NAME']==station_name]
            F_AIR_PRESSURE= select_station[['F_AIR_PRESSURE']]
            F_HUMIDITY=select_station[['F_HUMIDITY']]
            F_TEMPERATURE=select_station[['F_TEMPERATURE']]
            F_WIND_DIRECTION=select_station[['F_WIND_DIRECTION']]
            F_WIND_SPEED=select_station[['F_WIND_SPEED']]
            F_PM25 = select_station[['F_PM25']]
            F_MONITOR_TIME=select_station['F_MONITOR_TIME']
            F_TIME=cu.batch_time_feature(np.array(F_MONITOR_TIME))

            F_AIR_PRESSURE_list.append(np.array(F_AIR_PRESSURE).tolist())
            F_HUMIDITY_list.append(np.array(F_HUMIDITY).tolist())
            F_TEMPERATURE_list.append(np.array(F_TEMPERATURE).tolist())
            F_WIND_DIRECTION_list.append(np.array(F_WIND_DIRECTION).tolist())
            F_WIND_SPEED_list.append(np.array(F_WIND_SPEED).tolist())
            F_PM25_list.append(np.array(F_PM25).tolist())
            F_MONITOR_TIME_list.append(F_TIME)
        
        F_AIR_PRESSURE_A=np.array(F_AIR_PRESSURE_list)
        F_HUMIDITY_A=np.array(F_HUMIDITY_list)
        F_TEMPERATURE_A=np.array(F_TEMPERATURE_list)
        F_WIND_DIRECTION_A=np.array(F_WIND_DIRECTION_list)
        F_WIND_SPEED_A=np.array(F_WIND_SPEED_list)
        F_PM25_A=np.array(F_PM25_list)

        F_MONITOR_TIME_B=np.array(F_MONITOR_TIME_list)
        F_AIR_PRESSURE_B=np.mean(F_AIR_PRESSURE_A.astype(float), axis=0)
        F_AIR_PRESSURE_B=np.around(F_AIR_PRESSURE_B, decimals=4)
        F_HUMIDITY_B=np.mean(F_HUMIDITY_A.astype(float), axis=0)
        F_HUMIDITY_B=np.around(F_HUMIDITY_B, decimals=4)
        F_TEMPERATURE_B=np.mean(F_TEMPERATURE_A.astype(float), axis=0)
        F_TEMPERATURE_B=np.around(F_TEMPERATURE_B, decimals=4)
        F_WIND_DIRECTION_B=np.mean(F_WIND_DIRECTION_A.astype(float), axis=0)
        F_WIND_SPEED_B=np.mean(F_WIND_SPEED_A.astype(float), axis=0)
        F_WIND_SPEED_B=np.around(F_WIND_SPEED_B, decimals=4)
        F_PM25_B=np.mean(F_PM25_A.astype(float), axis=0)
        F_PM25_B=np.around(F_PM25_B, decimals=4)

        start_num = 0
        end_num=F_PM25_B.shape[0]-24
        #--气象特征中加入时间特征
        test_weather=select_test_dataset_by_num(start_num,end_num,F_AIR_PRESSURE_B,F_HUMIDITY_B,\
        F_TEMPERATURE_B,F_WIND_DIRECTION_B,F_WIND_SPEED_B, F_MONITOR_TIME_B[1,:])
        test_weather=np.array(test_weather).reshape(test_num,24, 10)

        test_weather_list.append(test_weather)
        pm25=select_PM25_dataset_by_num(start_num,end_num,F_PM25_B,24)
        pm25=np.array(pm25).reshape(test_num,24)
        pm25_list.append(pm25)
        div_direction_dic[key]=F_WIND_DIRECTION_B
        
    test_weather_list=np.array(test_weather_list).transpose(1,0,2,3)
    DIV_test_dict["weather"]=test_weather_list.tolist()
    pm25_list=np.array(pm25_list).transpose(1,0,2)
    DIV_test_dict["pm2.5"]=pm25_list.tolist()
    start_num=0
    end_num=div_direction_dic['DIV1'].shape[0]-24
    step=24
    conn_sim=get_sim_by_num_array(start_num,end_num,div_station_names,div_conns,div_direction_dic,step)
    
    print("np.array(conn_sim).shape: ",np.array(conn_sim).shape) 
    DIV_test_dict["sim"]=conn_sim
    
    json.dump(DIV_test_dict,test_fout)
    test_fout.close()
    print("OK")      
    
def split_div_station_feature():
    split_div_station_feature_test()
    print("test_finished")
    split_div_station_feature_train()
    print("all_finished")

if __name__ == '__main__':
    # get_station_localtion_dic()#测试用
    # split_div_station_feature_test()#测试用
    split_div_station_feature()
  


