import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data

import os
import json
import numpy as np
import pandas as pd

TIME_WINDOW = 24
PRED_TIME = 24
parser = argparse.ArgumentParser(description='Multi-city data processing')
#parser.add_argument('--weather_mean',type=float, default=[987.0958, 56.8941, 16.1947,174.3973,1.4650],help='weather mean')
#parser.add_argument('--weather_std',type=float, default=[115.6563, 23.5957,10.6629,101.6264,1.5104],help='weather std')
parser.add_argument('--weather_min',type=float, default=[0.0614,0.00230, -18.8, 0.008, 0.0001,1,1,1,0,0],help='weather min')
parser.add_argument('--weather_max',type=float, default=[6328.0, 255.0, 50.0,  360.0, 20.0,12,31,7,23,1],help='weather max')
args = parser.parse_args()



DATA_PATH = './'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class trainDataset(Data.Dataset):
    def __init__(self, transform=None, train=True):
        # 如果文件中包含BOM字符，会报错：调用json.loads前加个判断处理
        with open(os.path.join(DATA_PATH, 'data/final_air_data_13divs', 'new_station.json'), 'r',
                  encoding='utf_8') as f:
            self.stations = json.load(f)

        with open(os.path.join(DATA_PATH,'data/final_air_data_13divs_24','DIV_train.json'), 'r', encoding='utf_8') as f:
            self.divs = json.load(f)
        print('DIV_sim shape:', np.array(self.divs['sim']).shape)
        print('DIV_weather shape:', np.array(self.divs['weather']).shape)
        print('DIV_pm2.5 shape:', np.array(self.divs['pm2.5']).shape)
        print('DIV_weather length:', len(self.divs['weather']))

        with open(os.path.join(DATA_PATH,'data/final_air_data_13divs_24','DIV1_train.json'), 'r', encoding='utf_8',) as f:
            self.div1 = json.load(f)
        print('div1_sim shape:', np.array(self.div1['sim']).shape)
        print('div1_weather shape:', np.array(self.div1['weather']).shape)
        print('div1_weather_for shape:', np.array(self.div1['weather_for']).shape)
        print('div1_weather length:', len(self.div1['weather']))

        with open(os.path.join(DATA_PATH,'data/final_air_data_13divs_24','DIV2_train.json'), 'r',encoding='utf_8') as f:
            self.div2 = json.load(f)
        with open(os.path.join(DATA_PATH,'data/final_air_data_13divs_24','DIV3_train.json'), 'r',encoding='utf_8') as f:
            self.div3 = json.load(f)
        # with open(os.path.join(DATA_PATH,'data/final_air_data_13divs','DIV4_train.json'), 'r') as f:
        #     self.div4 = json.load(f)
        with open(os.path.join(DATA_PATH,'data/final_air_data_13divs_24','DIV5_train.json'), 'r',encoding='utf_8') as f:
            self.div5 = json.load(f)
        with open(os.path.join(DATA_PATH,'data/final_air_data_13divs_24','DIV6_train.json'), 'r',encoding='utf_8') as f:
            self.div6 = json.load(f)
        # with open(os.path.join(DATA_PATH,'data/final_air_data_13divs_24','DIV7_train.json'), 'r',encoding='utf_8') as f:
        #     self.div7 = json.load(f)
        with open(os.path.join(DATA_PATH,'data/final_air_data_13divs_24','DIV8_train.json'), 'r',encoding='utf_8') as f:
            self.div8 = json.load(f)
        with open(os.path.join(DATA_PATH,'data/final_air_data_13divs_24','DIV9_train.json'), 'r',encoding='utf_8') as f:
            self.div9 = json.load(f)
        with open(os.path.join(DATA_PATH,'data/final_air_data_13divs_24','DIV10_train.json'), 'r',encoding='utf_8') as f:
            self.div10 = json.load(f)
        # with open(os.path.join(DATA_PATH,'data/final_air_data_13divs','DIV11_train.json'), 'r') as f:
        #     self.div11 = json.load(f)
        # with open(os.path.join(DATA_PATH,'data/final_air_data_13divs_24','DIV12_train.json'), 'r',encoding='utf_8') as f:
        #     self.div12 = json.load(f)
        with open(os.path.join(DATA_PATH,'data/final_air_data_13divs_24','DIV13_train.json'), 'r',encoding='utf_8') as f:
            self.div13 = json.load(f)

        self._norm()

    def _norm(self):
        self.divs['weather'] = ((np.array(self.divs['weather']) - args.weather_min) /
                                (np.array(args.weather_max)-np.array(args.weather_min))).tolist()

        self.div1['weather'] = ((np.array(self.div1['weather']) - args.weather_min) /
                                (np.array(args.weather_max)-np.array(args.weather_min))).tolist()
        self.div1['weather_for'] = ((np.array(self.div1['weather_for'])-args.weather_min) /
                                    (np.array(args.weather_max)-np.array(args.weather_min))).tolist()

        self.div2['weather'] = ((np.array(self.div2['weather']) - args.weather_min) /
                                (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div2['weather_for'] = ((np.array(self.div2['weather_for']) - args.weather_min) /
                                    (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

        self.div3['weather'] = ((np.array(self.div3['weather']) - args.weather_min) /
                                (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div3['weather_for'] = ((np.array(self.div3['weather_for']) - args.weather_min) /
                                    (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

        self.div5['weather'] = ((np.array(self.div5['weather']) - args.weather_min) /
                                (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div5['weather_for'] = ((np.array(self.div5['weather_for']) - args.weather_min) /
                                    (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

        self.div6['weather'] = ((np.array(self.div6['weather']) - args.weather_min) /
                                (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div6['weather_for'] = ((np.array(self.div6['weather_for']) - args.weather_min) /
                                    (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

        self.div8['weather'] = ((np.array(self.div8['weather']) - args.weather_min) /
                                (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div8['weather_for'] = ((np.array(self.div8['weather_for']) - args.weather_min) /
                                    (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

        self.div9['weather'] = ((np.array(self.div9['weather']) - args.weather_min) /
                                (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div9['weather_for'] = ((np.array(self.div9['weather_for']) - args.weather_min) /
                                    (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

        self.div10['weather'] = ((np.array(self.div10['weather']) - args.weather_min) /
                                 (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div10['weather_for'] = ((np.array(self.div10['weather_for']) - args.weather_min) /
                                     (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

        self.div13['weather'] = ((np.array(self.div13['weather']) - args.weather_min) /
                                 (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div13['weather_for'] = ((np.array(self.div13['weather_for']) - args.weather_min) /
                                     (np.array(args.weather_max) - np.array(args.weather_min))).tolist()


    def GetDivData(self, div_name, div_source, index):
        station_list = self.stations[div_name]
        div_con = []
        div_y = []
        for x in station_list:
            div_con.append(div_source[x][index][:TIME_WINDOW])
            div_y.append(div_source[x][index][TIME_WINDOW:])

        div_con = torch.FloatTensor(div_con)
        div_y = torch.FloatTensor(div_y)
        div_sim = torch.FloatTensor(div_source['sim'][index])
        div_conn = torch.tensor(div_source['conn'])
        div_weather = torch.FloatTensor(div_source['weather'][index])
        div_for = torch.FloatTensor(div_source['weather_for'][index])
        div_poi = torch.FloatTensor(div_source['poi'])

        div_data = [div_con, div_conn, div_poi, div_sim,
                div_weather, div_for, div_y]

        return div_data

    def __getitem__(self, index):
        div1_data = self.GetDivData('DIV1', self.div1, index)
        div2_data = self.GetDivData('DIV2', self.div2, index)
        div3_data = self.GetDivData('DIV3', self.div3, index)
        #div4_data = self.GetDivData('DIV4', self.div4, index)
        div5_data = self.GetDivData('DIV5', self.div5, index)
        div6_data = self.GetDivData('DIV6', self.div6, index)
        # div7_data = self.GetDivData('DIV7', self.div7, index)
        div8_data = self.GetDivData('DIV8', self.div8, index)
        div9_data = self.GetDivData('DIV9', self.div9, index)
        div10_data = self.GetDivData('DIV10', self.div10, index)
        #div11_data = self.GetDivData('DIV11', self.div11, index)
        # div12_data = self.GetDivData('DIV12', self.div12, index)
        div13_data = self.GetDivData('DIV13', self.div13, index)

        divs_con = torch.FloatTensor(self.divs['pm2.5'][index])
        divs_conn = torch.tensor(self.divs['conn'])
        divs_weather = torch.FloatTensor(self.divs['weather'][index])
        divs_sim = torch.FloatTensor(self.divs['sim'][index])

        divs_data = [divs_con, divs_conn, divs_sim, divs_weather]

        return divs_data,div1_data,div2_data,div3_data,div5_data,div6_data,\
               div8_data,div9_data,div10_data,div13_data

    def __len__(self):
        return len(self.div1['weather'])


class valDataset(Data.Dataset):
    def __init__(self, transform=None, train=True):
        with open(os.path.join(DATA_PATH, 'data/final_air_data_13divs', 'new_station.json'), 'r',
                  encoding='utf_8') as f:
            self.stations = json.load(f)

        with open(os.path.join(DATA_PATH,'data/final_air_data_13divs_24','DIV_val.json'), 'r',encoding='utf_8') as f:
            self.divs = json.load(f)
        print("the val numbers of div:", len(self.divs['pm2.5']))
        print("the val div_weather shape:", (np.array(self.divs['weather'])).shape)
        print("the val div_sim shape:", (np.array(self.divs['sim'])).shape)

        with open(os.path.join(DATA_PATH,'data/final_air_data_13divs_24', 'DIV1_val.json'), 'r',encoding='utf_8') as f:
            self.div1 = json.load(f)
        print("the val numbers of div1:", len(self.div1['sim']))
        print("the val div1 weather shape:", (np.array(self.div1['weather_for'])).shape)

        with open(os.path.join(DATA_PATH,'data/final_air_data_13divs_24', 'DIV2_val.json'), 'r',encoding='utf_8') as f:
            self.div2 = json.load(f)
        with open(os.path.join(DATA_PATH,'data/final_air_data_13divs_24', 'DIV3_val.json'), 'r',encoding='utf_8') as f:
            self.div3 = json.load(f)
        # with open(os.path.join(DATA_PATH,'data/final_air_data_13divs', 'DIV4_val.json'), 'r') as f:
        #     self.div4 = json.load(f)
        with open(os.path.join(DATA_PATH,'data/final_air_data_13divs_24', 'DIV5_val.json'), 'r',encoding='utf_8') as f:
            self.div5 = json.load(f)
        with open(os.path.join(DATA_PATH,'data/final_air_data_13divs_24', 'DIV6_val.json'), 'r',encoding='utf_8') as f:
            self.div6 = json.load(f)
        # with open(os.path.join(DATA_PATH, 'data/final_air_data_13divs_24', 'DIV7_val.json'), 'r',encoding='utf_8') as f:
        #     self.div7 = json.load(f)
        with open(os.path.join(DATA_PATH,'data/final_air_data_13divs_24', 'DIV8_val.json'), 'r',encoding='utf_8') as f:
            self.div8 = json.load(f)
        with open(os.path.join(DATA_PATH,'data/final_air_data_13divs_24', 'DIV9_val.json'), 'r',encoding='utf_8') as f:
            self.div9 = json.load(f)
        with open(os.path.join(DATA_PATH,'data/final_air_data_13divs_24', 'DIV10_val.json'), 'r',encoding='utf_8') as f:
            self.div10 = json.load(f)
        # with open(os.path.join(DATA_PATH,'data/final_air_data_13divs', 'DIV11_val.json'), 'r') as f:
        #     self.div11 = json.load(f)
        # with open(os.path.join(DATA_PATH,'data/final_air_data_13divs_24', 'DIV12_val.json'), 'r',encoding='utf_8') as f:
        #     self.div12 = json.load(f)
        with open(os.path.join(DATA_PATH,'data/final_air_data_13divs_24', 'DIV13_val.json'), 'r',encoding='utf_8') as f:
            self.div13 = json.load(f)

        self._norm()

    def _norm(self):
        self.divs['weather'] = ((np.array(self.divs['weather']) - args.weather_min) /
                                (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

        self.div1['weather'] = ((np.array(self.div1['weather']) - args.weather_min) /
                                (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div1['weather_for'] = ((np.array(self.div1['weather_for']) - args.weather_min) /
                                    (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

        self.div2['weather'] = ((np.array(self.div2['weather']) - args.weather_min) /
                                (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div2['weather_for'] = ((np.array(self.div2['weather_for']) - args.weather_min) /
                                    (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

        self.div3['weather'] = ((np.array(self.div3['weather']) - args.weather_min) /
                                (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div3['weather_for'] = ((np.array(self.div3['weather_for']) - args.weather_min) /
                                    (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

        self.div5['weather'] = ((np.array(self.div5['weather']) - args.weather_min) /
                                (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div5['weather_for'] = ((np.array(self.div5['weather_for']) - args.weather_min) /
                                    (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

        self.div6['weather'] = ((np.array(self.div6['weather']) - args.weather_min) /
                                (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div6['weather_for'] = ((np.array(self.div6['weather_for']) - args.weather_min) /
                                    (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

        self.div8['weather'] = ((np.array(self.div8['weather']) - args.weather_min) /
                                (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div8['weather_for'] = ((np.array(self.div8['weather_for']) - args.weather_min) /
                                    (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

        self.div9['weather'] = ((np.array(self.div9['weather']) - args.weather_min) /
                                (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div9['weather_for'] = ((np.array(self.div9['weather_for']) - args.weather_min) /
                                    (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

        self.div10['weather'] = ((np.array(self.div10['weather']) - args.weather_min) /
                                 (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div10['weather_for'] = ((np.array(self.div10['weather_for']) - args.weather_min) /
                                     (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

        self.div13['weather'] = ((np.array(self.div13['weather']) - args.weather_min) /
                                 (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div13['weather_for'] = ((np.array(self.div13['weather_for']) - args.weather_min) /
                                     (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

    def GetDivData(self,div_name,div_source,index):
        station_list = self.stations[div_name]
        div_con = []
        div_y = []
        for x in station_list:
            #print(243,  x, index, x in div_source)
            #if x in div_source:
            #    print(245, index in div_source[x])
            div_con.append(div_source[x][index][:TIME_WINDOW])
            div_y.append(div_source[x][index][TIME_WINDOW:])

        div_con = torch.FloatTensor(div_con)
        div_y = torch.FloatTensor(div_y)
        div_sim = torch.FloatTensor(div_source['sim'][index])
        div_conn = torch.tensor(div_source['conn'])
        div_weather = torch.FloatTensor(div_source['weather'][index])
        div_for = torch.FloatTensor(div_source['weather_for'][index])
        div_poi = torch.FloatTensor(div_source['poi'])

        div_data = [div_con, div_conn, div_poi, div_sim, div_weather, div_for, div_y]

        return div_data


    def __getitem__(self, index):
        div1_data = self.GetDivData('DIV1', self.div1, index)
        div2_data = self.GetDivData('DIV2', self.div2, index)
        div3_data = self.GetDivData('DIV3', self.div3, index)
        #div4_data = self.GetDivData('DIV4', self.div4, index)
        div5_data = self.GetDivData('DIV5', self.div5, index)
        div6_data = self.GetDivData('DIV6', self.div6, index)
        # div7_data = self.GetDivData('DIV7', self.div7, index)
        div8_data = self.GetDivData('DIV8', self.div8, index)
        div9_data = self.GetDivData('DIV9', self.div9, index)
        div10_data = self.GetDivData('DIV10', self.div10, index)
        #div11_data = self.GetDivData('DIV11', self.div11, index)
        # div12_data = self.GetDivData('DIV12', self.div12, index)
        div13_data = self.GetDivData('DIV13', self.div13, index)

        divs_con = torch.FloatTensor(self.divs['pm2.5'][index])
        divs_conn = torch.tensor(self.divs['conn'])
        divs_weather = torch.FloatTensor(self.divs['weather'][index])
        divs_sim = torch.FloatTensor(self.divs['sim'][index])
        divs_data = [divs_con, divs_conn, divs_sim, divs_weather]

        return divs_data, div1_data, div2_data, div3_data,  div5_data, div6_data,\
               div8_data, div9_data, div10_data,  div13_data

    def __len__(self):
        return len(self.div1['weather'])

class testDataset(Data.Dataset):
    def __init__(self, transform=None, train=True):
        with open(os.path.join(DATA_PATH, 'data/final_air_data_13divs', 'new_station.json'), 'r',
                  encoding='utf_8') as f:
            self.stations = json.load(f)
        with open(os.path.join(DATA_PATH,'data/final_air_data_13divs_24', 'DIV_test.json'), 'r',encoding='utf_8') as f:
            self.divs = json.load(f)
        print("the test numbers of div:", len(self.divs['pm2.5']))
        print("the test div_weather shape:", (np.array(self.divs['weather'])).shape)
        print("the test div_sim shape:", (np.array(self.divs['sim'])).shape)

        with open(os.path.join(DATA_PATH,'data/final_air_data_13divs_24', 'DIV1_test.json'), 'r',encoding='utf_8') as f:
            self.div1 = json.load(f)
        print("the test numbers of div1:", len(self.div1['sim']))
        print("the test div1 shape:", (np.array(self.div1['weather_for'])).shape)

        with open(os.path.join(DATA_PATH, 'data/final_air_data_13divs_24', 'DIV2_test.json'), 'r',encoding='utf_8') as f:
            self.div2 = json.load(f)
        with open(os.path.join(DATA_PATH, 'data/final_air_data_13divs_24', 'DIV3_test.json'), 'r',encoding='utf_8') as f:
            self.div3 = json.load(f)
        # with open(os.path.join(DATA_PATH, 'data/final_air_data_13divs', 'DIV4_test.json'), 'r') as f:
        #     self.div4 = json.load(f)
        with open(os.path.join(DATA_PATH, 'data/final_air_data_13divs_24', 'DIV5_test.json'), 'r',encoding='utf_8') as f:
            self.div5 = json.load(f)
        with open(os.path.join(DATA_PATH, 'data/final_air_data_13divs_24', 'DIV6_test.json'), 'r',encoding='utf_8') as f:
            self.div6 = json.load(f)
        # with open(os.path.join(DATA_PATH, 'data/final_air_data_13divs_24', 'DIV7_test.json'), 'r',encoding='utf_8') as f:
        #     self.div7 = json.load(f)
        with open(os.path.join(DATA_PATH, 'data/final_air_data_13divs_24', 'DIV8_test.json'), 'r',encoding='utf_8') as f:
            self.div8 = json.load(f)
        with open(os.path.join(DATA_PATH, 'data/final_air_data_13divs_24', 'DIV9_test.json'), 'r',encoding='utf_8') as f:
            self.div9 = json.load(f)
        with open(os.path.join(DATA_PATH, 'data/final_air_data_13divs_24', 'DIV10_test.json'), 'r',encoding='utf_8') as f:
            self.div10 = json.load(f)
        # with open(os.path.join(DATA_PATH, 'data/final_air_data_13divs', 'DIV11_test.json'), 'r') as f:
        #     self.div11 = json.load(f)
        # with open(os.path.join(DATA_PATH, 'data/final_air_data_13divs_24', 'DIV12_test.json'), 'r',encoding='utf_8') as f:
        #     self.div12 = json.load(f)
        with open(os.path.join(DATA_PATH, 'data/final_air_data_13divs_24', 'DIV13_test.json'), 'r',encoding='utf_8') as f:
            self.div13 = json.load(f)

        self._norm()

    def _norm(self):
        self.divs['weather'] = ((np.array(self.divs['weather']) - args.weather_min) /
                                (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

        self.div1['weather'] = ((np.array(self.div1['weather']) - args.weather_min) /
                                (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div1['weather_for'] = ((np.array(self.div1['weather_for']) - args.weather_min) /
                                    (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

        self.div2['weather'] = ((np.array(self.div2['weather']) - args.weather_min) /
                                (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div2['weather_for'] = ((np.array(self.div2['weather_for']) - args.weather_min) /
                                    (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

        self.div3['weather'] = ((np.array(self.div3['weather']) - args.weather_min) /
                                (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div3['weather_for'] = ((np.array(self.div3['weather_for']) - args.weather_min) /
                                    (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

        self.div5['weather'] = ((np.array(self.div5['weather']) - args.weather_min) /
                                (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div5['weather_for'] = ((np.array(self.div5['weather_for']) - args.weather_min) /
                                    (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

        self.div6['weather'] = ((np.array(self.div6['weather']) - args.weather_min) /
                                (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div6['weather_for'] = ((np.array(self.div6['weather_for']) - args.weather_min) /
                                    (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

        self.div8['weather'] = ((np.array(self.div8['weather']) - args.weather_min) /
                                (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div8['weather_for'] = ((np.array(self.div8['weather_for']) - args.weather_min) /
                                    (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

        self.div9['weather'] = ((np.array(self.div9['weather']) - args.weather_min) /
                                (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div9['weather_for'] = ((np.array(self.div9['weather_for']) - args.weather_min) /
                                    (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

        self.div10['weather'] = ((np.array(self.div10['weather']) - args.weather_min) /
                                 (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div10['weather_for'] = ((np.array(self.div10['weather_for']) - args.weather_min) /
                                     (np.array(args.weather_max) - np.array(args.weather_min))).tolist()

        self.div13['weather'] = ((np.array(self.div13['weather']) - args.weather_min) /
                                 (np.array(args.weather_max) - np.array(args.weather_min))).tolist()
        self.div13['weather_for'] = ((np.array(self.div13['weather_for']) - args.weather_min) /
                                     (np.array(args.weather_max) - np.array(args.weather_min))).tolist()


    def GetDivData(self,div_name,div_source,index):
        station_list = self.stations[div_name]
        div_con = []
        div_y = []
        for x in station_list:
            div_con.append(div_source[x][index][:TIME_WINDOW])
            div_y.append(div_source[x][index][TIME_WINDOW:])

        div_con = torch.FloatTensor(div_con)
        div_y = torch.FloatTensor(div_y)
        div_sim = torch.FloatTensor(div_source['sim'][index])
        div_conn = torch.tensor(div_source['conn'])
        div_weather = torch.FloatTensor(div_source['weather'][index])
        div_for = torch.FloatTensor(div_source['weather_for'][index])
        div_poi = torch.FloatTensor(div_source['poi'])

        div_data = [div_con, div_conn, div_poi, div_sim,div_weather, div_for, div_y]

        return div_data

    def __getitem__(self, index): #__getitem__的调用要通过： 对象[index]调用
        div1_data = self.GetDivData('DIV1', self.div1, index)
        div2_data = self.GetDivData('DIV2', self.div2, index)
        div3_data = self.GetDivData('DIV3', self.div3, index)
        #div4_data = self.GetDivData('DIV4', self.div4, index)
        div5_data = self.GetDivData('DIV5', self.div5, index)
        div6_data = self.GetDivData('DIV6', self.div6, index)
        # div7_data = self.GetDivData('DIV7', self.div7, index)
        div8_data = self.GetDivData('DIV8', self.div8, index)
        div9_data = self.GetDivData('DIV9', self.div9, index)
        div10_data = self.GetDivData('DIV10', self.div10, index)
        #div11_data = self.GetDivData('DIV11', self.div11, index)
        # div12_data = self.GetDivData('DIV12', self.div12, index)
        div13_data = self.GetDivData('DIV13', self.div13, index)

        divs_con = torch.FloatTensor(self.divs['pm2.5'][index])
        divs_conn = torch.tensor(self.divs['conn'])
        divs_weather = torch.FloatTensor(self.divs['weather'][index])
        divs_sim = torch.FloatTensor(self.divs['sim'][index])
        divs_data = [divs_con, divs_conn, divs_sim, divs_weather]

        return divs_data, div1_data, div2_data, div3_data, div5_data, div6_data, \
                div8_data, div9_data, div10_data,  div13_data

    def __len__(self):
        return len(self.div1['weather'])

