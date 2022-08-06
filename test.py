import argparse
import json
import math
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

from dataset import testDataset
from model import CityModel, GlobalModel
from utils import result_display, math_utils
from torch_geometric.nn import MetaLayer

parser = argparse.ArgumentParser(description='Multi-city AQI forecasting')
parser.add_argument('--device', type=str, default='cuda', help='')
parser.add_argument('--run_times', type=int, default=1, help='')#运行N次求平均
parser.add_argument('--epoch', type=int, default=500, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--div_num', type=int, default=13, help='')
parser.add_argument('--gnn_h', type=int, default=32, help='')
parser.add_argument('--rnn_h', type=int, default=64, help='')
parser.add_argument('--rnn_l', type=int, default=1, help='')
parser.add_argument('--aqi_em', type=int, default=16, help='')
parser.add_argument('--poi_em', type=int, default=8, help='poi embedding')
parser.add_argument('--wea_em', type=int, default=12, help='wea embedding')
parser.add_argument('--lr', type=float, default=0.001, help='lr')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--pred_step', type=int, default=24, help='step')
parser.add_argument('--enable-cuda', default=True, help='Enable CUDA')
args = parser.parse_args()

device = args.device
test_dataset = testDataset()
test_loader = Data.DataLoader(test_dataset,
                              batch_size=args.batch_size,
                              num_workers=0,
                              shuffle=False)

mae_loss = torch.zeros(9, 4)
rmse_loss = torch.zeros(9, 4)
smape_loss=torch.zeros(9,4)
mae=[]
mae_num=0
rmse=[]
rmse_num=0
smape=[]
smape_num=0

def cal_loss(outputs, y, index):
    global mae, mae_num, rmse, rmse_num, smape, smape_num
    global mae_loss, rmse_loss, smape_loss
    temp_loss = torch.abs(outputs - y)
    mae_loss_1 = temp_loss[:, :, 0]
    mae_loss_3 = temp_loss[:, :, 5]
    mae_loss_6 = temp_loss[:, :, 11]
    mae_loss_12 = temp_loss[:, :, -1]
    mae_list = temp_loss.cpu().numpy()
    mae_num += mae_list.size
    mae.append(mae_list.sum())

    mae_loss[index, 0] += torch.mean(mae_loss_1, axis=(0, 1)).item()
    mae_loss[index, 1] += torch.mean(mae_loss_3, axis=(0, 1)).item()
    mae_loss[index, 2] += torch.mean(mae_loss_6, axis=(0, 1)).item()
    mae_loss[index, 3] += torch.mean(mae_loss_12, axis=(0, 1)).item()

    temp_loss = torch.pow(temp_loss, 2)
    rmse_loss_1 = temp_loss[:, :, 0]
    rmse_loss_3 = temp_loss[:, :, 5]
    rmse_loss_6 = temp_loss[:, :, 11]
    rmse_loss_12 = temp_loss[:, :, -1]
    rmse_list = temp_loss.cpu().numpy()
    rmse_num += rmse_list.size
    rmse.append(rmse_list.sum())

    rmse_loss[index, 0] += torch.mean(rmse_loss_1, axis=(0, 1)).item()
    rmse_loss[index, 1] += torch.mean(rmse_loss_3, axis=(0, 1)).item()
    rmse_loss[index, 2] += torch.mean(rmse_loss_6, axis=(0, 1)).item()
    rmse_loss[index, 3] += torch.mean(rmse_loss_12, axis=(0, 1)).item()

    temp_loss = 2.0 * torch.abs(outputs - y) / (torch.abs(outputs) + torch.abs(y))
    smape_loss_1 = temp_loss[:, :, 0]
    smape_loss_3 = temp_loss[:, :, 5]
    smape_loss_6 = temp_loss[:, :, 11]
    smape_loss_12 = temp_loss[:, :, -1]
    smape_list = temp_loss.cpu().numpy()
    smape_num += smape_list.size
    smape.append(smape_list.sum())

    smape_loss[index, 0] += torch.mean(smape_loss_1, axis=(0, 1)).item()
    smape_loss[index, 1] += torch.mean(smape_loss_3, axis=(0, 1)).item()
    smape_loss[index, 2] += torch.mean(smape_loss_6, axis=(0, 1)).item()
    smape_loss[index, 3] += torch.mean(smape_loss_12, axis=(0, 1)).item()
    #return mae_loss,rmse_loss, smape_loss

global_model = GlobalModel(args.aqi_em, args.rnn_h, args.rnn_l,args.gnn_h).to(device)
div1_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                          args.rnn_h, args.rnn_l, args.gnn_h).to(device)
div2_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                          args.rnn_h, args.rnn_l, args.gnn_h).to(device)
div3_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                          args.rnn_h, args.rnn_l, args.gnn_h).to(device)
div5_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                          args.rnn_h, args.rnn_l, args.gnn_h).to(device)
div6_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                          args.rnn_h, args.rnn_l, args.gnn_h).to(device)
# div7_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
#                           args.rnn_h, args.rnn_l, args.gnn_h).to(device)
div8_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                          args.rnn_h, args.rnn_l, args.gnn_h).to(device)
div9_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                          args.rnn_h, args.rnn_l, args.gnn_h).to(device)
div10_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                          args.rnn_h, args.rnn_l, args.gnn_h).to(device)
# div12_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
#                           args.rnn_h, args.rnn_l, args.gnn_h).to(device)
div13_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                          args.rnn_h, args.rnn_l, args.gnn_h).to(device)


global_model = torch.nn.DataParallel(global_model, device_ids=[0])#多个GPU来加速训练
div1_model = torch.nn.DataParallel(div1_model, device_ids=[0])
div2_model = torch.nn.DataParallel(div2_model, device_ids=[0])
div3_model = torch.nn.DataParallel(div3_model, device_ids=[0])
div5_model = torch.nn.DataParallel(div5_model, device_ids=[0])
div6_model = torch.nn.DataParallel(div6_model, device_ids=[0])
# div7_model = torch.nn.DataParallel(div7_model, device_ids=[0])
div8_model = torch.nn.DataParallel(div8_model, device_ids=[0])
div9_model = torch.nn.DataParallel(div9_model, device_ids=[0])
div10_model = torch.nn.DataParallel(div10_model, device_ids=[0])
# div12_model = torch.nn.DataParallel(div12_model, device_ids=[0])
div13_model = torch.nn.DataParallel(div13_model, device_ids=[0])

torch.backends.cudnn.benchmark = True

with torch.no_grad():
    global_model.load_state_dict(torch.load('./checkpoints/global.ckpt'))
    div1_model.load_state_dict(torch.load('./checkpoints/div1.ckpt'))
    div2_model.load_state_dict(torch.load('./checkpoints/div2.ckpt'))
    div3_model.load_state_dict(torch.load('./checkpoints/div3.ckpt'))
    div5_model.load_state_dict(torch.load('./checkpoints/div5.ckpt'))
    div6_model.load_state_dict(torch.load('./checkpoints/div6.ckpt'))
    # div7_model.load_state_dict(torch.load('./checkpoints/div7.ckpt'))
    div8_model.load_state_dict(torch.load('./checkpoints/div8.ckpt'))
    div9_model.load_state_dict(torch.load('./checkpoints/div9.ckpt'))
    div10_model.load_state_dict(torch.load('./checkpoints/div10.ckpt'))
    # div12_model.load_state_dict(torch.load('./checkpoints/div12.ckpt'))
    div13_model.load_state_dict(torch.load('./checkpoints/div13.ckpt'))

    div1_pre=[]
    div1_ture = []
    for i, (divs_data, div1_data,div2_data,div3_data,div5_data,div6_data,div8_data,
            div9_data,div10_data,div13_data) in enumerate(test_loader):
        divs_con, divs_conn, divs_sim, _ = [x.to(device) for x in divs_data]
        div_u = global_model(divs_con, divs_conn, divs_sim, args.div_num)

        div1_data = [item.to(device, non_blocking=True) for item in div1_data]
        div1_outputs = div1_model(div1_data, div_u[:, :, 0], device)
        cal_loss(div1_outputs, div1_data[-1], 0)

        div2_data = [item.to(device, non_blocking=True) for item in div2_data]
        div2_outputs = div2_model(div2_data, div_u[:, :, 1], device)
        cal_loss(div2_outputs, div2_data[-1], 1)

        div3_data = [item.to(device, non_blocking=True) for item in div3_data]
        div3_outputs = div3_model(div3_data, div_u[:, :, 2], device)
        cal_loss(div3_outputs, div3_data[-1], 2)

        div5_data = [item.to(device, non_blocking=True) for item in div5_data]
        div5_outputs = div5_model(div5_data, div_u[:, :, 4], device)
        cal_loss(div5_outputs, div5_data[-1], 3)

        div6_data = [item.to(device, non_blocking=True) for item in div6_data]
        div6_outputs = div6_model(div6_data, div_u[:, :, 5], device)
        cal_loss(div6_outputs, div6_data[-1], 4)

        # div7_data = [item.to(device, non_blocking=True) for item in div7_data]
        # div7_outputs = div7_model(div7_data, div_u[:, :, 6], device)
        # cal_loss(div7_outputs, div7_data[-1], 5)

        div8_data = [item.to(device, non_blocking=True) for item in div8_data]
        div8_outputs = div8_model(div8_data, div_u[:, :, 7], device)
        cal_loss(div8_outputs, div8_data[-1], 5)

        div9_data = [item.to(device, non_blocking=True) for item in div9_data]
        div9_outputs = div9_model(div9_data, div_u[:, :, 8], device)
        cal_loss(div9_outputs, div9_data[-1], 6)

        div10_data = [item.to(device, non_blocking=True) for item in div10_data]
        div10_outputs = div10_model(div10_data, div_u[:, :, 9], device)
        cal_loss(div10_outputs, div10_data[-1], 7)

        # div12_data = [item.to(device, non_blocking=True) for item in div12_data]
        # div12_outputs = div12_model(div12_data, div_u[:, :, 11], device)
        # cal_loss(div12_outputs, div12_data[-1], 9)

        div13_data = [item.to(device, non_blocking=True) for item in div13_data]
        div13_outputs = div13_model(div13_data, div_u[:, :, 12], device)
        cal_loss(div13_outputs, div13_data[-1], 8)

        ###获取div1预测标签，并进行合并处理####
        temp= div1_outputs.transpose(0, 1).contiguous()
        temp2=(temp.reshape(temp.size(0), -1)).cuda().data.cpu().numpy()
        if i==0:
            div1_pre_temp = temp2
        else:
            div1_pre_temp=np.hstack((div1_pre_temp,temp2))
        div1_pre = div1_pre_temp.tolist()

        ###获取真实标签，并进行合并处理####
        temp = div1_data[-1].transpose(0, 1).contiguous()
        temp2 = (temp.reshape(temp.size(0), -1)).cuda().data.cpu().numpy()
        if i == 0:
            div1_ture_temp = temp2
        else:
            div1_ture_temp = np.hstack((div1_ture_temp, temp2))
        div1_ture = div1_ture_temp.tolist()

    step = math.ceil(len(test_dataset) / args.batch_size)
    t_mae_loss = mae_loss.numpy()/ step
    t_rmse_loss = rmse_loss.numpy()/ step
    t_smape_loss = smape_loss.numpy()/ step
    print('(1,3,6,12)hours mae:', t_mae_loss)
    print('(1,3,6,12)hours rmse:', t_rmse_loss)
    print('(1,3,6,12)hours smape:', t_smape_loss)

    pre_data = div1_pre
    ture_data = div1_ture
    for id_station in range(1, len(pre_data)+1, 1):
        result_display.draw_site_data(id_station, pre_data, ture_data, 'pm2.5')

    mae_show = np.sum(np.array(mae))/ mae_num
    rmse_show = np.sum(np.array(rmse)) /rmse_num
    smape_show = np.sum(np.array(smape)) /smape_num
    print('mae:', mae_show)
    print('rmse:',rmse_show)
    print('smape:',smape_show )


print('Testing model finished!')
