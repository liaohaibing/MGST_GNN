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

from dataset import trainDataset, valDataset
from model import CityModel, GlobalModel
from torch_geometric.nn import MetaLayer

parser = argparse.ArgumentParser(description='Multi-city AQI forecasting')
parser.add_argument('--device', type=str, default='cuda', help='')
parser.add_argument('--run_times', type=int, default=1, help='')#运行5次求平均
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

train_dataset = trainDataset()
train_loader = Data.DataLoader(train_dataset,
                               batch_size=args.batch_size,
                               num_workers=0,
                               shuffle=True)

val_dataset = valDataset()
val_loader = Data.DataLoader(val_dataset,
                             batch_size=args.batch_size,
                             num_workers=0,
                             shuffle=True)

for runtimes in range(args.run_times):

    global_model = GlobalModel(args.aqi_em, args.rnn_h, args.rnn_l,
                               args.gnn_h).to(device)
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
    div8_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                           args.rnn_h, args.rnn_l, args.gnn_h).to(device)
    div9_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                           args.rnn_h, args.rnn_l, args.gnn_h).to(device)
    div10_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                           args.rnn_h, args.rnn_l, args.gnn_h).to(device)

    div13_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                           args.rnn_h, args.rnn_l, args.gnn_h).to(device)

    global_model = torch.nn.DataParallel(global_model, device_ids=[0])#多个GPU来加速训练
    div1_model = torch.nn.DataParallel(div1_model, device_ids=[0])
    div2_model = torch.nn.DataParallel(div2_model, device_ids=[0])
    div3_model = torch.nn.DataParallel(div3_model, device_ids=[0])
    #div4_model = torch.nn.DataParallel(div4_model, device_ids=[0, 1, 2, 3])
    div5_model = torch.nn.DataParallel(div5_model, device_ids=[0])
    div6_model = torch.nn.DataParallel(div6_model, device_ids=[0])
    # div7_model = torch.nn.DataParallel(div7_model, device_ids=[0,1])
    div8_model = torch.nn.DataParallel(div8_model, device_ids=[0])
    div9_model = torch.nn.DataParallel(div9_model, device_ids=[0])
    div10_model = torch.nn.DataParallel(div10_model, device_ids=[0])
    div13_model = torch.nn.DataParallel(div13_model, device_ids=[0])

    div_model_num = sum(p.numel() for p in global_model.parameters()
                         if p.requires_grad)
    print('city_model:', 'Trainable,', div_model_num)

    div1_model_num = sum(p.numel() for p in div1_model.parameters()
                         if p.requires_grad)
    div2_model_num = sum(p.numel() for p in div2_model.parameters()
                             if p.requires_grad)
    div3_model_num = sum(p.numel() for p in div3_model.parameters()
                         if p.requires_grad)
    # div4_model_num = sum(p.numel() for p in div4_model.parameters()
    #                      if p.requires_grad)
    div5_model_num = sum(p.numel() for p in div5_model.parameters()
                         if p.requires_grad)
    div6_model_num = sum(p.numel() for p in div6_model.parameters()
                         if p.requires_grad)
    # div7_model_num = sum(p.numel() for p in div7_model.parameters()
    #                      if p.requires_grad)
    div8_model_num = sum(p.numel() for p in div8_model.parameters()
                         if p.requires_grad)
    div9_model_num = sum(p.numel() for p in div9_model.parameters()
                         if p.requires_grad)
    div10_model_num = sum(p.numel() for p in div10_model.parameters()
                         if p.requires_grad)
    div13_model_num = sum(p.numel() for p in div13_model.parameters()
                         if p.requires_grad)

    criterion = nn.MSELoss(reduction='mean')
    #criterion = nn.SmoothL1Loss(reduction='mean')
    params = list(global_model.parameters()) + list(div1_model.parameters())+list(div2_model.parameters())+\
             list(div3_model.parameters())+list(div5_model.parameters())+\
             list(div6_model.parameters())+list(div8_model.parameters())+\
             list(div9_model.parameters())+list(div10_model.parameters())+list(div13_model.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)

    val_loss_min = np.inf
    for epoch in range(args.epoch):
        for i, (divs_data, div1_data,div2_data,div3_data,div5_data,div6_data,div8_data,
                div9_data,div10_data,div13_data) in enumerate(train_loader):
            divs_aqi, divs_conn, divs_sim,divs_weather = [x.to(device) for x in divs_data]
            #print(cities_aqi.shape, cities_conn.shape,cities_sim.shape,cities_weather.shape)
            # 计算全局的城市属性[128,24,10,32]，将和站点的气象合并组成最终站点的全局属性
            div_u = global_model(divs_aqi, divs_conn, divs_sim, args.div_num)

            div1_data = [item.to(device, non_blocking=True) for item in div1_data]
            div1_outputs = div1_model(div1_data, div_u[:, :, 0], device)  # div1对应编号为0
            div1_loss = criterion(div1_outputs, div1_data[-1])

            div2_data = [item.to(device, non_blocking=True) for item in div2_data]
            div2_outputs = div2_model(div2_data, div_u[:, :, 1], device)#
            div2_loss = criterion(div2_outputs, div2_data[-1])

            div3_data = [item.to(device, non_blocking=True) for item in div3_data]
            div3_outputs = div3_model(div3_data, div_u[:, :, 2], device)  #
            div3_loss = criterion(div3_outputs, div3_data[-1])

            div5_data = [item.to(device, non_blocking=True) for item in div5_data]
            div5_outputs = div5_model(div5_data, div_u[:, :, 4], device)
            div5_loss = criterion(div5_outputs, div5_data[-1])

            div6_data = [item.to(device, non_blocking=True) for item in div6_data]
            div6_outputs = div6_model(div6_data, div_u[:, :, 5], device)
            div6_loss = criterion(div6_outputs, div6_data[-1])

            # div7_data = [item.to(device, non_blocking=True) for item in div7_data]
            # div7_outputs = div7_model(div7_data, div_u[:, :, 6], device)
            # div7_loss = criterion(div7_outputs, div7_data[-1])

            div8_data = [item.to(device, non_blocking=True) for item in div8_data]
            div8_outputs = div8_model(div8_data, div_u[:, :, 7], device)
            div8_loss = criterion(div8_outputs, div8_data[-1])

            div9_data = [item.to(device, non_blocking=True) for item in div9_data]
            div9_outputs = div9_model(div9_data, div_u[:, :, 8], device)
            div9_loss = criterion(div9_outputs, div9_data[-1])

            div10_data = [item.to(device, non_blocking=True) for item in div10_data]
            div10_outputs = div10_model(div10_data, div_u[:, :, 9], device)
            div10_loss = criterion(div10_outputs, div10_data[-1])

            # div11_data = [item.to(device, non_blocking=True) for item in div11_data]
            # div11_outputs = div11_model(div11_data, div_u[:, :, 10], device)
            # div11_loss = criterion(div11_outputs, div11_data[-1])

            # div12_data = [item.to(device, non_blocking=True) for item in div12_data]
            # div12_outputs = div12_model(div12_data, div_u[:, :, 11], device)
            # div12_loss = criterion(div12_outputs, div12_data[-1])

            div13_data = [item.to(device, non_blocking=True) for item in div13_data]
            div13_outputs = div13_model(div13_data, div_u[:, :, 12], device)
            div13_loss = criterion(div13_outputs, div13_data[-1])
            #每一个batch时并不需要与其他batch的梯度混合起来累积计算，
            # 因此需要对每个batch调用一遍zero_grad（）将参数梯度置0.可以用model.zero_grad() or optimizer.zero_grad()
            div1_model.zero_grad()
            div2_model.zero_grad()
            div3_model.zero_grad()
            #div4_model.zero_grad()
            div5_model.zero_grad()
            div6_model.zero_grad()
            # div7_model.zero_grad()
            div8_model.zero_grad()
            div9_model.zero_grad()
            div10_model.zero_grad()
            #div11_model.zero_grad()
            # div12_model.zero_grad()
            div13_model.zero_grad()
            global_model.zero_grad()

            loss = div1_loss+div2_loss+div3_loss+div5_loss+\
                   div6_loss+div8_loss+div9_loss+div10_loss+div13_loss

            loss.backward()
            optimizer.step()

            if i % 50 == 0 and epoch % 10 == 0:

                print('{},Epoch [{}/{}], Step [{}/{}], trainLoss: {:.4f}'.format(
                    'div1', epoch, args.epoch, i,
                    int(15000 / args.batch_size), div1_loss.item()))
                # print('{},Epoch [{}/{}], Step [{}/{}], trainLoss: {:.4f}'.format(
                #     'div3', epoch, args.epoch, i,
                #     int(15000 / args.batch_size), div3_loss.item()))
                print('{},Epoch [{}/{}], Step [{}/{}], trainLoss: {:.4f}'.format(
                    'div5', epoch, args.epoch, i,
                    int(15000 / args.batch_size), div5_loss.item()))
                # print('{},Epoch [{}/{}], Step [{}/{}], trainLoss: {:.4f}'.format(
                #     'div7', epoch, args.epoch, i,
                #     int(15000 / args.batch_size), div7_loss.item()))
                print('{},Epoch [{}/{}], Step [{}/{}], trainLoss: {:.4f}'.format(
                    'div9', epoch, args.epoch, i,
                    int(15000 / args.batch_size), div9_loss.item()))
                # print('{},Epoch [{}/{}], Step [{}/{}], trainLoss: {:.4f}'.format(
                #     'div11', epoch, args.epoch, i,
                #     int(15000 / args.batch_size), div11_loss.item()))
                # print('{},Epoch [{}/{}], Step [{}/{}], trainLoss: {:.4f}'.format(
                #     'div13', epoch, args.epoch, i,
                #     int(15000 / args.batch_size), div13_loss.item()))

        val_loss = 0
        with torch.no_grad(): #反向传播时就不会自动求导
            for j, (divs_val,div1_val,div2_val,div3_val,div5_val,div6_val,div8_val,\
                    div9_val, div10_val, div13_val) in enumerate(val_loader):

                divs_aqi_val, divs_conn_val, divs_sim_val, divs_weather = [x.to(device) for x in divs_val]
                #print(cities_aqi.shape, cities_conn.shape,cities_sim.shape,cities_weather.shape)
                div_u_val = global_model(divs_aqi_val, divs_conn_val, divs_sim_val, args.div_num)

                div1_val = [item.to(device, non_blocking=True) for item in div1_val]
                div1_outputs_val = div1_model(div1_val, div_u_val[:, :, 0], device)
                div1_loss_val = criterion(div1_outputs_val, div1_val[-1])

                div2_val = [item.to(device, non_blocking=True) for item in div2_val]
                div2_outputs_val = div2_model(div2_val, div_u_val[:, :, 1], device)
                div2_loss_val = criterion(div2_outputs_val, div2_val[-1])

                div3_val = [item.to(device, non_blocking=True) for item in div3_val]
                div3_outputs_val = div3_model(div3_val, div_u_val[:, :, 2], device)
                div3_loss_val = criterion(div3_outputs_val, div3_val[-1])

                # div4_data_val = [item.to(device, non_blocking=True) for item in div4_val]
                # div4_outputs_val = div4_model(div4_data_val, div_u_val[:, :, 3], device)
                # div4_loss_val = criterion(div4_outputs_val, div4_data_val[-1])

                div5_val = [item.to(device, non_blocking=True) for item in div5_val]
                div5_outputs_val = div5_model(div5_val, div_u_val[:, :, 4], device)
                div5_loss_val = criterion(div5_outputs_val, div5_val[-1])

                div6_val = [item.to(device, non_blocking=True) for item in div6_val]
                div6_outputs_val = div6_model(div6_val, div_u_val[:, :, 5], device)
                div6_loss_val = criterion(div6_outputs_val, div6_val[-1])

                # div7_val = [item.to(device, non_blocking=True) for item in div7_val]
                # div7_outputs_val = div7_model(div7_val, div_u_val[:, :, 6], device)
                # div7_loss_val = criterion(div7_outputs_val, div7_val[-1])

                div8_val = [item.to(device, non_blocking=True) for item in div8_val]
                div8_outputs_val = div8_model(div8_val, div_u_val[:, :, 7], device)
                div8_loss_val = criterion(div8_outputs_val, div8_val[-1])

                div9_val = [item.to(device, non_blocking=True) for item in div9_val]
                div9_outputs_val = div9_model(div9_val, div_u_val[:, :, 8], device)
                div9_loss_val = criterion(div9_outputs_val, div9_val[-1])

                div10_val = [item.to(device, non_blocking=True) for item in div10_val]
                div10_outputs_val = div10_model(div10_val, div_u_val[:, :, 9], device)
                div10_loss_val = criterion(div10_outputs_val, div10_val[-1])

                # div11_data_val = [item.to(device, non_blocking=True) for item in div11_data_val]
                # div11_outputs_val = div11_model(div11_data_val, div_u_val[:, :, 10], device)
                # div11_loss_val = criterion(div11_outputs_val, div11_data_val[-1])

                # div12_val = [item.to(device, non_blocking=True) for item in div12_val]
                # div12_outputs_val = div12_model(div12_val, div_u_val[:, :, 11], device)
                # div12_loss_val = criterion(div12_outputs_val, div12_val[-1])

                div13_val = [item.to(device, non_blocking=True) for item in div13_val]
                div13_outputs_val = div13_model(div13_val, div_u_val[:, :, 12], device)
                div13_loss_val = criterion(div13_outputs_val, div13_val[-1])

                val_loss = val_loss+div1_loss_val.item()+div2_loss_val.item()+div3_loss_val.item()+\
                           div5_loss_val.item()+div6_loss_val.item()+div8_loss_val.item()+\
                           div9_loss_val.item()+div10_loss_val.item()+div13_loss_val.item()

                if epoch % 10 == 0 and j % 10 == 0:
                    print('{},Epoch [{}/{}], Step [{}],valLoss: {:.4f}'.format(
                        'div1', epoch, args.epoch, j, div1_loss_val.item()))
                    # print('{},Epoch [{}/{}], Step [{}],valLoss: {:.4f}'.format(
                    #     'div3', epoch, args.epoch, j,div3_loss_val.item()))
                    print('{},Epoch [{}/{}],Step [{}], valLoss: {:.4f}'.format(
                        'div5', epoch, args.epoch, j, div5_loss_val.item()))
                    # print('{},Epoch [{}/{}], Step [{}],valLoss: {:.4f}'.format(
                    #     'div7', epoch, args.epoch,j, div7_loss_val.item()))
                    print('{},Epoch [{}/{}], Step [{}],valLoss: {:.4f}'.format(
                        'div9', epoch, args.epoch, j, div9_loss_val.item()))
                    # print('{},Epoch [{}/{}], Step [{}],valLoss: {:.4f}'.format(
                    #     'div11', epoch, args.epoch, j,div11_loss_val.item()))
                    # print('{},Epoch [{}/{}], Step [{}],valLoss: {:.4f}'.format(
                    #     'div13', epoch, args.epoch, j, div13_loss_val.item()))

            if val_loss < val_loss_min and epoch > (args.epoch * 0.5):
                torch.save(global_model.state_dict(),
                           './checkpoints/global.ckpt')
                torch.save(div1_model.state_dict(),
                           './checkpoints/div1.ckpt')
                torch.save(div2_model.state_dict(),
                           './checkpoints/div2.ckpt')
                torch.save(div3_model.state_dict(),
                           './checkpoints/div3.ckpt')
                # torch.save(div4_model.state_dict(),
                #            './checkpoints/div4.ckpt')
                torch.save(div5_model.state_dict(),
                           './checkpoints/div5.ckpt')
                torch.save(div6_model.state_dict(),
                           './checkpoints/div6.ckpt')
                # torch.save(div7_model.state_dict(),
                #            './checkpoints/div7.ckpt')
                torch.save(div8_model.state_dict(),
                           './checkpoints/div8.ckpt')
                torch.save(div9_model.state_dict(),
                           './checkpoints/div9.ckpt')
                torch.save(div10_model.state_dict(),
                           './checkpoints/div10.ckpt')
                # torch.save(div11_model.state_dict(),
                #            './checkpoints/div11.ckpt')
                # torch.save(div12_model.state_dict(),
                #            './checkpoints/div12.ckpt')
                torch.save(div13_model.state_dict(),
                           './checkpoints/div13.ckpt')
                val_loss_min = val_loss

    print('Finished Training')

