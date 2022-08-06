import numpy as np
from matplotlib import pyplot as plt

def draw_site_data(site_id, data_pre, data_ture, air_factor_name):

    y_pred_station = data_pre[site_id-1][0:300]
    y_true_station = data_ture[site_id-1][0:300]
    time_list = list(range(len(y_pred_station)))

    plt.figure(figsize=(100, 6))
    plt.grid(axis="y")
    #x_major_locator = plt.MultipleLocator(5)  # 24小时间隔
    y_major_locator = plt.MultipleLocator(20)
    xticks = range(0,len(time_list)+1,6)
    ax = plt.gca()
    plt.xticks(rotation=300)
    #ax.xaxis.set_major_locator(x_major_locator)
    ax.set_xticks(xticks)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.plot(time_list, y_true_station, linewidth=1, linestyle="-", label="true value", c='red')
    plt.plot(time_list, y_pred_station, linewidth=1, linestyle="--", label="predict value", c='green')
    plt.legend(loc='best')  # 显示图例，前提是plot参数里写上label;loc是图例的位置
    plt.ylabel(air_factor_name)
    plt.xlabel('time/h')
    plt.title(site_id, fontsize=16)
    plt.grid(linestyle='-.')
    minx = 0
    maxx = len(time_list)
    miny = min(y_true_station)
    maxy = max(y_true_station)-20
    plt.xlim(minx, maxx)
    plt.ylim(miny, maxy)
    save_path = './test_result24/%s.png' % (site_id)
    plt.savefig(save_path,bbox_inches='tight', pad_inches=0.0)

