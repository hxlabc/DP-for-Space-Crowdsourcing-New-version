import argparse

import numpy as np

from dataprocess import p_bs, lng_min, lng_max, lat_min, lat_max
from draw import draw_linegraph
from fjag_hgrc import data_load, FJAG, HGRC
from utils import epsilon_after_sampling, worker_avg_distance


# 输入参数
def parse_opt():
    parser = argparse.ArgumentParser(description="Do Experiment")
    parser.add_argument('--k', type=int, default=3, help='Number of classifications')
    parser.add_argument('--a', type=float, default=0.3, help='Budget factor1')
    parser.add_argument('--B', type=float, default=0.5, help='Budget factor2')
    parser.add_argument('--thres', type=float, default=0.95, help='threshold')
    parser.add_argument('--dmax', type=float, default=1600, help='Maximum distance')
    parser.add_argument('--pmax', type=float, default=0.3, help='Maximum acceptance rate')
    parser.add_argument('--linear', default=False, help='Whether to use linear')
    parser.add_argument('--source', type=str, default='./data/Geolife.xls', help='Data path')
    parser.add_argument('--save', default=False, help='Whether to save the picture')

    opt = parser.parse_args()
    parser.print_help()
    print(opt)

    return opt


def do_experiment(opt):
    k = opt.k   # 分类数目
    a = opt.a   # 预算因子1
    B = opt.B   # 预算因子2
    thres = opt.thres   # GR接受率阈值
    dmax = opt.dmax     # 工作者最大旅行距离
    pmax = opt.pmax     # 工作者最大接受率
    is_linear = opt.linear  # 是否采用线性函数
    file_path = opt.source   # 数据的路径
    times = 1000        # 任务个数
    epsilon_list = [0.2, 0.4, 0.6, 0.8, 1]   # 隐私预算集合
    dis_list = []      # 平均旅行距离集合
    num_list = []      # 平均工作者数量集合
    # 随机任务坐标集合
    tlist = [[np.random.uniform(lng_min, lng_max), np.random.uniform(lat_min, lat_max)] for _ in range(times)]

    datalen, lat, lng = data_load(file_path)
    for epsilon in epsilon_list:
        # 计算采样后的隐私预算
        n_epsilon = epsilon_after_sampling(epsilon, p_bs)
        # print(n_epsilon)
        alist, blist = FJAG(lng, lat, k, n_epsilon, a, B)

        gr_list = []
        num_sum = 0
        dis_sum = 0
        valid_times = 0     # 有效分配的任务数
        for tl in tlist:
            gr, gr_max = HGRC(tl[0], tl[1], thres, dmax, pmax, alist, blist, is_linear)
            gr_list.append([gr, gr_max])

            dis, num = worker_avg_distance(tl[0], tl[1], gr)
            num_sum += num
            if dis != -1:
                dis_sum += dis
                valid_times += 1
        num_avg = num_sum / valid_times
        dis_avg = dis_sum / valid_times

        dis_list.append(dis_avg)
        num_list.append(num_avg)

    # 画折线图
    draw_linegraph(epsilon_list, dis_list, 'dis', True)
    draw_linegraph(epsilon_list, num_list, 'num', True)


if __name__ == '__main__':
    opt = parse_opt()
    do_experiment(opt)
