import math
from math import radians, sin, cos, asin, sqrt
import numpy as np


# 分类之后每一类的方差和
def get_sdcm(point_list, divlist):
    point_list = sorted(point_list)
    var_list = []
    temp_list = []
    t = 0

    for item in point_list:
        if item <= divlist[t]:
            temp_list.append(item)
        else:
            var_list.append(np.var(temp_list))
            temp_list = []
            temp_list.append(item)
            t += 1
    var_list.append(np.var(temp_list))

    sdcm = sum(var_list)
    return sdcm


# 原始数据整体方差
def get_sdam(point_list):
    sdam = np.var(point_list)
    return sdam


# SDCM越小，GVF越大，分类效果越好
def get_gvf(point_list, divlist):
    sdcm = get_sdcm(point_list, divlist)
    sdam = get_sdam(point_list)
    gvf = 1 - sdcm/sdam
    return gvf


# 采样后的隐私预算
def epsilon_after_sampling(epsilon, p):
    new_epsilon = np.log(np.exp(epsilon) - 1 + p) - np.log(p)
    return new_epsilon


# 根据分数计算每一个区间的可能性（指数机制）
def score_p(epsilon, sensitivity, score):
    probability = np.exp(-1 * epsilon * score / (2 * sensitivity))
    return probability


# 计算第一网格粒度
def get_m1(N, epsilon):
    c1 = 10
    res = np.sqrt((N * epsilon) / c1) / 4
    res = math.ceil(res)
    return max(10, res)


# 计算第二网格粒度
def get_m2(N_noise, epsilon):
    c2 = np.sqrt(2)
    res = np.sqrt((N_noise * epsilon) / c2)
    res = math.ceil(res)
    return res


# 计算两点间距离
def get_distance(x1, y1, x2, y2):
    xd = x1 - x2
    yd = y1 - y2
    d = np.sqrt(xd**2 + yd**2)
    return d


# 计算经纬度坐标之间的距离，单位是米
def haversine_dis(lng1, lat1, lng2, lat2):
    # 将十进制转为弧度
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])

    # haversine公式
    d_lng = lng2 - lng1
    d_lat = lat2 - lat1
    aa = sin(d_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(d_lng / 2) ** 2
    c = 2 * asin(sqrt(aa))
    r = 6371  # 地球半径，千米

    return c * r * 1000    # 返回结果的单位是米


# 网格接受率
def area_receive_rate(n, p):
    return 1 - (1-p)**n


# 工作者接受率（线性变化）
def worker_rec_rate_linear(pmax, d, dmax):
    if dmax < d:
        return 0
    return pmax * (1 - d / dmax)


# 工作者接受率（非线性变化）
def worker_rec_rate_nonlinear(pmax, d, dmax):
    if dmax < d:
        return 0
    x = dmax - d

    # 加上除以dmax避免自变量过大，接受率也过高
    return np.tanh(x / dmax) * pmax


# 网格接受率
def area_receive_rate(n, p):
    return 1 - (1-p)**n


# 最小子网格面积
def min_area(p, p_gr, thres, N, square):
    S_min = -1

    if 0 <= p_gr < 1 and 0 < p < 1 and N > 0:     # 分母不能为0
        temp = (1 - thres) / (1 - p_gr)

        N_min = math.log(temp, 1 - p)
        S_min = (N_min / N) * square

    return S_min


# 网格任务接受率和网格与任务之间的距离 线性加权确定优先级
def prec_distance(epsilon, w, d, dmax, p_rec):
    # pc = epsilon * w * d + (1 - epsilon) * (1 - w) * p_rec
    # 修改论文中的公式，防止d过大从而造成影响
    pc = epsilon * w * d/dmax + (1 - epsilon) * (1 - w) * p_rec
    return pc


# 一致性约束  加权平均的更精确的噪声计数值
def get_accnoise(B, m2, noise02, sumnoise03):
    ta = (B*m2)**2
    tb = (1-B)**2
    w1 = ta / (ta + tb)
    w2 = tb / (ta + tb)

    accnoise = w1 * noise02 + w2 * sumnoise03
    return accnoise


# 对m2×m2 个三级叶子网格进行一致性约束推理
def new_noise03(noise03, accnoise, sumnoise03, m2):
    new_noise = noise03 + (accnoise - sumnoise03) / (m2*m2)
    return new_noise


# 性能指标 计算工作者的平均旅行距离 和 通知的工作者数
def worker_avg_distance(tx, ty, gr):
    num = 0
    dis_sum = 0
    for area in gr:
        num += area.N
        for i in range(area.N):
            dis = haversine_dis(tx, ty, area.x[i], area.y[i])
            dis_sum += dis
    if num != 0:
        return dis_sum / num, num
    else:
        return -1, 0

