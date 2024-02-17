"""
------------------------------------------------------------------------------------
Author: hxl
Create Date: 2023/07/31
Centralized Differential Privacy for Space Crowdsourcing Location Privacy Protection

Usage:
    $ python fjag_hgrc.py

    $ python fjag_hgrc.py --k 3 --epsilon 0.8 --a 0.4 --B 0.6 --thres 0.9 --dmax 2500 --pmax 0.4 --linear True


------------------------------------------------------------------------------------
"""


import argparse

import numpy as np
import pandas as pd
import mapclassify

from My_worker import Area
from dataprocess import lat_min, lat_max, lng_max, lng_min, p_bs
from draw import draw_areas, show_detail, draw_gr_area
from utils import get_sdcm, get_sdam, get_gvf, score_p, get_m1, get_m2, min_area, get_accnoise, new_noise03, \
    prec_distance, epsilon_after_sampling


# 加载数据
def data_load(file_path):
    worker = pd.read_excel(file_path)

    lng = list(worker['经度'])
    lat = list(worker['纬度'])
    datalen = len(lat)

    return datalen, lat, lng


# 采用Fisher_Jenks算法获取分割点
def Fisher_Jenks(k,     # 分类数目
                 lng,   # 经度坐标集
                 lat,   # 纬度坐标集
                 print_imf=False
                 ):

    FJ_lng = mapclassify.FisherJenks(lng, k)
    FJ_lat = mapclassify.FisherJenks(lat, k)

    # 将分类边界转化为列表
    lng_div = FJ_lng.bins.tolist()
    lat_div = FJ_lat.bins.tolist()

    if print_imf:
        # 计算gvf
        lng_gvf = get_gvf(lng, lng_div)
        lat_gvf = get_gvf(lat, lat_div)
        print('经度分割的gvf:', lng_gvf)
        print('纬度分割的gvf:', lat_gvf)


    lng_div[-1] = lng_max
    lng_div = [lng_min] + lng_div

    lat_div[-1] = lat_max
    lat_div = [lat_min] + lat_div

    # 使用numpy的bincount函数获取每个类别的计数
    lng_count = np.bincount(FJ_lng.yb)
    lat_count = np.bincount(FJ_lat.yb)

    # print('FJ_lng:', FJ_lng)
    # print('FJ_lat:', FJ_lat)

    return lng_div, lat_div, lng_count, lat_count


# 分割点集扰动算法
def disturb_point(pointlist,  # 经度或纬度点集
                  div_point,  # 分割点集
                  epsilon,    # 隐私预算
                  note        # 标记是经度还是纬度
                  ):
    m = len(div_point)
    pointlist = sorted(pointlist)
    flag = 0  # 记录遍历到的位置
    new_dic_point = []  # 新的分割点的集合

    for i in range(1, m-1):
        # 构建扰动区间 [start, end]
        start = div_point[i-1] + (div_point[i] - div_point[i-1]) / 2
        end = div_point[i] + (div_point[i+1] - div_point[i]) / 2

        V_area = []  # 某个扰动区间里点的集合
        pos = 0    # 用来记录当前扰动区间V中的索引下标
        for j in range(flag, len(pointlist)-1):
            if start <= pointlist[j] < end:
                V_area.append({pointlist[j]: pointlist[j+1]})
                if pointlist[j] == div_point[i]:
                    pos = j - flag
            if pointlist[j] >= end:
                flag = j
                break

        # 记录每个区间的分值
        scores = []

        for u in range(len(V_area)):
            rank = np.abs(u - pos)
            scores.append(rank)

        # Calculate the probability for each element, based on its score
        probabilities = [score_p(epsilon, 1, score) for score in scores]

        # Normalize the probabilties so they sum to 1
        probabilities = probabilities / np.linalg.norm(probabilities, ord=1)

        # Choose an element from V_area based on the probabilities
        Ik = np.random.choice(V_area, 1, p=probabilities)[0]   # 选中区间

        res = [x for x in Ik.keys()] + [y for y in Ik.values()]  # [起始位置，结束位置]
        new_dp = np.random.uniform(res[0], res[1])  # 选取Ik中的一个均匀随机值作为新的分割点

        new_dic_point.append(new_dp)

    # 补充首尾分割点
    if note == 'lng':
        new_dic_point.append(lng_max)
        new_dic_point = [lng_min] + new_dic_point
    else:
        new_dic_point.append(lat_max)
        new_dic_point = [lat_min] + new_dic_point

    return new_dic_point


# 一致性约束
def consistency_constraints(A_area,    # 二级网格集合
                            B_area,    # 三级网格集合
                            B          # 用于二三级网格的隐私预算分割比例
                            ):
    for area in A_area:
        count = len(area.include_parts)
        if area.N > 0 and count > 0:
            sum_noise = 0
            m2 = np.sqrt(count)
            # 对二级和三级子网格的噪声计数值进行处理，进一步提高范围计数查询的精度
            for k in area.include_parts:
                sum_noise += B_area[k].N_noise
            accnoise = get_accnoise(B, m2, area.N_noise, sum_noise)

            area.N_noise = accnoise    # 更新二级网格的噪声计数值

            # 更新三级网格的噪声计数值
            for k in area.include_parts:
                B_area[k].N_noise = new_noise03(B_area[k].N_noise, accnoise, sum_noise, m2)


# Fisher-Jenks + 扰动 + AG
def FJAG(lng,        # 经度坐标集
         lat,        # 纬度坐标集
         k,          # 分类数目
         epsilon,    # 采样后的隐私预算
         a,          # 隐私预算分割比例 —— 用于扰动分割点算法和AG
         B,          # 隐私预算分割比例 —— 用于二三级网格
         print_imf=False
         ):

    e1 = epsilon * a            # 用于生产扰动分割点
    e2 = epsilon * (1 - a)      # 用于AG算法

    # Fisher-Jenks算法
    Fj_lng, Fj_lat, _, _ = Fisher_Jenks(k, lng, lat, print_imf)

    # 是否打印信息
    if print_imf:
        # 没扰动前的一级网格分割
        ori_Area01s = []
        for i in range(1, len(Fj_lng)):
            for j in range(1, len(Fj_lat)):
                ori_Area01s.append(Area([Fj_lng[i-1], Fj_lat[j]], [Fj_lng[i], Fj_lat[j-1]], 1))
        draw_areas(ori_Area01s, 'org_lev01')      # 画未扰动前的一级网格

    # 分割点扰动算法
    dis_lng = disturb_point(lng, Fj_lng, e1/2, 'lng')
    dis_lat = disturb_point(lat, Fj_lat, e1/2, 'lat')

    # 扰动后的一级网格分割
    Area01s = []
    for i in range(1, len(dis_lng)):
        for j in range(1, len(dis_lat)):
            area01 = Area([dis_lng[i-1], dis_lat[j]], [dis_lng[i], dis_lat[j-1]], 1)
            Area01s.append(area01)
            # 将属于该区域的点放入该区域，并返回没有划分进入该区域点的集合，提高算法效率
            lng, lat = area01.divide(lng, lat)

    if print_imf:
        draw_areas(Area01s, 'lev01')      # 画一级网格

    # 开始执行AG算法
    epsilon1 = B * e2  # 第一部分隐私预算
    epsilon2 = (1 - B) * e2  # 第二部分隐私预算

    Area02s = []  # 第二级网格集合
    Area03s = []  # 第三级网格集合

    real_num1 = 0
    real_num2 = 0

    vaild_num1 = 0  # 第一层有效网格数，即网格中工作者人数大于0
    vaild_num2 = 0  # 第二层有效网格数，即网格中工作者人数大于0

    for area01 in Area01s:
        # 某一个一级网格大于0才考虑分割
        if area01.N > 0:
            xlng = area01.x         # 某一个一级网格中的经度坐标
            ylat = area01.y         # 某一个一级网格中的纬度坐标
            Num = area01.N          # 某一个一级网格中的人数

            m1 = get_m1(Num, e2)

            # 计算 m1 * m1 个方格中每个方格的长宽
            disx = (area01.pos_down[0] - area01.pos_up[0]) / m1
            disy = (area01.pos_up[1] - area01.pos_down[1]) / m1

            # 第二层网格的划分
            for i in range(m1):
                for j in range(m1):
                    x_up = area01.pos_up[0] + j * disx
                    y_up = area01.pos_down[1] + i * disy + disy
                    x_down = area01.pos_up[0] + j * disx + disx
                    y_down = area01.pos_down[1] + i * disy

                    area02 = Area([x_up, y_up], [x_down, y_down], 2)

                    # 将属于该区域的点放入该区域，并返回没有划分进入该区域点的集合，提高算法效率
                    xlng, ylat = area02.divide(xlng, ylat)    # 更新x 和 y

                    # 判断其是否为有效网格
                    if area02.N > 0:
                        vaild_num1 += 1

                    area02.No = real_num1  # 记录该二级网格在Area02中的标号
                    real_num1 += 1
                    Area02s.append(area02)

    for area02 in Area02s:
        flag = 1
        if area02.N > 0:
            e = epsilon1 / vaild_num1  # 计算Ai分配的隐私预算
            area02.add_noise(e, 1, vaild_num1)  # 添加噪声

            # 第三层网格的划分
            if area02.N_noise > 0:  # 如果加噪后的人数少于等于0，直接跳过
                m2 = get_m2(area02.N_noise, epsilon2)
                # 计算 m2 * m2 个方格中每个方格的长宽
                disx = (area02.pos_down[0] - area02.pos_up[0]) / m2
                disy = (area02.pos_up[1] - area02.pos_down[1]) / m2
                xlng2 = area02.x
                ylat2 = area02.y

                flag = 0  # 用来标记未被划分的二级网格

                for i in range(m2):
                    for j in range(m2):
                        x_up = area02.pos_up[0] + j * disx
                        y_up = area02.pos_down[1] + i * disy + disy
                        x_down = area02.pos_up[0] + j * disx + disx
                        y_down = area02.pos_down[1] + i * disy

                        area03 = Area([x_up, y_up], [x_down, y_down], 3)
                        # 将属于该区域的点放入该区域，并返回没有划分进入该区域点的集合，提高算法效率
                        xlng2, ylat2 = area03.divide(xlng2, ylat2)  # 更新x 和 y

                        # 判断其是否为有效网格
                        if area03.N > 0:
                            vaild_num2 += 1

                        area03.No = real_num2  # 记录该二级网格在B_area中的标号
                        area02.include_parts.append(area03.No)  # 将该二级网格的标号归类到该一级网格中
                        real_num2 += 1
                        Area03s.append(area03)

        # 将未被划分的二级网格列入三级网格，防止find_pointarea返回空
        if flag:
            temp = area02.__copy__()
            temp.level = 2
            temp.No = real_num2
            area02.include_parts.append(temp.No)
            Area03s.append(temp)
            real_num2 += 1

    for area03 in Area03s:
        if area03.N > 0:
            e2 = epsilon2 / vaild_num2           # 计算Bi分配的隐私预算
            area03.add_noise(e2, 1, vaild_num2)  # 添加噪声

    # 对二、三级网格进行一致性约束
    consistency_constraints(Area02s, Area03s, B)

    return Area02s, Area03s


# 找相邻网格
def find_neighbor(area, X_area):
    n = len(X_area)
    for i in range(n):
        if i != area.No:
            if area.is_neighbor(X_area[i].pos_up, X_area[i].pos_down):
                area.add_neighbor(i)


# 找到该任务点所属的三级网格
def find_pointarea(pos_x, pos_y, A_area, B_area):
    for xnt in A_area:
        if xnt.is_inArea(pos_x, pos_y):
            for i in xnt.include_parts:
                if B_area[i].is_inArea(pos_x, pos_y):
                    return B_area[i]
    return None


# 判断该网格是否在GR_max内
def is_inGRmax(area, GR_max):
    x1, y1 = area.pos_up
    x2, y2 = area.pos_down
    # 一个矩形的上下顶点都在GR_max内改矩形才在GR_max内
    return GR_max.is_inArea(x1, y1) and GR_max.is_inArea(x2, y2)


# 完善分割网格中的信息
def update_imf(father_area, sub_area):
    # 更新划分区域内的坐标点及人数和加噪后的人数
    for i in range(father_area.N):
        if sub_area.is_inArea(father_area.x[i], father_area.y[i]):
            sub_area.x.append(father_area.x[i])
            sub_area.y.append(father_area.y[i])

    sub_area.N = len(sub_area.x)
    if father_area.N != 0:
        sub_area.N_noise = (sub_area.N / father_area.N) * father_area.N_noise


# 按照最小面积分割网格
def getarea_by_smin(original_area, neibor_area, S_min):
    # 新顶点的坐标
    x_up = y_up = x_down = y_down = 0

    # 首先判断 neibor_area网格在 original_area网格的哪一边
    x1, y1 = original_area.pos_up
    x2, y2 = original_area.pos_down
    x3, y3 = neibor_area.pos_up
    x4, y4 = neibor_area.pos_down

    # 分四种情况

    left = np.abs(x1 - x4) <= 0.00000001  # x1 == x4
    right = np.abs(x2 - x3) <= 0.00000001  # x2 == x3
    up = np.abs(y1 - y4) <= 0.00000001  # y1 == y4
    down = np.abs(y2 - y3) <= 0.00000001  # y2 == y3

    # neibor_area在 original_area的左边
    if left:
        dl = y3 - y4     # 左边的边长
        h = S_min / dl   # 面积除以边长得到高

        x_up = x4 - h
        y_up = y3
        x_down = x4
        y_down = y4

    # neibor_area在 original_area的右边
    elif right:
        dl = y3 - y4      # 右边的边长
        h = S_min / dl    # 面积除以边长得到高

        x_up = x3
        y_up = y3
        x_down = x3 + h
        y_down = y4

    # neibor_area在 original_area的上边
    elif up:
        dl = x4 - x3  # 左边的边长
        h = S_min / dl  # 面积除以边长得到高

        x_up = x3
        y_up = y4 + h
        x_down = x4
        y_down = y4

    # neibor_area在 original_area的下边
    elif down:
        dl = x4 - x3  # 左边的边长
        h = S_min / dl  # 面积除以边长得到高

        x_up = x3
        y_up = y3
        x_down = x4
        y_down = y3 - h

    new_area = Area([x_up, y_up], [x_down, y_down], 3)

    # 补充分割区域的信息
    update_imf(neibor_area, new_area)

    return new_area


# 任务分配算法
def HGRC(tx,                # 任务点的x坐标
         ty,                # 任务点的y坐标
         thres,             # 网格接受率阈值
         dmax,              # 最大旅行距离（单位：米）
         pmax,              # 工作者最大接受率
         A_area,            # 二级网格集合
         B_area,            # 三级网格集合
         is_linear=False    # 是否采用线性变化的工作者接受率
         ):
    GR = []         # 初始化任务广播域
    begin_area = find_pointarea(tx, ty, A_area, B_area)     # 任务t所在区域网格区域

    # GR_max 任务t为中心，边长为 2 * dmax 的正方形区域
    dm = dmax * 0.0000089932    # 将米转换为对应度数
    lim_up = [tx - dm, ty + dm]
    lim_down = [tx + dm, ty - dm]
    GR_max = Area(lim_up, lim_down, -1)

    if begin_area is not None:
        _, p_gr, _ = begin_area.calculate_p_rec(tx, ty, dmax, pmax, is_linear)

        GR.append(begin_area)

        u = 0   # 用来记录GR中当前遍历到的网格
        visited = [begin_area.No]     # 用来存已经访问过且未经过分割的网格的编号（因为GR中可能包含切割后的网格）

        # 如果当前GR的接受率达不到阈值thres 且 任务t所在区域网格在GR_max内
        if p_gr < thres and is_inGRmax(begin_area, GR_max):
            while p_gr < thres:
                qlist = []  # 候选栈

                if u < len(GR):             # 防止越界
                    cur_area = GR[u]
                    u += 1
                else:
                    break

                # 找该网格的邻居
                if len(cur_area.neighbors) == 0:
                    find_neighbor(cur_area, B_area)

                for k in cur_area.neighbors:
                    if k not in visited:
                        visited.append(k)       # 标记已访问
                        temp_area = B_area[k]
                        if is_inGRmax(B_area[k], GR_max) is False:    # 取B_area[k]与GR_max的相交网格
                            temp_area = B_area[k].get_intersect_area(GR_max)

                        if temp_area is not None:     # 防止B_area[k]网格与GR_max不相交
                            p, p_rec, d = temp_area.calculate_p_rec(tx, ty, dmax, pmax, is_linear)
                            if p_rec > 0:    # 网格接受率大于0才放进qlist
                                pc = prec_distance(0.3, 0.5, d, dmax, p_rec)     # 隐私预算先设成0.3 (奇奇怪怪的)
                                qlist.append([temp_area, pc, p_rec, p])

                # 排序，取加权平均pc的值最高的先遍历
                qlist = sorted(qlist, key=lambda x: x[1], reverse=True)

                # area_i 为网格    prec_i 为该网格对应的接受率   p_i 为该网格的工作者接受率
                for area_i, _, prec_i, p_i in qlist:
                    if prec_i == 0:    # 如果网格接受率为0则不分配，直接跳过
                        continue

                    pgr_old = p_gr          # 记录更新前的变量
                    p_gr = 1 - (1 - prec_i) * (1 - p_gr)    # 更新GR的接受率

                    if p_gr < thres:
                        GR.append(area_i)
                    else:                   # 如果更新后的p_gr超过阈值，则划定最小面积区域
                        square = area_i.get_square()
                        S_min = min_area(p_i, pgr_old, thres, area_i.N, square)    # 计算最小面积

                        if S_min != -1:    # 如果该最小面积存在
                            small_area = getarea_by_smin(cur_area, area_i, S_min)
                            GR.append(small_area)
                            break

        # 如果当前GR的接受率已超过阈值thres 且 任务t所在区域网格在GR_max内
        elif p_gr > thres and is_inGRmax(begin_area, GR_max):
            p, _, _ = begin_area.calculate_p_rec(tx, ty, dmax, pmax, is_linear)
            square = begin_area.get_square()
            S_min = min_area(p, 0, thres, begin_area.N, square)  # 计算最小面积

            # 构造以（tx, ty) 为中心，面积大小为S_min的正方形区域
            bian = np.sqrt(S_min) / 2
            small_area = Area([tx - bian, ty + bian], [tx + bian, ty - bian], 3)
            GR[0] = small_area
            # 补充分割区域的信息
            update_imf(begin_area, small_area)

        # 该区域超过最大广播域GR_max时，则其与GR_max的相交网格作为GR
        else:
            intersect_area = begin_area.get_intersect_area(GR_max)
            if intersect_area is not None:         # 防止出现None
                GR[0] = intersect_area

        return GR, GR_max


# 输入参数
def parse_opt():
    parser = argparse.ArgumentParser(description="Centralized Differential Privacy")
    parser.add_argument('--k', type=int, default=3, help='Number of classifications')
    parser.add_argument('--epsilon', type=float, default=1, help='Privacy budget')
    parser.add_argument('--a', type=float, default=0.3, help='Budget factor1')
    parser.add_argument('--B', type=float, default=0.5, help='Budget factor2')
    parser.add_argument('--thres', type=float, default=0.95, help='threshold')
    parser.add_argument('--dmax', type=float, default=1600, help='Maximum distance')
    parser.add_argument('--pmax', type=float, default=0.3, help='Maximum acceptance rate')
    parser.add_argument('--linear', default=False, help='Whether to use linear')
    parser.add_argument('--source', type=str, default='./data/Geolife.xls', help='Data path')
    parser.add_argument('--save', default=True, help='Whether to save the picture')

    opt = parser.parse_args()
    parser.print_help()
    print(opt)

    return opt


# 运行程序
def run(opt):
    k = opt.k
    epsilon = opt.epsilon  # 总共的隐私预算
    a = opt.a  # 预算因子1
    B = opt.B  # 预算因子2
    thres = opt.thres  # GR接受率阈值
    dmax = opt.dmax  # 工作者最大旅行距离
    pmax = opt.pmax  # 工作者最大接受率
    is_linear = opt.linear   # 是否采用线性函数
    file_path = opt.source  # 数据的路径
    is_Save = opt.save  # 是否保存结果（图片）

    # 计算采样后的隐私预算
    n_epsilon = epsilon_after_sampling(epsilon, p_bs)
    print(n_epsilon)

    datalen, lat, lng = data_load(file_path)
    alist, blist = FJAG(lng, lat, k, n_epsilon, a, B, True)
    # show_detail(alist, 2, True)
    draw_areas(alist, 'lev02')      # 画二级网格
    ax = draw_areas(blist, 'lev03')       # 画三级网格

    # 随机的任务地点集合
    tlist = [[np.random.uniform(lng_min, lng_max), np.random.uniform(lat_min, lat_max)] for _ in range(10)]
    gr_list = []
    for tl in tlist:
        gr, gr_max = HGRC(tl[0], tl[1], thres, dmax, pmax, alist, blist, is_linear)
        gr_list.append([gr, gr_max])

    draw_gr_area(tlist, gr_list, ax, is_Save)


if __name__ == '__main__':
    opt = parse_opt()
    run(opt)
