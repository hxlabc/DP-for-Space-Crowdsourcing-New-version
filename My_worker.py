import numpy as np
from matplotlib import pyplot as plt

from utils import get_distance, area_receive_rate, worker_rec_rate_nonlinear, haversine_dis, worker_rec_rate_linear


class Area:

    def __init__(self, pos_up, pos_down, level):
        self.level = level          # 属于第几级网格
        self.No = -1                # 标号（从1开始标号）
        self.pos_up = pos_up        # 上坐标
        self.pos_down = pos_down    # 下坐标
        self.x = []                 # 区域中工作者的横坐标
        self.y = []                 # 区域中工作者的纵坐标
        self.N = 0                  # 工作者的真实人数
        self.N_noise = 0            # 加噪后的人数
        self.include_parts = []     # 该一级网格所包含的二级网格的标号
        self.neighbors = []         # 记录该网格的相邻网格

    # 判断坐标点是否在该网格区域内
    def is_inArea(self, pos_x, pos_y):
        return self.pos_up[0] < pos_x <= self.pos_down[0] and self.pos_down[1] <= pos_y < self.pos_up[1]

    # 统计网格中的工作者
    def divide(self, xlist, ylist):
        xl = []
        yl = []
        #########################
        # 存放剩下的点
        x_rest = []
        y_rest = []
        #########################
        for i in range(len(xlist)):
            if self.is_inArea(xlist[i], ylist[i]):
                xl.append(xlist[i])
                yl.append(ylist[i])
            else:
                x_rest.append(xlist[i])
                y_rest.append(ylist[i])

        self.x = xl         # 区域中工作者的横坐标
        self.y = yl         # 区域中工作者的纵坐标
        self.N = len(xl)    # 工作者的真实人数

        return x_rest, y_rest

    # 添加拉普拉斯噪声
    def add_noise(self, epsilon, sensitivity, num):
         self.N_noise = self.N + np.random.laplace(loc=0, scale=sensitivity/epsilon) / num

    # 判断区域是否相邻
    def is_neighbor(self, pos_up, pos_down):
        x1, y1 = self.pos_up
        x2, y2 = self.pos_down
        x3, y3 = pos_up
        x4, y4 = pos_down

        # 加上0.00000001的判断是为了防止小数点的误差导致其不相等
        op1 = np.abs(x1 - x4) <= 0.00000001    # x1 == x4
        op2 = np.abs(x2 - x3) <= 0.00000001    # x2 == x3
        op3 = np.abs(y1 - y4) <= 0.00000001    # y1 == y4
        op4 = np.abs(y2 - y3) <= 0.00000001    # y2 == y3

        op_a = y2 <= y3 <= y1 or y2 <= y4 <= y1 or y4 <= y1 <= y3 or y4 <= y2 <= y3
        op_b = x1 <= x3 <= x2 or x1 <= x4 <= x2 or x3 <= x1 <= x4 or x3 <= x2 <= x4

        # 四大条件，满足其中之一即可相邻
        condition1 = op1 and op_a
        condition2 = op2 and op_a
        condition3 = op3 and op_b
        condition4 = op4 and op_b

        if condition1 or condition2 or condition3 or condition4:
            return True
        else:
            return False

    # 加入邻居
    def add_neighbor(self, neibor):
        self.neighbors.append(neibor)

    # 计算该网格对于某个任务t的任务接受率
    def calculate_p_rec(self, tx, ty, dmax, pmax, is_linear=False):
        n = len(self.x)

        # 计算该网格的中心点
        x_center = (self.pos_up[0] + self.pos_down[0]) / 2
        y_center = (self.pos_up[1] + self.pos_down[1]) / 2

        # d = get_distance(x_center, y_center, tx, ty)
        d = haversine_dis(x_center, y_center, tx, ty)    # 两个经纬度坐标之间的距离，单位是米

        # 网格工作者的任务接受率
        if is_linear:
            p = worker_rec_rate_linear(pmax, d, dmax)    # 线性
        else:
            p = worker_rec_rate_nonlinear(pmax, d, dmax)    # 非线性

        p_rec = area_receive_rate(n, p)        # 计算网格工作率

        return p, p_rec, d

    # 计算网格面积
    def get_square(self):
        dx = self.pos_down[0] - self.pos_up[0]
        dy = self.pos_up[1] - self.pos_down[1]
        square = dx * dy
        return square

    # 计算两个不完全重叠网格的相交网格
    def get_intersect_area(self, x_area):       # x_area为GR_max
        # 新顶点的坐标
        intersect_area = self.__copy__()

        x1, y1 = self.pos_up
        x2, y2 = self.pos_down
        x3, y3 = x_area.pos_up
        x4, y4 = x_area.pos_down

        # 不相交的条件：四个点都不在该矩形内
        op1 = self.is_inArea(x3, y3) is False
        op2 = self.is_inArea(x4, y3) is False
        op3 = self.is_inArea(x3, y4) is False
        op4 = self.is_inArea(x4, y4) is False

        # 防止出现大包小的情况，加上次条件
        op5 = x_area.is_inArea(x2, y2) is False

        # 如果两个网格不相交返回None
        if op1 and op2 and op3 and op4:
            if op5:      # 小网格右下方的点在大网格内
                return None

        # 坐标点排序
        xlist = sorted([x1, x2, x3, x4])
        ylist = sorted([y1, y2, y3, y4])

        # 中间两个即为相交的坐标点
        intersect_area.pos_up = [xlist[1], ylist[2]]
        intersect_area.pos_down = [xlist[2], ylist[1]]

        # 更新划分区域内的坐标点及人数和加噪后的人数
        intersect_area.x = []
        intersect_area.y = []
        for i in range(self.N):
            if intersect_area.is_inArea(self.x[i], self.y[i]):
                intersect_area.x.append(self.x[i])
                intersect_area.y.append(self.y[i])

        intersect_area.N = len(intersect_area.x)
        if self.N != 0:
            intersect_area.N_noise = (intersect_area.N / self.N) * self.N_noise

        return intersect_area

    # 画矩形
    def draw_rectangle(self, is_hollow=True, edgecolor='black', color='b'):
        x_new = self.pos_up[0]
        y_new = self.pos_down[1]
        dx = self.pos_down[0] - self.pos_up[0]
        dy = self.pos_up[1] - self.pos_down[1]

        if is_hollow:
            # 空心矩形
            rect = plt.Rectangle((x_new, y_new), dx, dy, edgecolor=edgecolor, facecolor='none')
        else:
            # 实心矩形
            rect = plt.Rectangle((x_new, y_new), dx, dy, color=color, alpha=.5)

        return rect

    # 复制网格并返回一个新的Area对象
    def __copy__(self):
        copy_area = Area(self.pos_up, self.pos_down, self.level)
        copy_area.No = self.No
        copy_area.x = self.x
        copy_area.y = self.y
        copy_area.N = self.N
        copy_area.N_noise = self.N_noise
        copy_area.include_parts = self.include_parts
        copy_area.neighbors = self.neighbors

        return copy_area

    # 打印信息
    def output(self):
        print("**************************")
        print("网格级数：", self.level)
        print("网格编号：", self.No)
        print("工作者的真实人数：", self.N)
        print("加噪后的人数：", self.N_noise)

        # print("上坐标：", self.pos_up)
        # print("下坐标：", self.pos_down)

        if self.level == 2:
            print("该网格所包含有效的三级网格数：", len(self.include_parts))
            # print("所包含有效的二级网格：", self.include_parts)

        if self.level == 3:
            print("邻居数：", len(self.neighbors))
            print("邻居：", self.neighbors)


