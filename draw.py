from matplotlib import pyplot as plt
from dataprocess import lat_min, lat_max, lng_max, lng_min


# 画分割网格
def draw_areas(alist, name):
    fig = plt.figure()

    ax = fig.add_subplot(111)
    plt.xlim(lng_min, lng_max)
    plt.ylim(lat_min, lat_max)

    for ant in alist:
        rect = ant.draw_rectangle(edgecolor='g')
        ax.add_patch(rect)

    savename = './run/' + name + '.jpg'
    plt.savefig(savename, dpi=600)

    return ax


# 打印网格信息
def show_detail(X_area, level, ok=False):
    res_noise = 0
    for xnt in X_area:
        if ok:
            xnt.output()
        res_noise += xnt.N_noise

    print("第{}级网格数：".format(level), len(X_area))
    print("加噪后总人数：", res_noise)


# 画某任务t的分配区域
def draw_gr_area(tlist,        # 所有任务的坐标点   eg: [[x1,y1], ...]
                 gr_list,      # 所有任务的gr集合（附gr_max） eg: [[gr, gr_max], ...]
                 ax,
                 save=False    # 是否保存为图片
                 ):

    k = 0
    for gr, gr_max in gr_list:
        for gnt in gr:
            rect = gnt.draw_rectangle(False)
            ax.add_patch(rect)

        rect = gr_max.draw_rectangle(edgecolor='r')
        ax.add_patch(rect)

        plt.plot(tlist[k][0], tlist[k][1], 'y-^', ms=3)
        k += 1

    if save:
        plt.savefig('./run/distribution.jpg', dpi=600)

    return ax


# 画不同隐私预算下平均旅行距离和被通知工作者人数的折线图
def draw_linegraph(x, y, name, save=False):

    plt.figure()
    plt.grid()

    if name == 'num':
        plt.plot(x, y, color='r')  # 传入x，y绘制出折线图
        plt.title('被通知工作者人数随隐私预算的变化图')
    elif name == 'dis':
        plt.plot(x, y)             # 传入x，y绘制出折线图
        plt.title('平均旅行距离随隐私预算的变化图')

    # 设置x轴刻度
    plt.xticks(x)
    if save:
        savename = './run/' + name + '.jpg'
        plt.savefig(savename, dpi=600)
