# 数据集的 维度 经度 高度 日期 时间
#
#
import os
import matplotlib.pyplot as plt
import numpy as np
import xlwt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据集中选取经纬度的范围
lat_min = 39.85
lat_max = 40.1
lng_min = 116.1
lng_max = 116.6

# 采样概率
p_bs = 0.1


# 解析plt文件
def plt2list():
    lat = []  # 纬度
    lng = []  # 经度

    path = "D:\\差分隐私Different Privacy" + "\\Geolife Trajectories 1.3" + "\\Data"
    numfiles = os.scandir(path)

    # 每一个文件的绝对路径
    for num in numfiles:
        path_item = path + "\\" + num.name + "\\Trajectory"
        print(path_item)
        pltsfiles = os.scandir(path_item)
        for pltitem in pltsfiles:
            plt_path = path_item + '\\' + pltitem.name
            with open(plt_path, 'r+') as fp:
                for item in fp.readlines()[6::50]:
                    item_list = item.split(',')
                    lat_i = float(item_list[0])
                    lng_i = float(item_list[1])

                    # 选取位于北京的位置点
                    if lat_min < lat_i < lat_max and lng_min < lng_i < lng_max:
                        lat.append(lat_i)
                        lng.append(lng_i)

    datalen = len(lat)

    return datalen, lat, lng


# 伯努利采样  p为采样概率
def bernoulli_sampling(p, size, lat, lng):
    lat_new = []
    lng_new = []

    # 使用numpy的random.binomial函数进行伯努利实验
    mask = np.random.binomial(n=1, p=p, size=size)

    for i in range(size):
        if mask[i] == 1:
            lat_new.append(lat[i])
            lng_new.append(lng[i])

    datalen = len(lat_new)

    return datalen, lat_new, lng_new


# 将采样后的数据保存到exel表中
def list2exel(lat, lng):
    workbook = xlwt.Workbook(encoding='utf-8')  # 设置一个workbook，其编码是utf-8
    worksheet = workbook.add_sheet("lat_lng")  # 新增一个sheet
    worksheet.write(0, 0, label='经度')
    worksheet.write(0, 1, label='纬度')
    for i in range(len(lat)):  # 循环将a和b列表的数据插入至excel
        worksheet.write(i + 1, 0, label=lng[i])
        worksheet.write(i + 1, 1, label=lat[i])
    workbook.save("./data/Geolife.xls")  # 这里save需要特别注意，文件格式只能是xls，不能是xlsx，不然会报错


# 可视化
def visiable(lat, lng):

    plt.title("Geolife抽样数据集")
    plt.xlabel("经度")
    plt.ylabel("纬度")
    plt.scatter(lng, lat, s=2, c='#f0a732', marker='*')

    plt.savefig('./run/Geolife.jpg', dpi=600)
    plt.show()


if __name__ == '__main__':
    datalen, lat, lng = plt2list()
    datalen, lat, lng = bernoulli_sampling(p_bs, datalen, lat, lng)
    list2exel(lat, lng)
    visiable(lat, lng)
    print("采样后的数目：", datalen)


