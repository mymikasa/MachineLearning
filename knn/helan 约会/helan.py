import numpy as np
import operator
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname='/usr/share/fonts/msyh_consola.ttf', size = 14)
#plt.rcParams['font.sans-serif'] = ['YaHeiConsola']

def classify0(inX, dataSet, labels, k):
    """

    :param inX:用于分类的数据（测试数据）
    :param dataSet:用于训练的数据
    :param labels:分类标签
    :param k:KNN算法的参数，选择距离最小的k个点
    :return: 返回分类结果
    """
    dataSetSize = dataSet.shape[0]
    # 在列向量方向上重复inX 1次， 行向量方向上重复inX共dataSetSize次
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    # sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis= 1)
    distances = sqDistances ** 0.5                  ## 以上四行是计算欧氏距离
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        #取出前k个元素的类别
        voteIlabel = labels[sortedDistIndicies[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1) , reverse =True)
    # 返回次数最多的类别， 即预测的类别
    return sortedClassCount[0][0]

def file2matrix(filename):
    """

    :param filename:文件名
    :return:
    return_mat 特征矩阵
    class_label_vector  标签向量
    """
    fr = open(filename)
    # 按行读取文件内容
    array_of_lines = fr.readlines()
    # 得到文件的行数
    number_of_lines = len(array_of_lines)
    # 新建一个numpy矩阵，并且此矩阵为number_of_lines行，3列并且每一个元素值都为0
    # 用于后续使用
    return_mat = np.zeros((number_of_lines, 3))
    # 标签向量
    # 后续使用
    class_label_vector = []
    index = 0
    for line in array_of_lines:
        line = line.strip()
        list_form_line = line.split('\t')
        return_mat[index, :] = list_form_line[0:3]
        if list_form_line[-1] == 'didntLike':
            class_label_vector.append(1)
        elif list_form_line[-1] == 'smallDoses':
            class_label_vector.append(2)
        elif list_form_line[-1] == 'largeDoses':
            class_label_vector.append(3)
        index += 1
    return return_mat, class_label_vector


def showdatas(dating_data_mat, dating_labels):
    myfont = fm.FontProperties(fname='/usr/share/fonts/msyh_consola.ttf')
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))

    num_of_labels = len(dating_labels)
    label_colors = []
    for i in dating_labels:
        if i == 1:
            label_colors.append('black')
        elif i == 2:
            label_colors.append('orange')
        elif i == 3:
            label_colors.append('red')

    axs[0][0].scatter(x = dating_data_mat[:, 0], y = dating_data_mat[:, 1], color = label_colors, s = 15, alpha = .5)
    axs0_title_text = axs[0][0].set_title('每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', FontProperties=myfont)
    axs0_xlabel_text = axs[0][0].set_xlabel('每年获得的飞行常客里程数',FontProperties=myfont)
    axs0_ylabel_text = axs[0][0].set_ylabel('玩视频游戏所消耗时间占', FontProperties=myfont)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    #
    # 设置标题,x轴label,y轴label
    axs[0][1].scatter(x=dating_data_mat[:, 0], y=dating_data_mat[:, 2], color=label_colors, s=15, alpha=.5)
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数', FontProperties=myfont)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=myfont)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=myfont)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=dating_data_mat[:, 1], y=dating_data_mat[:, 2], color=label_colors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数', FontProperties=myfont)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', FontProperties=myfont)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=myfont)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')

    # 设置图例

    didntLike = mlines.Line2D([], [], color = 'black', marker= '.',
                              markersize=6, label = 'didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                      markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                               markersize=6, label='largeDoses')

    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # 显示图片
    plt.show()

def autoNorm(data_set):
    """
    归一化数据
    因为在数据在不同的特征下取值是不同的，所以对所有的数据进行归一化处理， 即是所有的数据都拟合在0，1之间
    采用的归一化的方法是：
    1. 找到每个特征之间的最大值与最小值
    2. 求出二者之间的距离
    3. 每个特征值减去当前特征的最小值然后除以之前的范围值，最终得到的数值就是归一化的数值
    :param data_set:
    :return:
    """
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)

    #范围
    ranges = max_vals - min_vals

    norm_data_set = np.zeros(np.shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - np.tile(min_vals, (m, 1))
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))
    return norm_data_set, ranges, min_vals

def dating_class_Test():
    #打开的文件名
    filename = "datingTestSet.txt"
    dating_data_mat, dating_labels = file2matrix(filename)
    # 取所有数据的百分之十
    ho_ratio = 0.10
    # 数据归一化
    norm_mat, ranges, min_vals = autoNorm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ho_ratio)
    #
    errorCount = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :],
                                    dating_labels[num_test_vecs:m], 4)
        print("分类结果:%d\t真实类别:%d" % (classifier_result, dating_labels[i]))

        if classifier_result != dating_labels[i]:
            errorCount += 1.0
    print("错误率:%f%%" % (errorCount / float(num_test_vecs) * 100))


if __name__ == "__main__":
    # filename = 'datingTestSet.txt'
    # dating_data_mat, dating_labels = file2matrix(filename)
    # # showdatas(dating_data_mat, dating_labels)
    # normDataSet, ranges , min_vals = autoNorm(dating_data_mat)
    # print(normDataSet)
    # print(ranges)
    # print(min_vals)
    dating_class_Test()
