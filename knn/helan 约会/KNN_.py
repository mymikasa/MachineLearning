import numpy as np
import operator

## 构造数据集
def createDataSet():
    """
    创建数据
    :return:
    group  - 分类集
    labels - 分类标签
    """
    group = np.array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0., 0.],
        [0., 0.1],
    ])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


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
        if list_form_line[-1] == 'didnLike':
            class_label_vector.append(1)
        elif list_form_line[-1] == 'smallDoses':
            class_label_vector.append(2)
        elif list_form_line[-1] == 'largeDoses':
            class_label_vector.append(3)
        index += 1
    return return_mat, class_label_vector


if __name__ == "__main__":
    group, labels = createDataSet()
    # print(group)
    # print(labels)
    # 测试数据
    test = [1.1, 1.2]

    test_calss = classify0(test, group, labels, 3)
    print(test_calss)