import numpy as np
import operator
from os import listdir


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



def img2vector(filename):
    """
    :param filename:
    :return:
    """
    # 创建一个1*1024的Numpy数组
    return_vect = np.zeros((1, 1024))
    fr = open(filename)
    # 循环读取文件的前32行
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[0, 32 * i + j] = int(line_str[j])
            #返回转换后的1*1024向量
    return return_vect


def handle_writing_class_test():
    hw_labels = []
    training_file_list = listdir('trainingDigits')
    m = len(training_file_list)
    #初始化训练的mat矩阵,(测试集)
    training_mat = np.zeros((m, 1024))
    for i in range(m):
        # 获得文件的名字
        file_name_str = training_file_list[i]
        # 获得分类的数字
        class_number = int(file_name_str.split('_')[0])
        # 将分类添加到 hw_labels
        hw_labels.append(class_number)
        # 将每一个文件的1*1024数据存储到trainingMat矩阵中
        training_mat[i, :] = img2vector("trainingDigits/{}".format(file_name_str))
    #返回testDigits目录下的文件名
    test_file_list = listdir('testDigits')
    #错误检测计数
    error_count = 0.0
    #测试数据的数量
    m_test = len(test_file_list)
    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(m_test):
        file_name_str = test_file_list[i]
        class_number = int(file_name_str.split('_')[0])
        vector_under_test = img2vector('testDigits/{}'.format(file_name_str))
        #得到预测结果
        classifier_result = classify0(vector_under_test, training_mat, hw_labels, 3)
        print("分类返回结果为{}\t真实结果为{}".format(classifier_result, class_number))
        if classifier_result != class_number:
            error_count += 1.0

        print("总共错了%d个数据\n错误率为%f%%" % (error_count, error_count / m_test))

if __name__ == "__main__":
    handle_writing_class_test()