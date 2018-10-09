from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import operator
from sklearn.model_selection import train_test_split
from os import listdir


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
    knn = KNeighborsClassifier(n_neighbors= 3, algorithm= 'auto')
    knn.fit(training_mat, hw_labels)
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
        classifier_result = knn.predict(vector_under_test)
        #classifier_result = classify0(vector_under_test, training_mat, hw_labels, 3)
        print("分类返回结果为{}\t真实结果为{}".format(classifier_result, class_number))
        if classifier_result != class_number:
            error_count += 1.0

        print("总共错了%d个数据\n错误率为%f%%" % (error_count, error_count / m_test))

if __name__ == "__main__":
    handle_writing_class_test()