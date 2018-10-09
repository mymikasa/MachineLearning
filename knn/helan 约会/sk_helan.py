from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# sklearn 提供的归一化方法
from sklearn import preprocessing
# from sklearn.model_selection import cross_val_score
import numpy as np


def filematrix(filename):
    """

    :param filename:
    :return:
    return_mat 特征矩阵
    class_label_vector  标签向量
    """
    fr = open(filename)
    array_of_lines = fr.readlines()
    number_of_lines = len(array_of_lines)
    return_mat = np.zeros((number_of_lines, 3))

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


def test(return_mat, class_label_vector):
    """

    :param return_mat:
    :param class_label_vector:
    :return:

    结果：
    使用划分数据集进行训练的正确率为： 76.8%
    """
    X_train, X_test, y_train, y_test = train_test_split(return_mat, class_label_vector)
    knn = KNeighborsClassifier(n_neighbors= 3, algorithm= 'auto')
    # min_max_scaler = preprocessing.MinMaxScaler()
    # X_train_minmax = min_max_scaler.fit_transform()
    # 创建一个knn分类
    knn.fit(X_train, y_train)
    # 使用测试集进行训练
    y_predicted = knn.predict(X_test)
    # 计算正确率
    accuracy = np.mean(y_test == y_predicted) * 100
    # 打印
    print('使用划分数据集进行训练的正确率为：{0: .1f}%'.format(accuracy))


def test_autonorm(return_mat, class_label_vector):
    X_train, X_test, y_train, y_test = train_test_split(return_mat, class_label_vector)
    # 创建一个knn分类
    knn = KNeighborsClassifier(n_neighbors=3, algorithm='auto')
    # 归一化处理
    min_max_scaler = preprocessing.MinMaxScaler()

    X_train_minmax = min_max_scaler.fit_transform(X_train)
    print(min)
    X_test_minmax = min_max_scaler.transform(X_test)

    knn.fit(X_train_minmax, y_train)

    y_predicted = knn.predict(X_test_minmax)
    # 计算正确率
    accuracy = np.mean(y_test == y_predicted) * 100
    # 打印
    print('使用划分数据集进行训练的正确率为：{0: .1f}%'.format(accuracy))
    return knn, min_max_scaler

# def val_score(return_mat, class_label_vector):
#     """
#     使用sklearn中的交叉验证计算正确率
#     :param dataSet:
#     :param labels:
#     :return:
#     """
#     knn = KNeighborsClassifier()
#     scores = cross_val_score(knn, return_mat, class_label_vector, scoring='accuracy')
#     average_accuracy = np.mean(scores) * 100
#     print('使用交叉验证得到的准确率为：{0: .1f}%'.format(average_accuracy))


def classify_person():
    # 输出结果
    result_list = ['讨厌','有些喜欢','非常喜欢']
    #三维特征用户输入
    precentTats = float(input("玩视频游戏所耗时间百分比:"))
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    filename = 'knn/datingTestSet.txt'

    return_mat, class_label_vector = filematrix(filename)
    in_arr = np.array([ffMiles, precentTats, iceCream])
    knn, min_max_scaler = test_autonorm(return_mat, class_label_vector)
    test_data = min_max_scaler.transform(in_arr.reshape(1, -1))
    predicted_data = knn.predict(test_data)
    print(result_list[predicted_data])

if __name__ == "__main__":
    classify_person()