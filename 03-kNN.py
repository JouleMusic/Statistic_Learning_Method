#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project: 统计学习方法
# @File  : 03-kNN.py
# @Author: Codenergy
# @Github: https://github.com/JouleMusic/Statistic_Learning_Method
# @Date  : 2019/2/20 10:32
# @Software: PyCharm

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from Visualization import plot_init
# 初始化绘图参数
plot_init()


class kNN(object):
    def __init__(self, X_train, y_train, n_neighbors=3, p=2):
        """
        :param X_train:
        :param y_train:
        :param n_neighbors: 即书中的k值
        :param p: 距离度量的阶数
        """
        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train
    """  
    距离度量实现
    def dist(self, x, y):
        if len(x) == len(y) and len(x) > 1:
            sum = 0
            for i in range(len(x)):
                sum += math.pow(abs(x[i] - y[i]), self.p)
            return math.pow(sum, 1 / self.p)
        else:
            return 0
    """

    def predict(self, X):
        dist_list = np.zeros(len(self.X_train))
        predict = []
        for xi in X:
            # 计算X与当前所有样本点的距离
            for i in range(len(self.X_train)):
                dist = np.linalg.norm(xi - self.X_train[i], ord=self.p)
                dist_list[i] = dist

            # 对dist_list距离列表进行排序，取最小的self.n个
            topNindex = np.argsort(np.array(dist_list))[:self.n]

            """
            实现对列表内元素的出现的次数统计，但这里使用了python提供的Counter实现更方便
            list=[1,2,1,2,3,3,4,5,4]
            dict={}
            for key in list:
                dict[key]=dict.get(key,0)+1
            print(dict)
            """
            predict.append(Counter(self.y_train[topNindex]).most_common(1)[0][0])
        return np.array(predict)  # y_pred

    def score(self, y_pred, y_true):
        return accuracy_score(y_pred, y_true)


if __name__ == '__main__':
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    # plt.legend()
    # plt.show()

    data = np.array(df.iloc[:100, [0, 1, -1]])
    X, y = data[:, :-1], data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 在knn 算法中训练集就相当于已经存在的有标签的样本点，是作为计算距离度量，从而判断新进样本点的类别的
    clf = kNN(X_train, y_train, 3, 2)

    y_pred = clf.predict(X_test)
    acc = clf.score(y_pred, y_test)
    print("精度：", acc)
    test_point = [(6.0, 3.0)]
    print('Test Point: {}'.format(clf.predict(test_point)))

    plt.plot(test_point[0][0], test_point[0][1], 'ko', label='test_point')
    plt.legend()
    plt.show()
