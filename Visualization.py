#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project: 宝钢项目整理
# @File  : Visualization.py
# @Author: Codenergy
# @Github: https://github.com/JouleMusic/
# @Date  : 2019/2/24 18:55
# @Software: PyCharm


def plot_init():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings(action='once')
    # 如果要显示中文字体，则在此处设为：SimHei
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    large = 22
    med = 16
    small = 12
    params = {'axes.titlesize': large,
              'legend.fontsize': med,
              'figure.figsize': (12, 6),
              'axes.labelsize': med,
              'axes.titlesize': med,
              'xtick.labelsize': med,
              'ytick.labelsize': med,
              'figure.titlesize': large}
    plt.rcParams.update(params)
    plt.style.use('seaborn-whitegrid')
    # sns.set_style("white")

    # Version
    print(mpl.__version__)  # > 3.0.0
    print(sns.__version__)  # > 0.9.0


if __name__ == '__main__':
    plot_init()
