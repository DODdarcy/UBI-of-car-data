#用随机森林来研究特征的重要性


import numpy as np
import pandas as pd
import random

from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy import signal
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, LeaveOneOut, LeavePOut, ShuffleSplit  # 交叉验证所需的子集划分方法
import matplotlib as mpl
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection  import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

datas =  np.array(pd.read_excel("DBSCAN聚类结果.xlsx"))
# print(pd.DataFrame(datas))
data = datas[:,:len(datas[0])-1]
label = datas[:,len(datas[0])-1]
# #打乱数据集的顺序
# index = [i for i in range(len(data))]
# random.shuffle(index)
# data = data[index]
# label = label[index]



estimator = RandomForestClassifier( random_state=6)

estimator.fit(data,label)
# pre = estimator.predict(X_test)
#维度重要性
feature_label = ['点火次数',	'熄火次数','开始里程','结束里程','行驶时间','行驶里程','行停','油耗','最高时速','平均速度','平均油耗','急加速',
                 '急减速','急转弯']
impotances = estimator.feature_importances_
result = pd.concat([pd.DataFrame(feature_label),pd.DataFrame(impotances)],axis=1)
result.to_excel("result.xlsx")
# impotances = pd.DataFrame(impotances)

# temp = ['点火次数',	'熄火次数','开始里程','结束里程','行驶时间','行驶里程','行停','油耗','最高时速','平均速度','平均油耗','急加速',
#                  '急减速','急转弯']
# impotances = pd.concat([impotances,pd.DataFrame(temp)],axis=1)
# pd.DataFrame(impotances).to_excel("维度重要性.xlsx")
# print(impotances)


#画柱状图

import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

# 创建一个点数为 8 x 6 的窗口, 并设置分辨率为 80像素/每英寸
plt.figure(figsize=(8, 6), dpi=80)
# 再创建一个规格为 1 x 1 的子图
plt.subplot(1, 1, 1)
# 柱子总数
N = 14
# 包含每个柱子对应值的序列
values = impotances
# 包含每个柱子下标的序列
index = np.arange(N)
# 柱子的宽度
width = 0.35
# 绘制柱状图, 每根柱子的颜色为紫罗兰色
p2 = plt.bar(index, values, width, label="impotance", color="#87CEFA")
# 设置横轴标签
plt.xlabel('feature')
# 设置纵轴标签
plt.ylabel('importance')
# 添加标题
plt.title('feature importance')
# 添加纵横轴的刻度
plt.xticks(index, feature_label)
plt.yticks(np.arange(0, 0.2,0.02))
# 添加图例
plt.legend(loc="upper right")
plt.show()

# 生成随机森林的拟合结果
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
feature_label = ['number of ignition','number of extinction','Start mileage','end mileage','travel time','mileage','stop',
                 'fuel consumption','maximum speed','average speed','average fuel consumption','rapid acceleration',
                'Quick deceleration','sharp turn']
# from IPython.display import Image
# from sklearn import tree
# import pydotplus
# import os
# Estimators = estimator.estimators_
# i = 0
# for index, model in enumerate(Estimators):
#     print(i)
#     i = i+1
#     filename = 'tree' + str(i) + '.png'
#     dot_data = tree.export_graphviz(model , out_file=None,
#                          feature_names=feature_label,
#                          filled=True, rounded=True,
#                          special_characters=True)
#     graph = pydotplus.graph_from_dot_data(dot_data)
#     # 使用ipython的终端jupyter notebook显示。
#     Image(graph.create_png())
#     graph.write_png(filename)

