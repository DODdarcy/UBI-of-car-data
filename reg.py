from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.externals.six import StringIO
import random
from sklearn.model_selection import train_test_split

# 画柱状图
def zhu(impotances,feature_label):
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
    plt.yticks(np.arange(-1000, 1000000000, 20000))
    # 添加图例
    plt.legend(loc="upper right")
    plt.show()

train = np.array(pd.read_excel("第三部分类内研究.xlsx"))#.reshape(8506,1)
x = train[:,:15]
y = train[:,15]
# index = [i for i in range(len(x))]
# random.shuffle(index)
# x = x[index]
# y = y[index]
# 提取三个类别
type1 = []
type2 = []
type3 = []
for i in range(len(y)):
    if y[i] == 0:
        type1.append(x[i])
    elif y[i] == 1:
        type2.append(x[i])
    else:
        type3.append(x[i])
#三个类别分别进行分析回归
type1 = np.array(type1)
type2 = np.array(type2)
type3 = np.array(type3)

x1 = type1[:,:14]
y1 = type1[:,14]

x2 = type2[:,:14]
y2 = type2[:,14]

x3 = type3[:,:14]
y3 = type3[:,14]
feature_label = ['点火次数',	'熄火次数','开始里程','结束里程','行驶时间','行驶里程','行停','油耗','最高时速','平均速度','平均油耗','急加速',
                 '急减速','急转弯']
from sklearn.linear_model import LinearRegression
reg1 = LinearRegression().fit(x1, y1)
reg2 = LinearRegression().fit(x2, y2)
reg3 = LinearRegression().fit(x3, y3)
pd.DataFrame(x1).to_excel("reg_x1.xlsx")
pd.DataFrame(x2).to_excel("reg_x2.xlsx")
pd.DataFrame(x3).to_excel("reg_x3.xlsx")

#
# result = pd.concat([pd.DataFrame(feature_label),pd.DataFrame(reg1.coef_)],axis=1)
# result.to_excel("reg1.xlsx")
#
# result = pd.concat([pd.DataFrame(feature_label),pd.DataFrame(reg2.coef_)],axis=1)
# result.to_excel("reg2.xlsx")
#
# result = pd.concat([pd.DataFrame(feature_label),pd.DataFrame(reg3.coef_)],axis=1)
# result.to_excel("reg3.xlsx")
# print(reg1.coef_)
# zhu(reg1.coef_[0],feature_label)



