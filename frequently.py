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
    N = 5
    # 包含每个柱子对应值的序列
    values = impotances
    # 包含每个柱子下标的序列
    index = np.arange(N)
    # 柱子的宽度
    width = 0.35
    # 绘制柱状图, 每根柱子的颜色为紫罗兰色
    p2 = plt.bar(index, values, width, label="frequently", color="#87CEFA")
    # 设置横轴标签
    plt.xlabel('feature')
    # 设置纵轴标签
    plt.ylabel('frequently')
    # 添加标题
    plt.title('feature frequently')
    # 添加纵横轴的刻度
    plt.xticks(index, feature_label)
    plt.yticks(np.arange(0, max(list(impotances)), max(list(impotances))/10))
    # 添加图例
    plt.legend(loc="upper right")
    plt.show()

train = np.array(pd.read_excel("清洗后的数据2-第三部分.xlsx"))#.reshape(8506,1)
x = train[:,:14]
y = train[:,14]
print(y)
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

x2 = type2[:,:14]

x3 = type3[:,:14]
feature_label = ['点火次数',	'熄火次数','开始里程','结束里程','行驶时间','行驶里程','行停','油耗','最高时速','平均速度','平均油耗','急加速',
                 '急减速','急转弯']

x1 = pd.DataFrame(x1)
x1_temp = list(x1[10])
jian = (max(x1_temp)-min(x1_temp))/5
result = []
resultindex = []
for i in range(5):
    temp = 0
    for j in range(len(x1_temp)):
        if min(x1_temp)+i*jian <= x1_temp[j] and min(x1_temp)+(i+1)*jian >= x1_temp[j]:
            temp = temp+1
    resultindex.append(str(min(x1_temp)+i*jian)+"--"+str(min(x1_temp)+(i+1)*jian))
    result.append(temp)
su = sum(result)
for i in range(len(result)):
    result[i] = result[i]/su
zhu(result,resultindex)
# pd.concat([pd.DataFrame(resultindex),pd.DataFrame(result)],axis=1).to_excel("行驶里程.xlsx")
print(result)
# a = x1_temp.value_counts()
# pd.DataFrame(np.array(a)).to_excel("点火次数类别1.xlsx")
# x1_temp = pd.Series(x1[1])
# a = x1_temp.value_counts()
# pd.DataFrame(a).to_excel("熄火次数类别1.xlsx")

x2 = pd.DataFrame(x2)
x1_temp = list(x2[10])
jian = (max(x1_temp)-min(x1_temp))/5
result = []
resultindex = []
for i in range(5):
    temp = 0
    for j in range(len(x1_temp)):
        if min(x1_temp)+i*jian <= x1_temp[j] and min(x1_temp)+(i+1)*jian >= x1_temp[j]:
            temp = temp+1
    resultindex.append(str(min(x1_temp)+i*jian)+"--"+str(min(x1_temp)+(i+1)*jian))
    result.append(temp)
print(result )
su = sum(result)
for i in range(len(result)):
    result[i] = result[i]/su
zhu(result,resultindex)


x3 = pd.DataFrame(x3)
x1_temp = list(x3[10])
jian = (max(x1_temp)-min(x1_temp))/5
result = []
resultindex = []
for i in range(5):
    temp = 0
    for j in range(len(x1_temp)):
        if min(x1_temp)+i*jian <= x1_temp[j] and min(x1_temp)+(i+1)*jian >= x1_temp[j]:
            temp = temp+1
    resultindex.append(str(min(x1_temp)+i*jian)+"--"+str(min(x1_temp)+(i+1)*jian))
    result.append(temp)
su = sum(result)
for i in range(len(result)):
    result[i] = result[i]/su
zhu(result,resultindex)
# b = x2_temp.value_counts()
# pd.DataFrame(b).to_excel("点火次数类别2.xlsx")
# x2_temp = pd.Series(x2[1])
# b = x2_temp.value_counts()
# pd.DataFrame(b).to_excel("熄火次数类别2.xlsx")

