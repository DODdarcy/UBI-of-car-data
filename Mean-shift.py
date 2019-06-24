from sklearn.cluster import KMeans
import numpy as np
import pylab as pl
import pandas as pd
from sklearn.decomposition import PCA

def normal_X(X):
    """
    :param X:二维矩阵，N*D，N个D维的数据
    :return: 将X归一化的结果
    """
    N, D = X.shape
    for i in range(N):
        temp = np.sum(np.multiply(X[i], X[i]))
        X[i] /= np.sqrt(temp)
    return X
def normal_W(W):
    """
    :param W:二维矩阵，D*(n*m)，D个n*m维的数据
    :return: 将W归一化的结果
    """
    for i in range(W.shape[1]):
        temp = np.sum(np.multiply(W[:,i], W[:,i]))
        W[:, i] /= np.sqrt(temp)
    return W

#画图
def draw(C):
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    for i in range(len(C)):
        coo_X = []    #x坐标列表
        coo_Y = []    #y坐标列表
        for j in range(len(C[i])):
            coo_X.append(C[i][j][0])
            coo_Y.append(C[i][j][1])
        pl.scatter(coo_X, coo_Y, marker='x', color=colValue[i%len(colValue)], label=i)
    pl.title("K-Means")
    pl.legend(loc='upper right')
    pl.show()

#用的数据集
dataxl_true = pd.DataFrame(np.array(pd.read_excel("清洗后的数据2.xlsx")))
# print(max( dataxl_true[0]))
for i in range(dataxl_true.columns.size):
    dataxl_true[i] = dataxl_true[i].apply(lambda x: (x - min(dataxl_true[i])) / (max(dataxl_true[i]) - min(dataxl_true[i])))
# dataxl_true.to_excel("归一化后的数据.xlsx")
    # dataxl_true[i] = (dataxl_true[i]-min(dataxl_true[i])/(max(dataxl_true[i])-min(dataxl_true[i])))
# print(dataxl_true)
# data_yz = pd.DataFrame(np.array(pd.read_csv("test.csv",index_col=0)))
# dataxl_true = pd.concat([data_xl[0],data_xl[1],data_xl[2],data_xl[3]],axis=1)
# dataxl_true_result = data_xl[4]
# datayz_true = pd.concat([data_yz[0],data_yz[1],data_yz[2],data_yz[3]],axis=1)
# datayz_true_result = list(data_yz[4])

# dataxl_true_guiyi = normal_X(dataxl_true)
# dataset = dataxl_true
# dataset_old = dataxl_true_guiyi

pca = PCA(n_components=2)
dataset = pca.fit_transform(dataxl_true)
dataxl_true = np.array(dataxl_true)
dataset = np.array(dataset)
# pd.DataFrame(dataset).to_excel("毕业聚类数据.xlsx")
# print(dataset)
# dataset_old = dataset.copy()

#对数据进行归一化
from sklearn.cluster import MeanShift, estimate_bandwidth
# dataset = normal_X(dataset)
##带宽，也就是以某个点为核心时的搜索半径
bandwidth = estimate_bandwidth(dataxl_true, quantile=0.1, n_samples=200)
##设置均值偏移函数
kmeans = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(dataxl_true)
# kmeans = KMeans(n_clusters=4).fit(dataxl_true)


res = np.array(kmeans.labels_)

# pd.concat([pd.DataFrame(dataxl_true),pd.DataFrame(res)],axis=1).to_excel("k-means_clustering.xlsx")
# 计算轮廓系数
from sklearn import metrics
score = metrics.silhouette_score(dataxl_true,res,metric='euclidean')
print(score)

colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
# print(dataset)
# print(len(res))
for i in range(len(res)):
    pl.scatter(dataset[i][0], dataset[i][1], marker='x', color=colValue[res[i]],s=60)

pl.title("Mean-shift")
pl.legend(loc='upper right')
pl.show()
