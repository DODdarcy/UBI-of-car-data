import numpy as np
import pylab as pl
import pandas as pd
from sklearn.decomposition import PCA

class SOM(object):
    def __init__(self, X, output, iteration, batch_size):
        """
        :param X:  形状是N*D， 输入样本有N个,每个D维
        :param output: (n,m)一个元组，为输出层的形状是一个n*m的二维矩阵
        :param iteration:迭代次数
        :param batch_size:每次迭代时的样本数量
        初始化一个权值矩阵，形状为D*(n*m)，即有n*m权值向量，每个D维
        """
        self.X = X
        self.output = output
        self.iteration = iteration
        self.batch_size = batch_size
        self.W = np.random.rand(X.shape[1], output[0] * output[1])
        # print (self.W.shape)

    def GetN(self, t):
        """
        :param t:时间t, 这里用迭代次数来表示时间
        :return: 返回一个整数，表示拓扑距离，时间越大，拓扑邻域越小
        """
        a = min(self.output)
        return int(a-float(a)*t/self.iteration)

    def Geteta(self, t, n):
        """
        :param t: 时间t, 这里用迭代次数来表示时间
        :param n: 拓扑距离
        :return: 返回学习率，
        """
        return np.power(np.e, -n)/(t+2)

    def updata_W(self, X, t, winner):
        N = self.GetN(t)
        for x, i in enumerate(winner):
            to_update = self.getneighbor(i[0], N)
            for j in range(N+1):
                e = self.Geteta(t, j)
                for w in to_update[j]:
                    self.W[:, w] = np.add(self.W[:,w], e*(X[x,:] - self.W[:,w]))

    def getneighbor(self, index, N):
        """
        :param index:获胜神经元的下标
        :param N: 邻域半径
        :return ans: 返回一个集合列表，分别是不同邻域半径内需要更新的神经元坐标
        """
        a, b = self.output
        length = a*b
        def distence(index1, index2):
            i1_a, i1_b = index1 // a, index1 % b
            i2_a, i2_b = index2 // a, index2 % b
            return np.abs(i1_a - i2_a), np.abs(i1_b - i2_b)

        ans = [set() for i in range(N+1)]
        for i in range(length):
            dist_a, dist_b = distence(i, index)
            if dist_a <= N and dist_b <= N: ans[max(dist_a, dist_b)].add(i)
        return ans




    def train(self):
        """
        train_Y:训练样本与形状为batch_size*(n*m)
        winner:一个一维向量，batch_size个获胜神经元的下标
        :return:返回值是调整后的W
        """
        count = 0
        while self.iteration > count:
            train_X = self.X[np.random.choice(self.X.shape[0], self.batch_size)]
            normal_W(self.W)
            normal_X(train_X)
            train_Y = train_X.dot(self.W)
            winner = np.argmax(train_Y, axis=1).tolist()
            self.updata_W(train_X, count, winner)
            count += 1
        return self.W

    def train_result(self):
        normal_X(self.X)
        train_Y = self.X.dot(self.W)
        winner = np.argmax(train_Y, axis=1).tolist()
        # print (winner)
        return winner

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
        pl.scatter(coo_X, coo_Y, marker='x', color=colValue[i%len(colValue)], label=i,s = 50)
    pl.title("SOM")
    pl.legend(loc='upper right')
    pl.show()

#用的数据集
dataxl_true = pd.DataFrame(np.array(pd.read_excel("清洗后的数据2.xlsx")))
# print(max( dataxl_true[0]))
for i in range(dataxl_true.columns.size):
    dataxl_true[i] = dataxl_true[i].apply(lambda x: (x - min(dataxl_true[i])) / (max(dataxl_true[i]) - min(dataxl_true[i])))
    # dataxl_true[i] = (dataxl_true[i]-min(dataxl_true[i])/(max(dataxl_true[i])-min(dataxl_true[i])))
# print(dataxl_true)
# data_xl = pd.DataFrame(np.array(pd.read_csv("train.csv",index_col=0)))
# data_yz = pd.DataFrame(np.array(pd.read_csv("test.csv",index_col=0)))
# dataxl_true = pd.concat([data_xl[0],data_xl[1],data_xl[2],data_xl[3]],axis=1)
# dataxl_true_result = data_xl[4]
# datayz_true = pd.concat([data_yz[0],data_yz[1],data_yz[2],data_yz[3]],axis=1)
# datayz_true_result = list(data_yz[4])





#数据集：每三个是一组分别是西瓜的编号，密度，含糖量
# data = """
# 1,0.697,0.46,2,0.774,0.376,3,0.634,0.264,4,0.608,0.318,5,0.556,0.215,
# 6,0.403,0.237,7,0.481,0.149,8,0.437,0.211,9,0.666,0.091,10,0.243,0.267,
# 11,0.245,0.057,12,0.343,0.099,13,0.639,0.161,14,0.657,0.198,15,0.36,0.37,
# 16,0.593,0.042,17,0.719,0.103,18,0.359,0.188,19,0.339,0.241,20,0.282,0.257,
# 21,0.748,0.232,22,0.714,0.346,23,0.483,0.312,24,0.478,0.437,25,0.525,0.369,
# 26,0.751,0.489,27,0.532,0.472,28,0.473,0.376,29,0.725,0.445,30,0.446,0.459"""
#
# a = data.split(',')
# dataset = np.mat([[float(a[i]), float(a[i+1])] for i in range(1, len(a)-1, 3)])
# dataset_old = dataset.copy()
#
# print(type(dataset))
# print(dataset)
# print(dataset_old)
pca = PCA(n_components=2)
dataset = pca.fit_transform(dataxl_true)

dataset_old = dataset.copy()
print(dataxl_true)
dataxl_true = np.array(dataxl_true)
dataxl_true = np.asmatrix(dataxl_true)

som = SOM(dataxl_true, (1, 3), 10, 6)
som.train()
res = som.train_result()
# 计算轮廓系数
from sklearn import metrics
dataxl_true = np.array(dataxl_true)
score = metrics.silhouette_score(dataxl_true,res,metric='euclidean')
print(score)

# 导出聚类结果
# pd.concat([pd.DataFrame(dataxl_true),pd.DataFrame(res)],axis=1).to_excel("SOM_clustering.xlsx")

classify = {}
for i, win in enumerate(res):
    if not classify.get(win[0]):
        classify.setdefault(win[0], [i])
    else:
        classify[win[0]].append(i)

C = []#未归一化的数据分类结果
D = []#归一化的数据分类结果
for i in classify.values():
    C.append(dataset_old[i].tolist())
    D.append(dataset[i].tolist())

draw(C)
# draw(D)
