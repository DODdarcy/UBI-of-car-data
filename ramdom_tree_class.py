from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.externals.six import StringIO
import random
from sklearn.model_selection import train_test_split
import pydotplus
#类间随机森林
train = np.array(pd.read_excel("DBSCAN聚类结果.xlsx"))#.reshape(8506,1)
x = train[:,:14]
y = train[:,14]
index = [i for i in range(len(x))]
random.shuffle(index)
x = x[index]
y = y[index]
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.33, random_state=42)
from sklearn.metrics import accuracy_score
# print(y)
# print(x)
k = 11
for i in range(2,k):
    clf = RandomForestClassifier(n_estimators=i)
    clf.fit(X_train, y_train)
    # feature_name = ['Mg', 'Zn', 'Fe', 'Si	', 'Se', 'Ca', 'Cu', 'As', 'K', 'Na', 'Mn', '有效镉', '土壤镉', 'pH', 'CEC',
    #                 'SOM', '镉活性']
    # ss = "importance"+str(i+10)+".xlsx"
    # pd.concat([pd.DataFrame(feature_name),pd.DataFrame(clf.feature_importances_)],axis=1).to_excel(ss)
    print("决策树个数为："+str(i))
    pre = clf.predict(X_test)
    accu = accuracy_score(y_test,pre)
    print("对应准确率为："+str(accu))