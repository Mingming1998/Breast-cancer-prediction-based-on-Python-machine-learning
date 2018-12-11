# 导入类库
from pandas import read_csv
import pandas as pd
from sklearn import datasets
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #要注意的是一旦导入了seaborn，matplotlib的默认作图风格就会被覆盖成seaborn的格式

import numpy as np



#breast_cancer_data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',header=None
#                               ,names = ['C_D','C_T','U_C_Si','U_C_Sh','M_A','S_E_C_S'
#                                        ,'B_N','B_C','N_N','M','Class'])
breast_cancer_data = pd.read_csv('breast_data.csv',header=None
                               ,names = ['C_D','C_T','U_C_Si','U_C_Sh','M_A','S_E_C_S'
                                        ,'B_N','B_C','N_N','M','Class'])
#打印数据
print(breast_cancer_data)

#查看维度
print('查看维度：')
print(breast_cancer_data.shape)
#查看数据
print('查看数据')
breast_cancer_data.info()
breast_cancer_data.head(25)

#数据统计描述
print('数据统计描述')
print(breast_cancer_data.describe())
#数据分布情况
print('数据分布情况')
print(breast_cancer_data.groupby('Class').size())

#缺失数据处理
mean_value = breast_cancer_data[breast_cancer_data["B_N"]!="?"]["B_N"].astype(np.int).mean()
breast_cancer_data = breast_cancer_data.replace('?',mean_value)
breast_cancer_data["B_N"] = breast_cancer_data["B_N"].astype(np.int64)

#数据的可视化处理
    #单变量图表
    #箱线图
breast_cancer_data.plot(kind='box',subplots=True,layout=(3,4),sharex=False,sharey=False)
pyplot.show()
    #直方图
breast_cancer_data.hist()
pyplot.show()
#多变量的图表
    #散点矩阵图
scatter_matrix(breast_cancer_data)
pyplot.show()


#评估算法
    #分离数据集
array = breast_cancer_data.values
X = array[:,1:9]
y = array[:,10]

validation_size = 0.2
seed = 7
#train训练，validation验证确认
X_train,X_validation,y_train,y_validation = train_test_split(X,y,test_size=validation_size,random_state=seed)

    #评估算法（算法审查）
models = {}
models['LR'] = LogisticRegression()
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['NB'] = GaussianNB()
models['SVM'] = SVC()



num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds,random_state=seed)

    #评估算法（评估算法）
results = []
for name in models:
    result = cross_val_score(models[name],X_train,y_train,cv=kfold,scoring='accuracy')
    results.append(result)
    msg = '%s:%.3f(%.3f)'%(name,result.mean(),result.std())
    print(msg)
    #评估算法（图标显示）
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(models.keys())
pyplot.show()


#实施预测
#使用评估数据集评估算法
knn = KNeighborsClassifier()
knn.fit(X=X_train,y=y_train)
predictions = knn.predict(X_validation)

print('最终使用KNN算法')
print(accuracy_score(y_validation,predictions))
print(confusion_matrix(y_validation,predictions))
print(classification_report(y_validation,predictions))

