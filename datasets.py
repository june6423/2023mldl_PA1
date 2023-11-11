from sklearn import datasets
import numpy as np
import random
from logistic import *
from perceptron import *
from util import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

iris = datasets.load_iris()
x = iris.data
y = iris.target
index = list(range(len(x)))
sample = random.sample(index,int(len(x)/4))

tr_x = list()
tr_y = list()
val_x = list()
val_y = list()
x_2d = list()

temp_tr = list()
temp_val = list()

for item in range(len(x)):
    #new = x[item][0:2]
    new = x[item]
    x_2d.append(new)
    #if(y[item] == 2):
    #    continue
    if(item in sample):
        temp_val.append((new,y[item]))
    else:
        temp_tr.append((new,y[item]))

random.shuffle(temp_tr)
random.shuffle(temp_val)

for index in range(len(temp_tr)):
    tr_x.append(temp_tr[index][0])
    tr_y.append(temp_tr[index][1])
    
for index in range(len(temp_val)):
    val_x.append(temp_val[index][0])
    val_y.append(temp_val[index][1])

""" #maxIter = [1000,5000,10000,50000,100000,500000]
maxIter = 100000
eta = 0.05
result = []
threshold = np.array(np.arange(0,100),dtype='float32')
#Train_GA MaxIter 1000, eta 0.0005
#Train_SGA MaxIter 100000, eta 0.002
#Train_reg_SGA MaxIter 20000, eta 0.005
#Preceptron threshold = 0.05

for item in range(len(threshold)):
    threshold[item] = np.divide(threshold[item],1000.0)
    model = Perceptron2D(threshold=threshold[item])
    model.train(tr_x,tr_y)
    y_hat = model.predict(val_x)
    acc = computeClassificationAcc2D(val_y,y_hat)
    #print(val_y,y_hat)
    print("threshold",threshold[item],"ACC",acc)
    result.append(acc)
plt.plot(threshold,result)
plt.show() """

""" 
total_hat = model.predict(x_2d)
weight = model.weight()
print("weight",weight)

x_axes = [4,5,6,7]
y_axes = []
for item in range(len(x_axes)):
    y_axes.append(-1*(weight[0]/weight[1]*x_axes[item]))
plt.plot(x_axes,y_axes,c="black")

"""  """
X = np.array(x)
pca = PCA(n_components=2)
printcipalComponents = pca.fit_transform(X)
#print(pca.components_)
#print(pca.explained_variance_ratio_) 
"""
""" 
for item in range(len(total_hat)):
    if(y[item] == 2):
        continue
    if(total_hat[item]==0):
        plt.scatter(x[item][0],x[item][1],c='red',edgecolors = 'none')
        #plt.scatter(printcipalComponents[item][0],printcipalComponents[item][1],c='red',edgecolors = 'none')
    if(total_hat[item]==1):
        plt.scatter(x[item][0],x[item][1],c='blue',edgecolors = 'none')
        #plt.scatter(printcipalComponents[item][0],printcipalComponents[item][1],c='blue',edgecolors = 'none')
    #if(total_hat[item]==2):
        #plt.scatter(x[item][0],x[item][1],c='yellow',edgecolors = 'none')
        #plt.scatter(printcipalComponents[item][0],printcipalComponents[item][1],c='yellow',edgecolors = 'none')
plt.show()  """