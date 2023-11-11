import numpy as np
import numpy.linalg as lin

def sigmoid(x):
    return 1/(1+np.exp(-x))

def Z(x,w):
    if(len(x)+1 != len(w)):
        print("X and W dimension mismatch on Z",len(x),len(w))
        return 0
    sum = w[0]
    for dim in range(len(x)):
        sum = sum + w[dim+1]*x[dim]
    return np.exp(sum)

def sum_list(w):
    sum = 0
    for i in range(len(w)):
        sum = sum + w[i]
    return sum

def list_product(x,w):
    if(len(x) != len(w)):
        print("X and W dimension mismatch on list_product",len(x),len(w))
        return 0
    value = 0
    for dim in range(len(x)):
        value = value + x[dim]*w[dim]
    return value

def sign(value):
    if(value>0):
        return 1
    return 

def new_coordinate(x,eigenvectors,withbias=True):
    bias = [5.50236513, 5.32695258]
    if(withbias == False):
        bias = [0, 0]
    x = np.array(x)
    eigenvector = np.array(eigenvectors).T
    return np.dot(np.dot(lin.inv(np.dot(eigenvector.T,eigenvector)),eigenvector.T),x) - bias

def func(point,normal):
    if(len(point)!=2 or len(normal)!=2):
        print("func need 2 dimension point argument",len(point),len(normal))
        return 0
    x = [-3,-2,-1,0,1,2,3,4]
    y = []
    for item in range(len(x)):
        next = -(x[item]-point[0])*normal[0]/normal[1]+point[1]
        y.append(next)
    return y