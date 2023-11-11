from util import *
import numpy as np
import random

from datasets import *

class Perceptron():
    def __init__(self,threshold):
        self.threshold = threshold
        self.maxIter = 10000
        self.w = [[0 for i in range(4)] for j in range(3)]
        #print(len(self.w),len(self.w[0]),self.w[2][3])
        
    def train(self,x,y):
        for iter in range(self.maxIter):
            #if(iter % 1000 == 0):
            #    print("iter ",iter)
            correct = 0
            check = random.randint(1,500)
            
            for i in range(check):
                item  = random.randint(0,len(x)-1)
                prediction = self.predict_single(x[item])
                if(y[item] != prediction):
                    for dim in range(len(x[0])):
                        self.w[prediction][dim] = self.w[prediction][dim] - x[item][dim]
                        self.w[y[item]][dim] = self.w[y[item]][dim] + x[item][dim]
                else:
                    correct = correct + 1
    
    def predict_single(self,x):
        y_hat = 0
        expect = np.zeros(3)
        expect[0] = list_product(x,self.w[0])
        expect[1] = list_product(x,self.w[1])
        expect[2] = list_product(x,self.w[2])
        
        if(expect[0] > expect[1] and expect[0] > expect[2]):
            y_hat=0
        elif(expect[1] > expect[0] and expect[1] > expect[2]):
            y_hat=1
        else:
            y_hat=2
        return y_hat
    
    def predict(self,x):
        y_hat = list()
        expect = np.zeros(3)
        for item in range(len(x)):
            expect[0] = list_product(x[item],self.w[0])
            expect[1] = list_product(x[item],self.w[1])
            expect[2] = list_product(x[item],self.w[2])
            
            if(expect[0] > expect[1] and expect[0] > expect[2]):
                y_hat.append(0)
            elif(expect[1] > expect[0] and expect[1] > expect[2]):
                y_hat.append(1)
            else:
                y_hat.append(2)
        #print("Terminal Weight",self.w)
        return y_hat
    
    def weight(self):
        return self.w
    
    def discard(self):
        self.w = [[0 for i in range(4)] for j in range(3)]
        
def computeClassificationAcc(y, y_hat):
    if(len(y) != len(y_hat)):
        print("y and y_hat dimension mismatch ",len(y),len(y_hat))
        return 0
    correct = 0
    for num in range(len(y)):
        correct = correct + (y[num] == y_hat[num])
    return correct/len(y)

def computeClassificationAcc2D(y, y_hat):
    if(len(y) != len(y_hat)):
        print("y and y_hat dimension mismatch ",len(y),len(y_hat))
        return 0
    correct = 0
    for num in range(len(y)):
        correct = correct + (y[num] == y_hat[num])
    return correct/len(y)


class Perceptron2D():
    def __init__(self,threshold):
        self.threshold = threshold
        self.maxIter = 10000
        self.w = [0, 0]
        #print(len(self.w),len(self.w[0]),self.w[2][3])
        
    def train(self,x,y):
        for iter in range(self.maxIter):
            iter = iter + 1
            #if(iter % 1000 == 0):
            #    print("iter ",iter)
            correct = 0
            check = random.randint(1,500)
            
            for i in range(check):
                item  = random.randint(0,len(x)-1)
                prediction = self.predict_single(x[item])
                
                if(y[item] != prediction):
                    for dim in range(len(x[item])):
                        self.w[dim] = self.w[dim] + x[item][dim]*(2*y[item]-1)
                else:
                    correct = correct + 1
    
    def predict_single(self,x):
        y_hat = 0
        expect = list_product(x,self.w)
        if(expect > self.threshold):
            y_hat=1
        else:
            y_hat=0
        return y_hat
    
    def predict(self,x):
        y_hat = list()
        for item in range(len(x)):
            expect = list_product(x[item],self.w)
            if(expect > self.threshold):
                y_hat.append(1)
            else:
                y_hat.append(0)
        #print("Terminal Weight",self.w)
        return y_hat
    
    def weight(self):
        return self.w
    
    def discard(self):
        self.w = [0,0]
