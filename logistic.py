from util import *
import numpy as np
import random

from datasets import *

class Logistic():
    def __init__(self,maxIter, eta, scalar_lambda=0):
        self.maxIter = maxIter
        self.eta = eta
        self.w = list()
        self.scalar_lambda = scalar_lambda
        #for dim in range(2):
            #self.w.append(np.random.normal(loc=0,scale=1,size=5))
        self.w = [[0 for i in range(5)] for j in range(2)]
        # I had an ablation study about initial weight. (Zero mean gaussian and zero weight)
        # zero weight has better performance. 
        # print("Initial Weight ",self.w)
        
    def train_GA(self,x,y):
        for iter in range(self.maxIter):
            gradient = [[0 for i in range(len(self.w[0]))] for i in range(2)]
            #print("Weight on iter ",iter, self.w[0])
            for item in range(len(y)):
                if(y[item] == 0):
                    gradient[0][0] = gradient[0][0] + (1-Z(x[item],self.w[0])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1])))
                    for dim in range(1,len(self.w[0])):
                        gradient[0][dim] = gradient[0][dim] + x[item][dim-1]*(1-Z(x[item],self.w[0])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1])))
                elif(y[item] == 1):
                    gradient[1][0] = gradient[1][0] + (1-Z(x[item],self.w[1])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1])))
                    for dim in range(1,len(self.w[0])):
                        gradient[1][dim] = gradient[1][dim] + x[item][dim-1]*(1-Z(x[item],self.w[1])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1])))
                else:
                    gradient[0][0] = gradient[0][0] - Z(x[item],self.w[0])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1]))
                    gradient[1][0] = gradient[1][0] - Z(x[item],self.w[1])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1]))
                    for dim in range(1,len(self.w[0])):
                        gradient[0][dim] = gradient[0][dim] - x[item][dim-1]*Z(x[item],self.w[0])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1]))
                    for dim in range(1,len(self.w[0])):
                        gradient[1][dim] = gradient[1][dim] - x[item][dim-1]*Z(x[item],self.w[1])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1]))
            #print("Gradient on iter ",iter, gradient)
            for dim in range(len(self.w[0])):
                for i in range(2):
                    self.w[i][dim] = self.w[i][dim] + self.eta*gradient[i][dim]
                    
    def train_SGA(self,x,y):
        gradient = [[0 for i in range(len(self.w[0]))] for i in range(2)]
        for iter in range(self.maxIter):
            #print("Weight on iter ",iter, self.w)
            item = random.randint(0,len(y)-1)
            if(y[item] == 0):
                gradient[0][0] = gradient[0][0] + (1-Z(x[item],self.w[0])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1])))
                for dim in range(1,len(self.w[0])):
                    gradient[0][dim] = gradient[0][dim] + x[item][dim-1]*(1-Z(x[item],self.w[0])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1])))
            elif(y[item] == 1):
                gradient[1][0] = gradient[1][0] + (1-Z(x[item],self.w[1])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1])))
                for dim in range(1,len(self.w[0])):
                    gradient[1][dim] = gradient[1][dim] + x[item][dim-1]*(1-Z(x[item],self.w[1])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1])))
            else:
                gradient[0][0] = gradient[0][0] - Z(x[item],self.w[0])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1]))
                gradient[1][0] = gradient[1][0] - Z(x[item],self.w[1])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1]))
                for dim in range(1,len(self.w[0])):
                    gradient[0][dim] = gradient[0][dim] - x[item][dim-1]*Z(x[item],self.w[0])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1]))
                for dim in range(1,len(self.w[0])):
                    gradient[1][dim] = gradient[1][dim] - x[item][dim-1]*Z(x[item],self.w[1])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1]))
            #print("Gradient on iter ",iter, gradient)
            if(random.randint(0,10) == 1):
                for dim in range(len(self.w[0])):
                    for i in range(2):
                        self.w[i][dim] = self.w[i][dim] + self.eta*gradient[i][dim]
                gradient = [[0 for i in range(len(self.w[0]))] for i in range(2)]
                                      
    def train_reg_SGA(self,x,y):
        gradient = [[0 for i in range(len(self.w[0]))] for i in range(2)]
        for iter in range(self.maxIter):
            #print("Weight on iter ",iter, self.w)
            item = random.randint(0,len(y)-1)
            if(y[item] == 0):
                gradient[0][0] = gradient[0][0] + (1-Z(x[item],self.w[0])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1])))
                for dim in range(1,len(self.w[0])):
                    gradient[0][dim] = gradient[0][dim] + x[item][dim-1]*(1-Z(x[item],self.w[0])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1])))
            elif(y[item] == 1):
                gradient[1][0] = gradient[1][0] + (1-Z(x[item],self.w[1])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1])))
                for dim in range(1,len(self.w[0])):
                    gradient[1][dim] = gradient[1][dim] + x[item][dim-1]*(1-Z(x[item],self.w[1])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1])))
            else:
                gradient[0][0] = gradient[0][0] - Z(x[item],self.w[0])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1]))
                gradient[1][0] = gradient[1][0] - Z(x[item],self.w[1])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1]))
                for dim in range(1,len(self.w[0])):
                    gradient[0][dim] = gradient[0][dim] - x[item][dim-1]*Z(x[item],self.w[0])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1]))
                for dim in range(1,len(self.w[0])):
                    gradient[1][dim] = gradient[1][dim] - x[item][dim-1]*Z(x[item],self.w[1])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1]))
            #print("Gradient on iter ",iter, gradient)
            if(random.randint(0,10) == 1):
                for dim in range(len(self.w[0])):
                    for i in range(2):
                        self.w[i][dim] = self.w[i][dim] - self.scalar_lambda*self.w[i][dim] + self.eta*gradient[i][dim]
                gradient = [[0 for i in range(len(self.w[0]))] for i in range(2)]
                
    def predict(self,x):
        y_hat = list()
        expect = np.zeros(3)
        for item in range(len(x)):
            expect[0] = Z(x[item],self.w[0])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1]))
            expect[1] = Z(x[item],self.w[1])/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1]))
            expect[2] = 1/(1+Z(x[item],self.w[0])+Z(x[item],self.w[1]))
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
        self.w = [[0 for i in range(5)] for j in range(2)]
    
def computeClassificationAcc(y, y_hat):
    if(len(y) != len(y_hat)):
        print("y and y_hat dimension mismatch ",len(y),len(y_hat))
        return 0
    correct = 0
    total = 0
    for num in range(len(y)):
        if(y[num] == 2):
            continue
        correct = correct + (y[num] == y_hat[num])
        total = total + 1
    return correct/total

    
def computeClassificationAcc2D(y, y_hat):
    if(len(y) != len(y_hat)):
        print("y and y_hat dimension mismatch ",len(y),len(y_hat))
        return 0
    correct = 0
    for num in range(len(y)):
        correct = correct + (y[num] == y_hat[num])
    return correct/len(y)

class Logistic_2D():
    def __init__(self,maxIter, eta, scalar_lambda=0):
        self.maxIter = maxIter
        self.eta = eta
        self.scalar_lambda = scalar_lambda
        self.w = [0 , 0, 0 ]
        # I had an ablation study about initial weight. (Zero mean gaussian and zero weight)
        # zero weight has better performance. 
        # print("Initial Weight ",self.w)
        
    def train_GA(self,x,y):
        for iter in range(self.maxIter):
            gradient = [0, 0, 0]
            #print("Weight on iter ",iter, self.w)
            for item in range(len(y)):
                gradient[0] = gradient[0] +y[item]-1/(1+Z(x[item],self.w))
                for dim in range(1,len(self.w)):
                    gradient[dim] = gradient[dim] + x[item][dim-1]*(y[item]-Z(x[item],self.w)/(1+Z(x[item],self.w)))
            for dim in range(len(self.w)):
                self.w[dim] = self.w[dim] + self.eta*gradient[dim]
                    
    def train_SGA(self,x,y):
        gradient = [0, 0, 0]
        for iter in range(self.maxIter):
            #print("Weight on iter ",iter, self.w)
            item = random.randint(1,len(y)-1)
            gradient[0] = gradient[0] +y[item]-1/(1+Z(x[item],self.w))
            for dim in range(1,len(self.w)):
                gradient[dim] = gradient[dim] + x[item][dim-1]*(y[item]-Z(x[item],self.w)/(1+Z(x[item],self.w)))
            
            chance = random.randint(1,10)
            if(chance == 1):
                for dim in range(len(self.w)):
                    self.w[dim] = self.w[dim] + self.eta*gradient[dim]
                gradient = [0,0,0]
                                      
    def train_reg_SGA(self,x,y):
        gradient = [0, 0, 0]
        for iter in range(self.maxIter):
            #print("Weight on iter ",iter, self.w)
            item = random.randint(1,len(y)-1)
            gradient[0] = gradient[0] +y[item]-1/(1+Z(x[item],self.w))
            for dim in range(1,len(self.w)):
                gradient[dim] = gradient[dim] + x[item][dim-1]*(y[item]-Z(x[item],self.w)/(1+Z(x[item],self.w)))
            
            chance = random.randint(1,10)
            if(chance == 1):
                for dim in range(len(self.w)):
                    self.w[dim] = self.w[dim] + self.eta*gradient[dim] - self.scalar_lambda*gradient[dim]
                gradient = [0,0,0]
                
    def predict(self,x):
        y_hat = list()
        expect = np.zeros(2)
        for item in range(len(x)):
            expect[0] = 1/(1+Z(x[item],self.w))
            expect[1] = Z(x[item],self.w)/(1+Z(x[item],self.w))
            if(expect[0] > expect[1]):
                y_hat.append(0)
            else:
                y_hat.append(1)
        #print("Terminal Weight",self.w)
        return y_hat
    
    def weight(self):
        return self.w
    