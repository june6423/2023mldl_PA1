from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import os
from datasets import *


class Naivebayes():
    def __init__(self):
        self.ham_path = "./data/data/myham/"
        self.spam_path = "./data/data/myspam/"
        self.test_path = "./data/data/mytest/"
        self.ham_prior = 0
        self.spam_prior = 0
        self.likehood_ham = dict()
        self.likehood_spam = dict()
    
    def train(self):
        file_list = os.listdir(self.ham_path)
        content = []

        for file in file_list:
            with open(self.ham_path + file) as f:
                content.append(f.read())

        vectorizer = CountVectorizer()
        words = vectorizer.fit_transform(content)

        namedict = dict()
        words_array = words.toarray()
        likehood = np.zeros(len(words_array[0]))

        for word in vectorizer.vocabulary_:
            namedict[vectorizer.vocabulary_[word]] = word
            
        for fileindex in range(len(words_array)):
            for index in range(len(words_array[0])):
                likehood[index] = likehood[index] + words_array[fileindex][index]
                self.ham_prior = self.ham_prior + words_array[fileindex][index]

        for index in range(len(likehood)):
            self.likehood_ham[namedict[index]]=likehood[index]
                
        file_list = os.listdir(self.spam_path)
        content = []
        
        for file in file_list:
            with open(self.spam_path + file) as f:
                content.append(f.read())

        vectorizer = CountVectorizer()
        words = vectorizer.fit_transform(content)
        
        namedict = dict()
        words_array = words.toarray()
        likehood = np.zeros(len(words_array[0]))

        for word in vectorizer.vocabulary_:
            namedict[vectorizer.vocabulary_[word]] = word
            
        for fileindex in range(len(words_array)):
            for index in range(len(words_array[0])):
                likehood[index] = likehood[index] + words_array[fileindex][index]
                self.spam_prior = self.spam_prior + words_array[fileindex][index]

        for index in range(len(likehood)):
            self.likehood_spam[namedict[index]]=likehood[index]
                
    def predict(self):
        file_list = os.listdir(self.test_path)
        for file in file_list:
            content = []
            p_ham = np.log(1/2)
            p_spam = np.log(1/2)
            
            with open(self.test_path + file) as f:
                content.append(f.read())
            
            vectorizer = CountVectorizer()
            words = vectorizer.fit_transform(content)
            print(vectorizer.stop_words_)
            for word in vectorizer.vocabulary_:
                for i in range(vectorizer.vocabulary_[word]):
                    if(word not in self.likehood_ham):
                        p_ham = p_ham * (0 / self.ham_prior)
                    else:
                        p_ham = p_ham  * self.likehood_ham[word] /self.ham_prior
                    if(word not in self.likehood_spam):
                        p_spam = p_spam + (0 / self.spam_prior)
                    else:
                        p_spam = p_spam * self.likehood_spam[word] / self.spam_prior
                        
            if(p_ham > p_spam):
                print(file," is not spam")
            else:
                print(file," is spam")
            print("Probability of p_ham and p_spam",p_ham, p_spam)
            
    def predict_with_laplace(self):
        file_list = os.listdir(self.test_path)
        for file in file_list:
            content = []
            p_ham = np.log(self.ham_prior)
            p_spam = np.log(self.spam_prior)
            
            with open(self.test_path + file) as f:
                content.append(f.read())
            
            vectorizer = CountVectorizer()
            words = vectorizer.fit_transform(content)
            for word in vectorizer.vocabulary_:
                if(word not in self.likehood_ham):
                    self.likehood_ham[word] = 0
                if(word not in self.likehood_spam):
                    self.likehood_spam[word] = 0
                    
            for word in vectorizer.vocabulary_:
                for i in range(vectorizer.vocabulary_[word]):
                    p_ham = p_ham  + np.log((self.likehood_ham[word] + 1) / (self.ham_prior + len(self.likehood_ham)))
                    p_spam = p_spam  + np.log((self.likehood_spam[word] + 1) / (self.spam_prior + len(self.likehood_spam)))
                        
            if(p_ham > p_spam):
                print(file," is not spam")
            else:
                print(file," is spam")
            print("Log Probability of p_ham and p_spam with laplace smoothing",p_ham, p_spam)
            
    def print_weight(self):
        print("Ham prior",self.ham_prior)
        print("Spam prior",self.spam_prior)
        print("Ham likewood",self.likehood_ham)
        print("Spam likewood",self.likehood_spam)
        
#Code for my naivebayes classifier implementation
""" model = Naivebayes()
model.train()
model.predict()
model.predict_with_laplace() 
model.print_weight() """


#Code for scikit learn naive bayes classifier

def scikit_naivebayes():
    
    ham_path = "./data/data/myham/"
    spam_path = "./data/data/myspam/"
    test_path = "./data/data/mytest/"
    
    file_list = os.listdir(ham_path)
    content = []
    for file in file_list:
        with open(ham_path + file) as f:
            content.append(f.read())
    file_list = os.listdir(spam_path)
    for file in file_list:
        with open(spam_path + file) as f:
            content.append(f.read())
    label = [0,0,0,1,1,1] # 0 means not spam, 1 means spam
    
    vectorizer = CountVectorizer()
    words = vectorizer.fit_transform(content)
    tf_transformer = TfidfTransformer(use_idf=False).fit(words)
    train_words = tf_transformer.transform(words)
    clf = MultinomialNB().fit(train_words,label)
    
    content = []
    file_list = os.listdir(test_path)
    for file in file_list:
        with open(test_path + file) as f:
            content.append(f.read())
    test_words = vectorizer.transform(content)
    test_tf = tf_transformer.transform(test_words)
    predicted = clf.predict(test_tf)
    print(predicted) # 0 means not spam, 1 means spam
    print(clf.predict_proba(test_tf))
    
scikit_naivebayes()