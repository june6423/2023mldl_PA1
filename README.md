[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/HNm_Jrs1)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10977744&assignment_repo_type=AssignmentRepo)
# Coding Assignment 1: Naive Bayes classifier with bag-of-words model, Logistic regression models and Perceptron.

## Introduction

The first coding assignment asks you to implement three classification models given dataset.

The assignment has three parts as following:
1. Implement Naive Bayes text classification model with bag-of-words.
1. Implement classification models for Iris dataset; logistic regression, regularized logistic regression and perceptron.
2. Compare your implementation with the same models implemented in `scikit-learn`

We provide the code consisting of several Python files, which you will need to read and understand in order to complete the assignment.

**Note**: we will use `Python 3.x` for the project. 

## Deadline
May 10, 2023 11:59PM KST (*Late Submission not allowed*)

### Submission checklist
* Push your code to [our github classroom page's CA1 section](https://classroom.github.com/a/HNm_Jrs1)
* Submit your report to [LMS](https://lms.gist.ac.kr)

---
## Preparation

### Installing prerequisites

The prerequisite usually refers to the necessary library that your code can run with. They are also known as `dependency`. To install the prerequisite, simply type in the shell prompt (not in a python interpreter) the following:

```
$ pip install -r requirements.txt
```

---
## Files

**Files you'll make:**

* `datasets.py`: Data provider. 
* `naivebayes.py`: Naive Bayes text classification with bag-of-words model.
* `logistic.py`: Logistic regression models.
* `perceptron.py`: Perceptron classification model.
* `util.py`: A bunch of utility functions!

---
## What to submit
**Push to your github classroom** 

- All of the python files listed above (under "Files you'll edit"). 
- `report.pdf` file that answers all the written questions in this assignment (denoted by `"REPORT#:"` in this documentation).

---
### Note
**Academic dishonesty:** We will be checking your code against other submissions in the class for logical redundancy. If you copy someone else's code and submit it with minor changes, we will know. These cheat detectors are quite hard to fool, so please don't try. We trust you all to submit your own work only; please don't let us down. If you do, we will pursue the strongest consequences available to us.

---
## Prepare the dataset for text classification (5%)

**For Naive Bayes classifier**, using provided 'data.zip' file, build training data for SPAM/HAM emails.



--
## Implementation of the Naive Bayes classifier for SPAM/HAM email text classification (using Bag-of-words model) (40%)
**1. Text preprocessing: Turn the text content into numerical feature vectors and compute frequencies (10%)**

For this text preprocessing, you can use 'scikit-learn' package as follows:

```
>>> from sklearn.feature_extraction.text import CountVectorizer
```

**2. Training a Naive Bayes classifier (10%)**

**3. Load testing data and Predict a class (10%)**

After training, predict an email in 'mytest' directory.

**4. Implement Laplace smoothing (10%)**
 
 
`REPORT1`: Report prior and likelihood distributions for each class.

 
 ---
## Prepare the dataset for Logistic regression and perceptron (5%)

**For Logistic regressions**, use 'sklearn` package and 'Iris' dataset. 

```
>>> from sklearn import datasets
>>> iris = datasets.load_iris()
>>> x=iris.data
>>> y=iris.target
```

For the binary classification, use first two features and only [0,1] labels in the dataset.  
Divide this dataset into two groups: train set (tr_x, tr_y) and validation set (val_x, val_y).

 
 
---
## Logistic Regression Model (Gradient ascent algorithm) (10%)

You can now implement the logistic regression model to predict the class (`y`) with the input data (`x`).

```
>>> from logistic import *
>>> model = Logistic(maxIter, eta) # set maximum iteration and eta (learning rate) for model updates
>>> model.train_GA(tr_x, tr_y)
>>> y_hat = model.predict(val_x)
>>> acc = computeClassificationAcc(val_y, y_hat) 
>>> print(acc)
0.96  # for example
```

`REPORT2`: Report the error of your logistic regression model and draw the decision boundary using the first two features. Sweep `eta` from 0.01 to 10.0 (or some other reasonable values) with a reasonable sized step (0.01, 0.1, etc), plot a graph (x-axis: eta, y-axis: acc) and discuss the effect of the eta.


---
## Logistic Regression Model (Stochastic gradient ascent algorithm) (10%)

You can now implement the logistic regression model to predict the class (`y`) with the input data (`x`).

```
>>> from logistic import *
>>> model = Logistic(maxIter, eta) # set maximum iteration and eta (learning rate) for model updates
>>> model.train_SGA(tr_x, tr_y)
>>> y_hat = model.predict(val_x)
>>> acc = computeClassificationAcc(val_y, y_hat) 
>>> print(acc)
0.96  # for example
```

`REPORT3`: Report the error of your logistic regression model and draw the decision boundary using the first two features. Using the `eta` you found from `REPORT4`, run with different numbers of `iterations`, plot a graph (x-axis: eta, y-axis: acc) and discuss the effect of the number of iterations.

---
## Regularized Logistic Regression (MCAP) Model (Stochastic gradient ascent algorithm) (10%)

You can now implement the regularized logistic regression model to predict the class (`y`) with the input data (`x`).

```
>>> from logistic import *
>>> model = Logistic(maxIter, eta, lambda) # set maximum iteration and eta (learning rate), lambda (weight for the regularization term) for model updates
>>> model.train_reg_SGA(tr_x, tr_y)
>>> y_hat = model.predict(val_x)
>>> acc = computeClassificationAcc(val_y, y_hat) 
>>> print(acc)
0.96  # for example
```

`REPORT4`: Report the error of your regularized logistic regression model and draw the decision boundary using the first two features. Sweep `lambda` from 0.0 to 5.0 (or some other reasonable values) with a reasonable sized step (e.g., 0.5), plot a graph (x-axis: lambda, y-axis: acc) and discuss the effect of the lambda (especially comparing with vanilla logistic when `lambda=0`.)



---
## Perceptron (10%)

You can now implement the perceptron to predict the class (`y`) with the input data (`x`).


```
>>> from perceptron import *
>>> model = Perceptron(threshold) # set threshold value or 
>>> model.train(tr_x, tr_y)
>>> y_hat = model.predict(val_x)
>>> acc = computeClassificationAcc(val_y, y_hat) 
>>> print(acc)
0.96  # for example
```

`REPORT5`: Report the error of your perceptron and draw the decision boundary using the first two features. Using different values for `threshold`, plot a graph (x-axis: thresh, y-axis: acc) and discuss the effect of the threshold. Discuss what happens when 'threshold=0' and why.



---
## Compare your implementations with `scikit-learn` library (20%)

In [scikit-learn library](https://scikit-learn.org/), there are all implementation of what you have implemented
1. [naive bayes text classifier](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
2. [vanilla logistic regression model](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
3. [regularized logistic regression model](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
4. [perceptron](https://scikit-learn.org/stable/modules/linear_model.html#perceptron)

`REPORT6`: Compare the error by your implementations of Naive Bayes text classification model and Naive Bayes text classification model in scikit-learn and discuss the reason for the difference. If they are identical, report and claim you're awesome!

`REPORT7`: Compare the error by your implementations of logistic regression and logistic regression model in scikit-learn and discuss the reason for the difference. If they are identical, report and claim you're awesome!

`REPORT8`: Compare the error by your implementations of l2-regularized logistic regression and l2-regularized logistic regression model in scikit-learn and discuss the reason for the difference. If they are identical, report and claim you're awesome!

`REPORT9`: Compare the error by your implementations of perceptron and perceptron model in scikit-learn and discuss the reason for the difference. If they are identical, report and claim you're awesome!


