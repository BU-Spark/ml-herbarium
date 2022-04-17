import pandas as pd
import numpy as np
from numpy import log, dot, e
from numpy.random import rand
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# start default svm classifier

data = load_breast_cancer()

X = data['data']
Y = data['target']
#Y[Y == 0] = -1

print("X shape", X.shape, sep='\t')

print("Y shape", Y.shape, sep='\t')

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)# -- Code Required --

print("xtrain shape:", xtrain.shape, sep='\t')
print("xtest shape:", xtest.shape, '\n', sep='\t')

print("ytrain shape:", ytrain.shape, sep='\t')
print("ytest shape:", ytest.shape, sep='\t')

def sigmoid(z): 
    sig = 1/(1+np.exp(-z))
    return sig

from numpy import linalg as LA
def loss(X, y, w, lmda=0.001):
    n = X.shape[0]
    hx = sigmoid(np.dot(X,w))
    neghx = 1- sigmoid(np.dot(X,w))
    nll = (-1/n) *np.sum(np.matmul(y,np.log(hx)) + np.matmul((1-y), np.log(neghx)))
    reg = lmda/2 * LA.norm(w,2)

    return nll + reg

w = rand(xtrain.shape[1], 1)
#print(xtrain.shape, w.shape)
temp = loss(xtrain, ytrain, w)

print(temp)

def predict(X, w): 
    sig = sigmoid(np.dot(X,w))     
    predictions = [x >= 1/2 for x in sig]
    return predictions

def gradient(X, y, w, lmda=0.001):
    n, d = X.shape
    y = np.expand_dims(y, axis=1)
    y_hat = sigmoid(np.dot(X,w))
    grad = (-1/n) * (np.matmul(np.transpose(X),(y-y_hat))) + np.dot(lmda,w)
    return grad

w = rand(xtrain.shape[1], 1)
print(xtrain.shape, w.shape)
temp = gradient(xtrain, ytrain, w)
print(temp.shape)

def fit(X, y, epochs=200, lr=0.001, lmda=0.001): # default arguments     
    loss_arr = []
    w = rand(X.shape[1], 1)
    #print(w.shape, "W")
                
    for _ in range(epochs):        
        # Gradient Descent!
        curr_loss = loss(X,y,w)
        loss_arr.append(curr_loss) 

        grad = gradient(X,y,w,lmda)
        w = w - lr * grad
    
    return w, loss_arr

w, loss_arr = fit(xtrain,ytrain,500, 0.1, 0.01)
plt.figure()
plt.plot(range(1,501), loss_arr)
plt.title("Loss over epochs")
plt.ylim(bottom = -0.1)
plt.show()

# percentage of cases correctly classified
CCR = predict(xtest,w)
count = 0
for i in range(sum(sum(CCR))):
  if sum(CCR[i]) == ytest[i]:
    count += 1
print(count / sum(sum(CCR)))