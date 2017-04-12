
# coding: utf-8

# In[196]:

import pandas as pd
import numpy as np


# In[197]:

data_train = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/kaggle_test_data.csv')


# In[198]:

features = list(data_train)[1:-1]
target = list(data_train)[-1]


# In[199]:

strings = []
for feature in list(data_train):
    if data_train[feature].dtype == 'object':
        curr_mode = data_train[feature].mode()[0]
        data_train[feature] = data_train[feature].apply(lambda x :
                                                       curr_mode if x == " ?" else x)
for feature in list(data_test):
    if data_test[feature].dtype == 'object':
        curr_mode = data_test[feature].mode()[0]
        data_test[feature] = data_test[feature].apply(lambda x :
                                                       curr_mode if x == " ?" else x)
continuous = []
for feature in features:
    if data_train[feature].dtype != 'object':
        continuous.append(feature)


# In[200]:

# one hot encoding train
for feature in list(data_train):
    if data_train[feature].dtype == "object":
        values = data_train[feature].unique()
        for value in values:
            data_train[value] = data_train[feature].apply(lambda z : 1 if z == value else 0)
#one hot encoding test
for feature in list(data_test):
    if data_test[feature].dtype == "object":
        values = data_test[feature].unique()
        for value in values:
            data_test[value] = data_test[feature].apply(lambda z : 1 if z == value else 0)


# In[201]:

#data_test doesnt has  Holand-Netherlands feature hence adding
data_test[' Holand-Netherlands'] = 0


# In[202]:

features = list(set(list(data_train)) - set(features+[target,"id"])) + continuous
#print features
#normalizing continuous values
for feature in continuous:
    data_train[feature] = data_train[feature]*1.0/(max(data_train[feature]) - min(data_train[feature]))
    data_test[feature] = data_test[feature]*1.0/(max(data_test[feature]) - min(data_test[feature]))


# In[203]:

input_train_matrix = data_train[features].as_matrix()
input_test_matrix = data_test[features].as_matrix()
target_train_matrix = np.array([data_train[target]]).T
print "Train matrix : ",input_train_matrix.shape
print "Test matrix : ",input_test_matrix.shape
print "Target input matrix : ",target_train_matrix.shape


# In[214]:

class Neural_Network(object):
    
    def __init__(self,n_inputs,n_hiddens,n_outputs):
        self.inputLayerSize = n_inputs
        self.hiddenLayerSize = n_hiddens
        self.outputLayerSize = n_outputs
        
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
    
    def forward(self,X):
        self.z2 = np.dot(X,self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat
    
    def sigmoid(self,z):
        return 1.0/(1.0 + np.exp(-z))
    
    def sigmoid_prime(self,z):
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self,x,y):
        self.yHat = self.forward(x)
        J = 0.5*sum((y-self.yHat)**2)
        return J
    
    def costFunctionPrime(self,x,y):
        self.yHat = self.forward(x)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoid_prime(self.z3))
        dJdW2 = np.dot(self.a2.T,delta3)
        
        delta2 = np.dot(delta3,self.W2.T)*self.sigmoid_prime(self.z2)
        dJdW1 = np.dot(x.T,delta2)
        
        return dJdW1,dJdW2
    
    def train(self,x,y,n_iter,l_rate):
            
            for n in range(n_iter):
                for j in range(x.shape[0]):
                    curr_x = x[j:j+1,:]
                    curr_y = y[j:j+1,:]
                    cost = self.costFunction(curr_x,curr_y)
                    del1,del2 = self.costFunctionPrime(curr_x,curr_y)
                
                    self.W1 -= l_rate*del1
                    self.W2 -= l_rate*del2
                print "Iteration : ",n," Cost : ",self.costFunction(x,y)
    
    def predict(self,x):
        yHat = self.forward(x)
        self.yHat = (yHat >= 0.5).astype(int)
        return self.yHat
        


# In[215]:

NN1 = Neural_Network(105,105,1)


# In[217]:

NN1.train(input_train_matrix,target_train_matrix,10,0.01)


# In[218]:

prediction = NN1.predict(input_test_matrix)
prediction = np.c_[data_test['id'],prediction] 

np.savetxt('output.csv',prediction,fmt = "%i",delimiter = ",",header = "'id','salary'", newline =  "\n")

# In[ ]:



