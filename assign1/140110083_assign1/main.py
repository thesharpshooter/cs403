import numpy as np

data =raw_input()
print data
print data+"/train.csv"
print data+"/test.csv"

data_train = np.genfromtxt((str(data)+'/train.csv'),delimiter = ",",skip_header = 1)
data_test = np.genfromtxt((str(data)+'/test.csv'),delimiter = ',',skip_header = 1)
for i in range(1,14):
    mean = data_train[:,i].mean()
    std = data_train[:,i].std()
    data_train[:,i] = (data_train[:,i] - mean)/std
for i in range(1,14):
    mean = data_test[:,i].mean()
    std = data_test[:,i].std()
    data_test[:,i] = (data_test[:,i] - mean)/std

X = data_train[:,1:-1]
Y = data_train[:,-1]
ones = np.array([1]*400)
H_TRAIN = np.c_[ones,X,X**2]


W = np.array([0.0]*27)
STEP = 6e-5
LAMBDA = 0.01
iterations = 10000

def ridge_regression(x,y,step,l2,iteraions,w):
    H = x
    Y = y
    STEP = step
    LAMBDA = l2
    W = w

    OUTPUT = H.dot(W)
    ERROR = Y-OUTPUT
    GRADIANT = -2*(H.transpose().dot(ERROR))
    MAG = np.math.sqrt((GRADIANT**2).sum())

    print "GRADIANT : ",GRADIANT
    print "MAG : ",MAG

    for i in range(iterations):
        W[0] = W[0] - STEP*GRADIANT[0]
        W[1:] = (1 - 2*STEP*LAMBDA)*W[1:] - STEP*GRADIANT[1:]

        OUTPUT = H.dot(W)
        ERROR = Y-OUTPUT
        GRADIANT = -2*(H.transpose().dot(ERROR))
        MAG = np.math.sqrt((GRADIANT**2).sum())
        
        print "W IS : ",W
        print "MAG IS : ",MAG
    
    print "FINAL W IS : ",W
    print "FINAL MAGNITUDE IS : ",MAG

ridge_regression(H_TRAIN,Y,STEP,LAMBDA,iterations,W)
ones_test = np.array([1]*105)
H_TEST = np.c_[ones_test,data_test[:,1:],(data_test[:,1:]**2)]
Y_OUTPUT = H_TEST.dot(W)
Y_OUTPUT = np.c_[data_test[:,0],Y_OUTPUT]


np.savetxt('output.csv',Y_OUTPUT,delimiter = ",",fmt = "%d,%f",header = "ID,MEDV",comments = "")
np.savetxt('output1.csv',Y_OUTPUT,delimiter = ",",fmt = "%d,%f",header = "ID,MEDV",comments = "")
np.savetxt('output2.csv',Y_OUTPUT,delimiter = ",",fmt = "%d,%f",header = "ID,MEDV",comments = "")
