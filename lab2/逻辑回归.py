from sklearn import datasets, model_selection
import numpy as np
import math
from matplotlib import pyplot as plt


lamb = 100
learningRate = 0.02

def getData(filePath):
    data =  datasets.load_svmlight_file(filePath, n_features= 123)
    return data[0], data[1]

#这里表示选择的loss函数
def getLoss(X,Y,w):
    loss = 0
    i = 0
    while i < X.size / 123:
        loss += np.log(1 + math.exp(-Y[i] * X[i].dot(w)))
        i += 1
    return loss / (X.size / 123) + 0.5 * lamb * np.linalg.norm(w,ord=2) ** 2

def getBetterW(X,Y,w):
    sum = np.zeros((123,))
    for i in range(0, batch_size):
        sum = sum + (1 / (1 + np.exp(-w.T.dot(X[i].T))) - Y[i]) * X[i].T
    return w - (learningRate * sum )/ batch_size



def getLValid(X,Y,w,threshod):
    loss = 0
    for i in range(0, int(X.size / 123)):
        if 1 / (1 + np.exp(-w.T.dot(X[i].T))) > threshod:
            if Y[i] != 1:
                loss += np.log(1 + math.exp(-Y[i] * X[i].dot(w)))
        else:
            if Y[i] != 0:
                loss += np.log(1 + math.exp(-Y[i] * X[i].dot(w)))
        i += 1
    return loss / (X.size / 123)
    


batch_size = 1024
x_train, y_train = getData('F:\机器学习\lab2\\a9a')
x_valid, y_valid = getData('F:\机器学习\lab2\\a9a.t')
x_train = x_train.A
x_valid = x_valid.A
w =np.zeros((123,))
OLoss = getLValid(x_valid,y_valid,w,0.5)
print('original loss is: ', OLoss)

x = range(0, 200)
y = [OLoss]

for i in range(1, 200):
    #制作batch
    state = np.random.get_state()
    np.random.shuffle(x_train)
    np.random.set_state(state)
    np.random.shuffle(y_train)
    x_batch = x_train[0:batch_size]
    y_batch = y_train[0:batch_size]
    w = getBetterW(x_batch,y_batch,w)
    learningRate *= 0.9

    LossOfValid = getLValid(x_valid,y_valid,w,0.5)
    print('loss of valid becomes: ', LossOfValid)
    y.append(LossOfValid)
plt.title('LValidation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(x,y)
plt.show()