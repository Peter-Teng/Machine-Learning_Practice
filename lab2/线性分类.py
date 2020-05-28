from sklearn import datasets, model_selection
import numpy as np
import math
from matplotlib import pyplot as plt

w =np.ones((123,))
C = 100
b = 1
learningRate = 0.01
batch_size = 1024


def getData(filePath):
    data =  datasets.load_svmlight_file(filePath, n_features= 123)
    return data[0], data[1]

#这里表示选择的loss函数
def getLoss(X,Y,w):
    sum = 0
    for i in range(0, int(X.size / 123)):
        sum += np.max([0, 1 - Y[i] * (w.T.dot(X[i].T) + b)])
    return 0.5 * np.linalg.norm(w) ** 2 + C / (X.size / 123) * sum

def MSGD(X,Y,w):
    sumW = 0
    sumB = 0
    for i in range(0, int(X.size / 123)):
        if 1 - Y[i] * (w.T.dot(X[i].T) + b) >= 0:
            sumW += -Y[i] * X[i]
            sumB += -Y[i]
    GraW = w + C / batch_size * sumW
    GraB = C / batch_size * sumB
    return w - learningRate * GraW, b - learningRate * GraB

def getLValid(X,Y,w,threshold):
    global Errorcount
    Errorcount = 0
    loss = 0
    for i in range(0, int(X.size / 123)):
        tmp = X[i].dot(w) + b
        if tmp >= threshold:
            if Y[i] != 1:
                Errorcount += 1
                loss += np.max([0, 1 - Y[i] * (w.T.dot(X[i].T) + b)])
        elif tmp < threshold:
            if Y[i] != -1:
                Errorcount += 1
                loss += np.max([0, 1 - Y[i] * (w.T.dot(X[i].T) + b)])
        i += 1
    return 0.5 * np.linalg.norm(w) ** 2 + C / (X.size / 123) * loss

x_train, y_train = getData('F:\机器学习\lab2\\a9a')
x_valid, y_valid = getData('F:\机器学习\lab2\\a9a.t')
x_train = x_train.A
x_valid = x_valid.A
OLoss = getLValid(x_valid,y_valid,w,0)
print('original loss is: ', OLoss)

x = range(0, 2000)
y = [OLoss]

for i in range(1, 2000):
    #制作batch
    state = np.random.get_state()
    np.random.shuffle(x_train)
    np.random.set_state(state)
    np.random.shuffle(y_train)
    x_batch = x_train[0:batch_size]
    y_batch = y_train[0:batch_size]

    learningRate = learningRate * (1 + learningRate*i) ** -1
    w, b = MSGD(x_batch,y_batch,w)
    newLoss = getLValid(x_valid,y_valid,w, 0)
    print('new loss becomes: ', newLoss)
    y.append(newLoss)
print(Errorcount)
print('correct rate:', str(100 - Errorcount / 16281 * 100) + '%')
plt.title('LValidation')
plt.xlabel('epoch')
plt.ylabel("Loss")
plt.plot(x,y)
plt.show()
plt.savefig('F://tmp.png')