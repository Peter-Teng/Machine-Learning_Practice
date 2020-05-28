from sklearn import datasets, model_selection
import numpy as np
from matplotlib import pyplot as plt

def getLoss(X,Y,w):
    loss = 0
    i = 0
    while i < X.size/13:
        loss += 0.5 * (Y[i] - w.T.dot(X[i])) ** 2
        i += 1
    return loss


def calculateGradient(X,Y,w):
    return - (X.T * Y).reshape((13,1)) + X.T.dot(X) * (w)

data = datasets.load_svmlight_file('F:\机器学习\lab1\housing_scale')
x = data[0]
y = data[1]
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(x, y, test_size=0.33,random_state=42)
X_train = X_train.A
X_valid = X_valid.A
w = np.zeros((13,1))
learningRate = 0.001
print('initial w is: \n', w)
Loss = getLoss(X_valid,y_valid,w)
print('the loss is: ' , Loss)

x = range(0,1000)
y = [Loss]

for i in range(1,1000):
    random = int(np.random.rand() * 339)
    gradient = calculateGradient(X_train[random],y_train[random],w)
    # print('the gradient is: \n', gradient)
    w = w - learningRate * gradient
    # print('new w is: \n', w)
    #print('new loss of training is: ', getLoss(X_train,y_train,w))
    Loss = getLoss(X_valid,y_valid,w)
    print('new loss of validation set is: ',Loss)
    y.append(Loss)
plt.title('Loss of SGD')
plt.xlabel("iterations")
plt.ylabel("loss")
plt.plot(x,y)
plt.savefig("D:/temp.png")

