from sklearn import datasets, model_selection
import numpy as np

def getLoss(X,Y,w):
    loss = 0
    i = 0
    while i < X.size/13:
        loss += 0.5 * (Y[i] - w.T.dot(X[i])) ** 2
        i += 1
    return loss

def getBestW(X,Y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)


data = datasets.load_svmlight_file('F:\机器学习\lab1\housing_scale')
x = data[0]
y = data[1]
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(x, y, test_size=0.33,random_state=42)
X_train = X_train.A
X_valid = X_valid.A
w = np.zeros((13,1))
print('initial w is: \n', w)
print('the loss is: ' , getLoss(X_train,y_train,w))
w_star = getBestW(X_train,y_train)
print('w* is: ', w_star)
print('then loss then become: ', getLoss(X_train,y_train,w_star))
print('the loss of validation set is: ', getLoss(X_valid,y_valid,w_star))
