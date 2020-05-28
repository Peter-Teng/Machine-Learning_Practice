import numpy as np
import pickle
import ensemble
from sklearn.metrics import classification_report

if __name__ == "__main__":
    #划分训练集与验证集
    print('loading features!')
    dataX = pickle.load(open('trainingX.data','rb'))
    dataY = pickle.load(open('trainingY.data','rb'))
    print('load success!')
    print(dataX)
    print(dataX.shape)
    x_train = dataX[0:900]
    y_train = dataY[0:900].flatten()
    x_valid = dataX[900:1000]
    y_valid = dataY[900:1000].flatten()
    classifier = ensemble.AdaBoostClassifier('sklearn.tree.DecisionTreeClassifier',3)
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_valid,0)
    print(classification_report(y_valid,y_pred))
    with open('laclassifier_report1.txt','w+') as f:
        f.write(classification_report(y_valid, y_pred))
        f.close()

