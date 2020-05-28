from PIL import Image
import feature
import numpy as np
import pickle

#提取特征的代码
X = np.zeros(shape = (165600,))
Y = np.zeros(shape = (1,))
#提取NPD特征
for i in range(0,20):
    f = np.array(Image.open("F:\机器学习\lab3\ML2019-lab-03-master\data\\face\%03d.jpg"%i))
    X = np.row_stack((X,feature.NPDFeature(f).extract()))
    Y = np.row_stack((Y,np.ones((1,))))
    nf = np.array(Image.open("F:\机器学习\lab3\ML2019-lab-03-master\data\\nonface\\non_%03d.jpg"%i))
    X = np.row_stack((X,feature.NPDFeature(f).extract()))
    Y = np.row_stack((Y,np.array([-1])))
X = np.delete(X,0,axis = 0)
Y = np.delete(Y,0,axis = 0)

state = np.random.get_state()
np.random.shuffle(X)
np.random.set_state(state)
np.random.shuffle(Y)


dataX = open('lab3\\trainingX.data','wb')
dataY = open('lab3\\trainingY.data','wb')
pickle.dump(X,dataX)
pickle.dump(Y,dataY)