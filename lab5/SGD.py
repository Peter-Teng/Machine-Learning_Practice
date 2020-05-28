from sklearn import datasets, model_selection
import numpy as np
from matplotlib import pyplot as plt

potential_features = 50
lamb_P = 0.5
lamb_Q = 0.5
learningrate = 0.01

def getLoss(U,I,R,P,Q):
    loss = 0 
    TMP = P.dot(Q)
    for i in range(0, len(U)):
        loss += (R[U[i]][I[i]] - TMP[U[i]][I[i]]) ** 2 + lamb_P * np.linalg.norm(P[U[i]],1) ** 2 + lamb_Q * np.linalg.norm(Q[:,I[i]],1)
    return loss


def getGradient(r, guess, p, q):
    E = r - guess
    GradientP = E * (-q) + lamb_P * p
    GradientQ = E * (-p) + lamb_Q * q
    return GradientP, GradientQ


file_test = open(r'F:\机器学习\lab5\ml-100k\u1.base','r',encoding='ANSI')
file_valid = open(r'F:\机器学习\lab5\ml-100k\u1.test','r',encoding='ANSI')

print('loading statistics')
Omega_u_train = []
Omega_i_train = []
Omega_u_valid = []
Omega_i_valid = []
matrix_R = np.zeros((943,1682))
matrix_Valid = np.zeros((943,1682))
for line in file_test:
    tmp = line.split()
    matrix_R[int(tmp[0]) - 1][int(tmp[1]) - 1] = int(tmp[2])
    Omega_u_train.append(int(tmp[0]) - 1)
    Omega_i_train.append(int(tmp[1]) - 1)

for line in file_valid:
    tmp = line.split()
    matrix_Valid[int(tmp[0]) - 1][int(tmp[1]) - 1] = int(tmp[2])
    Omega_u_valid.append(int(tmp[0]) - 1)
    Omega_i_valid.append(int(tmp[0]) - 1)
print('statistics okay!')

matrix_P = np.ones((943,potential_features))
matrix_Q = np.ones((potential_features,1682))

loss = getLoss(Omega_u_valid,Omega_i_valid,matrix_Valid,matrix_P,matrix_Q)
print('the original loss is:', loss)

x = range(0, 5000)
y = [loss]

for k in range(1, 5000):
    #随机Omega集中的r[u][i] = r[Omega_u_train[i]][Omega_i_train[i]]
    i = np.random.randint(0,80000)
    GradientP, GradientQ = getGradient(matrix_R[Omega_u_train[i]][Omega_i_train[i]], 
                                        matrix_P[Omega_u_train[i]].dot(matrix_Q[:,Omega_i_train[i]]), 
                                        matrix_P[Omega_u_train[i]], matrix_Q[:,Omega_i_train[i]])
    matrix_P[Omega_u_train[i]] = matrix_P[Omega_u_train[i]] - learningrate * GradientP
    matrix_Q[:,Omega_i_train[i]] = matrix_Q[:,Omega_i_train[i]] - learningrate * GradientQ
    loss = getLoss(Omega_u_valid,Omega_i_valid,matrix_Valid,matrix_P,matrix_Q)
    y.append(loss)
    print('the loss then becomes:',loss)
plt.title('LValidation')
plt.xlabel("iterations")
plt.ylabel("loss")
plt.plot(x,y)
plt.savefig("F:\机器学习\lab5\\result.png")
plt.show()