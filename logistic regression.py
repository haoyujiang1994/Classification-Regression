import numpy as np
from numpy import random
from math import *
datanup = np.loadtxt("classification.txt",delimiter=",",dtype=float)
datanup_x = datanup[:,0:3]
datanup_y = datanup[:,4]
a = np.ones(datanup_x.shape[0])
datanup_x = np.insert(datanup_x, 0, values=a, axis=1)
randomw = random.random(size=(datanup_x.shape[1],1))

def logreg(datanup_x,randomw,datanup_y):
    for j in range(0,7000):
        for i in range(0,datanup_x.shape[0]):
            a = np.dot(np.dot(datanup_y[i],randomw.T),(datanup_x[i,:]).T)
            part_randomw = (1/(1+exp(a)))*datanup_y[i]*(datanup_x[i,:]).T
            randWT = np.matrix(part_randomw).transpose()
        randomw = randomw + (np.dot((0.1/datanup_y.shape[0]),randWT))
    print randomw

logreg(datanup_x, randomw, datanup_y)