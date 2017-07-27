import numpy as np
from numpy import random
import matplotlib.pyplot as mp
datanup = np.loadtxt("classification.txt",delimiter=",",dtype=float)
datanup_x = datanup[:,0:3]
datanup_y = datanup[:,4]
a = np.ones(datanup_x.shape[0])
datanup_x = np.insert(datanup_x, 0, values=a, axis=1)
randomw = random.random(size=(datanup_x.shape[1],1))

def pla(datanup_x,randomw,datanup_y):
    compare = []
    for j in range(0,7000):
        new_testy = []
        testy = np.dot(datanup_x,randomw)
        for item in testy:
            new_testy.append(item[0])
        new_testy = np.array(new_testy)
        R = new_testy > 0
        M = datanup_y > 0
        b = R!= M
        violate_idxes = [i for i, x in enumerate(b) if x]
        A = sum(b)
        if A == testy.shape[0]:
            break
        else:
            compare.append(A)
            if len(compare) <= 1:
                old_randomw = randomw
            elif compare[-1] < compare[-2]:
                old_randomw = randomw
            elif compare[-1] >= compare[-2]:
                compare[-1] = compare[-2]
            vio_idx = random.choice(violate_idxes)
            if datanup_y[vio_idx]>0:
                randomw = (old_randomw.T + np.dot(1.6,datanup_x[vio_idx, :]).T).T
            elif datanup_y[vio_idx]<0:
                randomw = (old_randomw.T - np.dot(1.6,datanup_x[vio_idx, :]).T).T

    x = range(0, 7000)
    y = compare
    mp.plot(x,y)
    mp.show()
    print old_randomw
    print y

pla(datanup_x,randomw,datanup_y)
