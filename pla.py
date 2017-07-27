import numpy as np
from numpy import random
datanup = np.loadtxt("classification.txt",delimiter=",",dtype=float)
datanup_x = datanup[:,0:3]
datanup_y = datanup[:,3]
a = np.ones(datanup_x.shape[0])
datanup_x = np.insert(datanup_x, 0, values=a, axis=1)
randomw = random.random(size=(datanup_x.shape[1],1))

def pla(datanup_x,randomw,datanup_y):
    for j in range(0, 7000):
        new_testy = []
        testy = np.dot(datanup_x, randomw)
        for item in testy:
            new_testy.append(item[0])
        new_testy = np.array(new_testy)
        R = new_testy > 0
        M = datanup_y > 0
        b = R != M
        violate_idxes = [i for i, x in enumerate(b) if x]
        A = sum(b)
        if A == testy.shape[0]:
            break
        else:
            vio_idx = random.choice(violate_idxes)
            if datanup_y[vio_idx] > 0:
                randomw = (randomw.T + np.dot(0.1, datanup_x[vio_idx, :]).T).T
            elif datanup_y[vio_idx] < 0:
                randomw = (randomw.T - np.dot(0.1, datanup_x[vio_idx, :]).T).T
    return randomw, A

[w, violations] = pla(datanup_x,randomw,datanup_y)
print w
print "#Violations", violations