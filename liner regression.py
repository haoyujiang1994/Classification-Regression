import numpy as np
datanup = np.loadtxt("linear-regression.txt",delimiter=",",dtype=float)
datanup_x = datanup[:,0:2]
datanup_y = datanup[:,2]
a = np.ones(datanup_x.shape[0])
datanup_x = np.insert(datanup_x, 0, values=a, axis=1)
randomw = np.dot((np.dot(np.linalg.inv(np.dot(datanup_x.T,datanup_x)), datanup_x.T)), datanup_y)

print randomw.T
