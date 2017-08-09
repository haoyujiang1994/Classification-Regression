# Classification-Regression
introduction and implementation of algorithm about Classification &amp; Regression


Question:
Implement the Perceptron Learning algorithm. Run it on the data file "classification.txt" ignoring the 5th column. That is, consider only the first 4 columns in each row. The first 3 columns are the coordinates of a point; and the 4th column is its classification label +1 or -1. Report your results.


Answer:In Python, we firstly handle the original data set datanup_x by adding a list of x(0) = 1 so that we could employ the perceptron learning on data better. Then a for-loop is designed for ensuring the the output after 7000 loops and avoid endless loop. We need to compare the data set of y with the testy calculated by the product of datanup_x and randomw randomly chosen. At the same time, the violations would be add into the violate_indxes list. If each point’s testy matches datanup_y, the randomw turns out to be right. Otherwise, we need to make some changes to randomw according to the difference between any violation’s testy and datanup_y, at the same time, the new randomw has to be tested again in the for-loop.

randomw = [[ 0.02675924][ 5.92127432][-4.89289481][-3.71029569]]
#Violations 22


Question:
Implement the Pocket algorithm and run it on the data file "classification.txt" ignoring the 4th column. That is, consider only the first 3 columns and the 5th column in each row. The first 3 columns are the coordinates of a point; and the 5th column is its classification label +1 or -1. Plot the number of misclassified points against the number of iterations of the algorithm. Run up to 7000 iterations.


Answer:In Python, pocket algorithm is applied to improve perceptron learning by remembering the current best randomw and avoiding the risks of increasing violations led by every change to randomw. In the for-loop, it is necessary to compare the violation caused by new randomw and the remembered best randomw. If the violations have been decreased, we would replace the best randomw with new randomw. Otherwise, we will keep the best randomw unchanged. A result of weights W with step size alpha= 1.6 and max number of iterations=7000 can be:

randomw = [[-0.6][-0.04680538][ 0.06739427][ 0.7663457 ]]

A plot on the number of misclassified points (vertical axis) against the number of iterations of algorithm (horizontal axis) can be:

violations = [1012, 1007, 1007, 988, 979, 979…]


Question:
Implement Logistic Regression and run it on the points in the data file "classification.txt" ignoring the 4th column. That is, consider only the first 3 columns and the 5th column in each row. The first 3 columns are the coordinates of a point; and the 5th column is its classification label +1 or -1. Use the sigmoid function Ɵ(s) = es/(1+es). Run up to 7000 iterations.


Answer:In Python, the aim of logistic regression is to maximize the production of possibilities of dataset, which takes sigmoid function into consideration. In the process, we need to change randomw continuously according to the concept of gradient descent.

randomw +=
(0.1/datanup_y.shape[0])*np.matrix((1/(1+exp(datanup_y[i]*randomw.T*datanup_x[i,:]).T))*datanup_y[i]*(datanup_x[i,:].T).transpose()

A result of weights W with step size alpha=0.1 and max number of iterations=7000 can be:

randomw = [[ 0.62173953][ 0.36069405][ 0.65464598][ 0.68697845]]


Question:
Implement Linear Regression and run it on the points in the data file "linear-regression.txt". The first 2 columns in each row represent the independent X and Y variables; and the 3rd column represents the dependent Z variable. Report your results.


Answer:In Python, we are able to obtain easily the randomw by an equation expressed in the class:

randomw = np.linalg.inv(datanup_x.T * datanup_x) * datanup_x.T * datanup_y

Therefore, it is simple to calculate the randomw in this case:

randomw = [ 0.01523535  1.08546357  3.99068855]

