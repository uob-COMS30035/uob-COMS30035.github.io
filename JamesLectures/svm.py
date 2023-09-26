from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

X = [[0, 1], [2, 3], [4, 2], [0, 2], [11, 2], [3, 2]]
y = [0 if x1 < x2 else 1 for [x1,x2] in X]

#X.append([6,2])
#y.append(0)

X = np.array(X)
y = np.array(y)

test_data = [[2,3],[33,-1]]

for knl in 'linear', 'poly', 'rbf':

    clf = svm.SVC(kernel=knl)
    clf.fit(X, y)

    preds = clf.predict(test_data)

    for i, td in enumerate(test_data):
        print('Prediction for {0} is {1}.\n'.format(td,preds[i]))


    print('Support vectors are:', clf.support_vectors_)
    print('Dual coefficients are:', clf.dual_coef_)

    plt.scatter(X[:,0],X[:,1],s=10,c=y)
    plt.scatter(X[clf.support_,0],X[clf.support_,1],s=50,facecolors='none',edgecolors='red')
    plt.show()
