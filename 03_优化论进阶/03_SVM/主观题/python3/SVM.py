# -*- coding: utf-8 -*-
# @Author: IBNBlank
# @Date:   2019-02-04 15:53:36
# @Last Modified by:   IBNBlank
# @Last Modified time: 2019-02-04 20:21:05

import numpy as np
from cvxpy import *
import matplotlib.pyplot as plt



if __name__ == '__main__':

    ### load datas
    q1x_data = np.loadtxt("q1x.dat")
    q1y_data = np.loadtxt("q1y.dat")


    ### define variables
    X = q1x_data
    y = 2*(q1y_data-0.5)
    C = 1
    m = X.shape[0]
    n = X.shape[1]


    ### train svm using cvx
    # initial
    w = Variable(n)
    b = Variable()
    xi = Variable(m)
    # set and solve problem
    objective = Minimize(1/2*sum(square(w)) + C*sum(xi))
    constrains = [mul_elemwise(y,(X*w+b)) >= 1-xi, xi >= 0]
    problem = Problem(objective, constrains)
    problem.solve()


    ### visualize
    # initial
    xp = np.linspace(np.min(X[:,0]), np.max(X[:,0]), num=100)
    yp = - (w.value[0]*xp + b.value) / w.value[1]
    yp1 = - (w.value[0]*xp + b.value - 1) / w.value[1]
    yp0 = - (w.value[0]*xp + b.value + 1) / w.value[1]
    idx0 = np.argwhere(q1y_data == 0)
    idx1 = np.argwhere(q1y_data == 1)
    # plot
    plt.plot(q1x_data[idx0,0], q1x_data[idx0,1], "rx")
    plt.plot(q1x_data[idx1,0], q1x_data[idx1,1], "go")
    plt.plot(xp, yp.T, "-b")
    plt.plot(xp, yp1.T, "--g")
    plt.plot(xp, yp0.T, "--r")
    plt.show()