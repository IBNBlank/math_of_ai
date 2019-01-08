# -*- coding: utf-8 -*-
# @Author: IBNBlank
# @Date:   2019-01-01 20:43:41
# @Last Modified by:   IBNBlank
# @Last Modified time: 2019-01-02 21:37:53

import matplotlib.pyplot as plt
import numpy as np
import math

def BP_BACK(training_example, eta):
    ### 初始化 ###
    sigma = []
    (m, n) = np.shape(training_example)

    w = np.random.random((2,3)) - 0.5
    v = np.random.random((3,2)) - 0.5
    u = np.random.random((2,3)) - 0.5

    ### 取样 ###
    for num in range(n):
        one_sample = training_example[:,num]
        x = np.mat(one_sample[0:3])
        x = x.T
        y = np.mat(one_sample[3:5])
        y = y.T


        ### 计算各层输出 ###
        hidden1 = np.mat([[0.],[0.]])
        net2 = w * x
        for i in range(2):
            hidden1[i] = 1 / (1 + math.exp(0-net2[i]))

        hidden2 = np.mat([[0.],[0.],[0.]])
        net3 = v * hidden1
        for i in range(3):
            hidden2[i] = 1 / (1 + math.exp(0-net3[i]))

        o = np.mat([[0.],[0.]])
        net4 = u * hidden2
        for i in range(2):
            o[i] = 1 / (1 + math.exp(0-net4[i]))


        ### 计算δ ###
        delta3 = np.mat([[0.],[0.]])
        for i in range(2):
            delta3[i] = (y[i]-o[i]) * o[i] * (1-o[i])

        delta2 = np.mat([[0.],[0.],[0.]])
        for i in range(3):
            delta2[i] = hidden2[i] * (1-hidden2[i]) * u[:,i] * delta3

        delta = np.mat([[0.],[0.]])
        for i in range(2):
            delta[i] = hidden1[i] * (1-hidden1[i]) * v[:,i] * delta2


        ### 更新权值 ###
        for i in range(2):
            for j in range(3):
                u[i,j] = u[i,j] + eta*delta3[i]*hidden2[j]

        for i in range(3):
            for j in range(2):
                v[i,j] = v[i,j] + eta*delta2[i]*hidden1[j]

        for i in range(2):
            for j in range(3):
                w[i,j] = w[i,j] + eta*delta[i]*x[j]


        ### 记录误差 ###
        e = o - y
        loss = e.T*e
        sigma.append(loss[0,0])


    ### 绘制误差曲线 ###
    plt.figure()
    plt.plot(sigma)
    plt.show()