# -*- coding: utf-8 -*-
# @Author: IBNBlank
# @Date:   2018-12-30 23:32:04
# @Last Modified by:   IBNBlank
# @Last Modified time: 2018-12-31 10:54:24

import numpy as np

def gradient_method_quadratic(A, b, x0, epsilon):
    x = x0
    iter_num = 0
    grad = 2*(A*x + b)

    while np.linalg.norm(grad) > epsilon:
        iter_num += 1
        t = np.linalg.norm(grad)**2 / (2*grad.T*A*grad)
        x -= grad*t
        grad = 2*(A*x + b)
        fun_val = x.T*A*x + 2*b.T*x
        print('iter_number = {0:3d} norm_grad = {1:.6f} fun_val = {2:.6f} \n'.format(iter_num, np.linalg.norm(grad), fun_val[0,0]))

    return x, fun_val

if __name__ == '__main__':
    A = np.mat([[1,0], [0,2]])
    b = np.mat([[0], [0]])
    x0 = np.mat([[2.0], [1.0]])
    epsilon = 1e-5
    x, fun_val = gradient_method_quadratic(A, b, x0, epsilon)