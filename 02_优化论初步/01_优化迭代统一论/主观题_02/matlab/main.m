clear all
clc;

A = [1 0;0 2]
b = [0;0]
x0 = [2;1]
epsilon = 1e-5
gradient_method_quadratic(A, b, x0, epsilon)