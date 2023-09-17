"""
The implementation of important benchmark cost functions.
For the formula of important bencahmark costs, you can see these links:
https://arxiv.org/pdf/1807.01844.pdf
https://arxiv.org/pdf/1809.09284.pdf
"""
import numpy as np


def sphere(x):
    return sum([xi**2 for xi in x])

def schwefel(x):
    d = len(x)
    f = 418.9829 * d
    for xi in x:
        f = f - (xi * np.sin(np.sqrt(np.abs(xi))))
    return f

def schaffer(x):
    d = len(x)
    f = 0
    for i in range(d-1):
        f = f + (x[i]**2 + x[i+1]**2)**0.25 * ((np.sin(50 * (x[i]**2 + x[i+1]**2)**0.1))**2 + 1)
    return f

def griewank(x):
    d = len(x)
    f1 = 0
    for i in range(d):
        f1 = f1 + x[i]**2
    f1 = f1 / 4000
    f2 = 1
    for i in range(d):
        f2 = f2 * np.cos(x[i] / ((i+1)**0.5))
    f = f1 - f2 + 1
    return f