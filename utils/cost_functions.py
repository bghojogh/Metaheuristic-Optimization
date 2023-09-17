"""
This is an implementation of important benchmark cost functions.
For the formula of important bencahmark costs, you can visit the links below.
https://arxiv.org/pdf/1807.01844.pdf
https://arxiv.org/pdf/1809.09284.pdf
"""
import numpy as np
from typing import List


def sphere(x: List[float]) -> float:
    return sum([xi**2 for xi in x])

def schwefel(x: List[float]) -> float:
    d = len(x)
    f = 418.9829 * d
    for xi in x:
        f = f - (xi * np.sin(np.sqrt(np.abs(xi))))
    return f

def schaffer(x: List[float]) -> float:
    d = len(x)
    f = 0
    for i in range(d-1):
        f = f + (x[i]**2 + x[i+1]**2)**0.25 * ((np.sin(50 * (x[i]**2 + x[i+1]**2)**0.1))**2 + 1)
    return f

def griewank(x: List[float]) -> float:
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

def func7(x: List[float]) -> float:
    d = len(x)
    f1 = 0
    f2 = 1
    for i in range(d):
        f1 = f1 + ((x[i]**2) / 4000)
        f2 = f2 * np.cos(x[i] / np.sqrt(i + 1))
    f = f1 - f2 + 1 - 180
    return f

def func8(x: List[float]) -> float:
    d = len(x)
    f1 = 0
    f2 = 0
    for i in range(d):
        f1 = f1 + (x[i]**2)
    f1 = f1 / d
    for i in range(d):
        f2 = f2 + np.cos(2 * np.pi * x[i])
    f2 = f2 / d
    f = (-20 * np.exp(-0.2 * np.sqrt(f1))) - np.exp(f2) + 20 + np.exp(1) - 140
    return f

def func9(x: List[float]) -> float:
    d = len(x)
    f = -300
    for i in range(d):
        f = f + (x[i]**2 - (10 * np.cos(2 * np.pi * x[i])) + 10)
    return f

def func11(x: List[float]) -> float:
    d = len(x)
    f1 = 0
    f2 = 0
    for i in range(d):
        f11 = 0
        for k in range(21):
            f11 = f11 + ((0.5**k) * np.cos(2 * np.pi * (3**k) * (x[i] + 0.5)))
        f1 = f1 + f11
    for k in range(21):
        f2 = f2 + ((0.5**k) * np.cos(2 * np.pi * (3**k) * 0.5))
    f = f1 - (d * f2) + 90
    return f