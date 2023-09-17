"""The implementation of important benchmark cost functions"""
import numpy as np

def sphere(x):
    return sum([xi**2 for xi in x])

def schwefel(x):
    d = len(x)
    f = 418.9829 * d
    for xi in x:
        f = f - (xi * np.sin(np.sqrt(np.abs(xi))))
    return f