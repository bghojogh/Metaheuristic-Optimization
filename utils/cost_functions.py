"""The implementation of important benchmark cost functions"""
import numpy as np

def sphere(x):
    return sum([xi**2 for xi in x])

def schwefel(x):
    d = len(x)
    x1, x2 = x[:2]
    return (418.9829 * d - (x1 * (np.sin(np.sqrt(np.abs(x1)))) + (x2 * np.sin(np.sqrt(np.abs(x2))))))