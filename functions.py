import numpy as np
from scipy.stats import norm
import random

def bernoulli():
    return random.uniform(0, 1)

def monte_carlo(ppf):
    x = bernoulli()
    return ppf(x)

def normal_ppf(m, s):
    def helper(t):
        return norm.ppf(t, m, s)
    return helper

def normal_pdf(m, s):
    def helper(x):
        return norm.pdf(x, m, s)
    return helper

def normalize(vector):
    return vector/np.linalg.norm(vector)

def random_vector():
    return np.array([bernoulli(), bernoulli(), bernoulli()])

def extend_to_O(direction):
    direction = normalize(direction)
    M = np.array([direction, random_vector(), random_vector()]).transpose()
    q, r = np.linalg.qr(M)
    return r.item(0)*np.dot(q,np.array([[0,0,1],[0,1,0],[1,0,0]]))

def transport_to(direction, vector):
    return np.dot(extend_to_O(direction), vector)

def transport_from(direction, vector):
    return np.dot(extend_to_O(direction).transpose(), vector)

def lens_to_hemisphere(x,y):
    return np.array([x,y,1])*(1/(x**2 + y**2 + 1)**(0.5))

def hemisphere_to_lens(direction):
    return (direction.item(0)/direction.item(2), direction.item(1)/direction.item(2))