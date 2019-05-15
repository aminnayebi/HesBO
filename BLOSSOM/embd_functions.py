import math
import numpy as np
# import torch
# from BOCK_benchmarks.mnist_weight import mnist_weight
# All functions are minimization problems

def back_projection(low_obs, high_to_low, sign, bx_size):
    if len(low_obs.shape) == 1:
        low_obs = low_obs.reshape((1, low_obs.shape[0]))
    n = low_obs.shape[0]
    high_dim = high_to_low.shape[0]
    low_dim = low_obs.shape[1]
    high_obs = np.zeros((n, high_dim))
    scale = 1
    for i in range(high_dim):
        high_obs[:, i] = sign[i] * low_obs[:, high_to_low[i]] * scale
    for i in range(n):
        for j in range(high_dim):
            if high_obs[i][j] > bx_size:
                high_obs[i][j] = bx_size
            elif high_obs[i][j] < -bx_size:
                high_obs[i][j] = -bx_size
    return high_obs

class Branin(object):
    def __init__(self, act_var, high_to_low, sign, bx_size, noise_var=0):
        self.range=np.array([[-5,10],
                             [0,15]])
        self.high_to_low=high_to_low
        self.sign=sign
        self.bx_size=bx_size
        self.var = noise_var

    def scale_domain(self,x):
        # Scaling the domain
        x_copy = np.copy(x)
        x_copy = back_projection(x_copy,self.high_to_low,self.sign,self.bx_size)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                        self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def __call__(self,x,**ev):
        scaled_x=self.scale_domain(x)
        # Calculating the output
        f = [[0]]
        f[0] = [-((i[1] - (5.1 / (4 * math.pi ** 2)) * i[0] ** 2 + i[0] * 5 / math.pi - 6) ** 2 + 10 * (
                1 - 1 / (8 * math.pi)) * np.cos(i[0]) + 10) for i in scaled_x]
        f = np.transpose(f)
        return -np.squeeze(f) + np.random.normal(0,self.var), 1, dict()

class Hartmann6(object):
    def __init__(self, act_var, high_to_low, sign, bx_size, noise_var=0):
        self.range = np.array([[0, 1],
                             [0, 1],
                             [0, 1],
                             [0, 1],
                             [0, 1],
                             [0, 1]])
        self.act_var = act_var
        self.high_to_low = high_to_low
        self.sign = sign
        self.bx_size = bx_size
        self.var = noise_var

    def scale_domain(self,x):
        # Scaling the domain
        x_copy = np.copy(x)
        x_copy = back_projection(x_copy, self.high_to_low, self.sign, self.bx_size)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                    self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def __call__(self,x,**ev):
        # Calculating the output
        #Created on 08.09.2016
        # @author: Stefan Falkner
        alpha = [1.00, 1.20, 3.00, 3.20]
        A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
                      [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
                      [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
                      [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])
        P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                               [2329, 4135, 8307, 3736, 1004, 9991],
                               [2348, 1451, 3522, 2883, 3047, 6650],
                               [4047, 8828, 8732, 5743, 1091, 381]])
        scaled_x = self.scale_domain(x)
        n=len(scaled_x)
        external_sum = np.zeros((n,1))
        for r in range(n):
            for i in range(4):
                internal_sum = 0
                for j in range(6):
                    internal_sum = internal_sum + A[i, j] * (scaled_x[r, self.act_var[j]] - P[i, j]) ** 2
                external_sum[r] = external_sum[r] + alpha[i] * np.exp(-internal_sum)
        return -np.squeeze(external_sum) + np.random.normal(0,self.var), 1, dict()

class Rosenbrock(object):
    def __init__(self, act_var, high_to_low, sign, bx_size, noise_var=0):
        self.range = np.array([[-2, 2],
                               [-2, 2]])
        # self.range=np.array([[-2,1],
        #                      [1,2]])
        self.act_var = act_var
        self.high_to_low = high_to_low
        self.sign = sign
        self.bx_size = bx_size
        self.var = noise_var

    def scale_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        x_copy = back_projection(x_copy, self.high_to_low, self.sign, self.bx_size)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                    self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def __call__(self, x, **ev):
        # Calculating the output
        scaled_x = self.scale_domain(x)
        f = [[0]]
        f[0] = [-(math.pow(1 - i[self.act_var[0]], 2) + 100 * math.pow(
            i[self.act_var[1]] - math.pow(i[self.act_var[0]], 2), 2)) for i in scaled_x]
        f = np.transpose(f)
        return -np.squeeze(f) + np.random.normal(0,self.var), 1, dict()

class StybTang(object):
    def __init__(self, act_var, high_to_low, sign, bx_size, noise_var=0):
        D = len(act_var)
        a = np.ones((D, 2))
        a = a * 5
        a[:, 0] = a[:, 0] * -1
        self.range = a
        self.high_to_low=high_to_low
        self.sign=sign
        self.bx_size=bx_size
        self.var = noise_var

    def scale_domain(self,x):
        # Scaling the domain
        x_copy = np.copy(x)
        x_copy = back_projection(x_copy,self.high_to_low,self.sign,self.bx_size)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                        self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def __call__(self,x,**ev):
        scaled_x=self.scale_domain(x)
        # Calculating the output
        f = [-0.5 * np.sum(np.power(scaled_x, 4) - 16 * np.power(scaled_x, 2) + 5 * scaled_x, axis=1)]
        f = np.transpose(f)
        return -np.squeeze(f) + np.random.normal(0,self.var), 1, dict()

class MNIST(object):
    def __init__(self, act_var, high_to_low, sign, bx_size):
        D = len(act_var)
        a = np.ones((D, 2))
        a = a * 1
        a[:, 0] = a[:, 0] * -1
        self.range = a
        self.act_var = act_var
        self.high_to_low = high_to_low
        self.sign = sign
        self.bx_size = bx_size

    def scale_domain(self,x):
        # Scaling the domain
        x_copy = np.copy(x)
        x_copy = back_projection(x_copy, self.high_to_low, self.sign, self.bx_size)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                        self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def __call__(self,x, **ev):
        scaled_x = self.scale_domain(x)
        if len(scaled_x.shape) == 1:
            scaled_x = scaled_x.reshape((1, scaled_x.shape[0]))
        n = len(scaled_x)
        res = np.zeros((n, 1))
        for i in range(n):
            x = scaled_x[i]
            x = torch.from_numpy(x).type(torch.FloatTensor)
            res[i] = -mnist_weight(x)
        return -res, 1, dict()

