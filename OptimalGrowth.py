#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 20:54:04 2018

@author: lbali
"""

import numpy as np
from DeterministicBellman import *
from scipy import interp

"""
Based on Stokey and Lucas (1989)
Chapter 2.
Labor fixed at 1.

Input:
- U: utility function (depends on consumption)
- F: production function (depends on capital)
- delta: depreciation factor (0 < delta < 1)
- beta: discount factor (0 < beta < 1)


"""


"""
Builds an utility function as the power of gamma.
When gamma = 0, it considers the log.
"""
def build_utility(gamma):
    def U(k):
        if gamma == 0:
            return np.log(k)
        else:
            return k**gamma
    return U
"""
Builds a "Cobb-Doublas" production function, taking as a parameter the alpha.
(no labor is considered in this model)
"""

def build_prod_function(alpha):
    def F(k):
        return k**alpha
    return F


########### DETERMINISTIC OPTIMAL GROWTH PROBLEM ################################

class OptimalGrowth:
    
    
    def __init__(self, U, F, delta, beta):
        assert delta >= 0 and delta <= 1
        assert beta < 1
        self.U = U
        self.beta = beta
        def f(k): # Available goods at the beginning of period given capital k.
            return F(k) + (1.0 - delta)*k
        self.f = f
    """
    Determine if a sequence k's constitutes a truncated feasible solution.
    It must comply with 0 <= k_{t+1} <= f(k_t)
    (eq 4 of chapter 2 of Stokey and Lucas (1989))
    """
    def feasible_solution(self, ks):
        if ks[0] < 0:
            return False
        for i in range(1, len(ks)):
            if ks[i] < 0 or ks[i] > self.f(ks[i-1]):
                return False
        return True
    
    """
    Evaluates the total disounted utility for a truncated feasible solution.
    (eq 3 of chapter 2 of Stokey and Lucas (1989))
    """
    def compute_total_utility(self, ks):
        assert self.feasible_solution(ks)
        value = 0.0
        for i in range(len(ks) - 1):
            value += (self.beta**i)*self.U(f(ks[i]) - ks[i+1])
        return value
    

#    def set_initial_capital(self, k0):
#        assert k0 > 0
#        self.k0 = k0

    """
    Value Function Iteration on a regular grid.
    """
    # CHECK! NO RESTRICTION YET AND f is not taken care of.
    def VFI(self, grid, search_grid, eps):
        def U(x,y):
            return self.U(self.f(x) - y)
        def G(x):
            def c0(y): # constraint y >= 0
                return y
            def c1(y): # contraint f(x) - y >= 0
                return self.f(x) - y
            return InequalityConstraintSet([c0,c1])
        
        bellman = DeterministicBellman1D(U, G, self.beta)
        V,g = bellman.VFI_simple_solver(eps, 1.0, grid, search_grid)
        self.V = V
        self.g = g
        self.grid = grid
        return V,g


    def get_investment_plan(self, initial_capital, n_periods=100):
        investments = np.zeros(n_periods)
        def V(k):
            return interp(k, self.grid, self.V)
        investments[0] = initial_capital
        for i in range(1, len(investments)):
            investments[i] = V(investments[i-1])
        return investments

