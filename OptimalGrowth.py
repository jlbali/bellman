#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 20:54:04 2018

@author: lbali
"""

import numpy as np
from DeterministicBellman import *
from scipy import interp
from scipy.optimize import minimize
import sys

from Numerical import *

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
def build_power_utility(gamma):
    def U(k):
        if gamma == 0:
            return np.log(k)
        else:
            return k**gamma
    return U

"""
Builds an utility function as the power of gamma.
When sigma = 1, it considers the log.
"""
def build_sigma_utility(sigma):
    def U(k):
        if sigma == 1:
            return np.log(k)
        else:
            return (k**(1-sigma)-1)/(1-sigma)
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
        self.beta = beta
        def f(k): # Available goods at the beginning of period given capital k.
            return F(k) + (1.0 - delta)*k
        self.f = f
        def U2(x,y):
            return U(self.f(x) - y)
        self.U = U2

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
    Uses the general Bellman framework.
    """
    def VFI_Bellman(self, grid, search_grid, eps):
        def U(x,y):
            return self.U(self.f(x) - y)
        def G(x):
            def c0(y): # constraint y >= 0
                return y
            def c1(y): # contraint f(x) - y >= 0
                return self.f(x) - y
            return InequalityConstraintSet([c0,c1])
        
        bellman = DeterministicBellman1D(U, G, self.beta)
        #V,g = bellman.VFI_simple_solver(eps, 1.0, grid, search_grid)
        V,g = bellman.VFI_simple_solver_no_barrier(eps, grid, search_grid)
        self.V = V
        self.g = g
        self.grid = grid
        return V,g


    def VFI_grid_search(self, grid, eps):
        """
        Value Function Iteration, directly solving it.
        """
        V_old = np.zeros(len(grid)) 
        g_values = np.zeros(len(grid)) # policy values.
        stop = False
        while not stop:
            V_new = np.zeros(len(grid)) # Could be before...
            for i in range(len(grid)):
                x = grid[i]
                values = np.zeros(len(grid))
                for j in range(len(grid)):
                    y = grid[j]
                    if 0 <= y <= self.f(x): 
                        values[j] = self.U(x,y) + self.beta*V_old[j]
                    else:
                        values[j] = -np.inf
                j_opt = np.argmax(values)
                y_opt = grid[j_opt]
                g_values[i] = y_opt
                V_new[i] = np.max(values)
            difference = np.max(np.abs((V_old - V_new)/V_old))
            print("Difference: ", difference)
            if difference < eps:
                print("Process converged!")
                break
            V_old = np.copy(V_new)
        self.V = V_new
        self.g = g_values
        self.grid = grid
        return self.V, self.g


    def VFI_interpolate(self, interp_nodes, eps):
        """
        Value Function Iteration, directly solving it.
        """
        V_old = np.zeros(len(interp_nodes)) 
        g_values = np.zeros(len(interp_nodes)) # policy values.
        stop = False
        while not stop:
            def V(y):
                return interp(y, interp_nodes, V_old)
            V_new = np.zeros(len(interp_nodes)) # Could be before...
            for i in range(len(interp_nodes)):
                x = interp_nodes[i]
                def opt_fun(y_arr):
                    y = y_arr[0]
                    return -(self.U(x,y) + self.beta*V(y))
                display = True
                res = minimize(opt_fun, [(0 + self.f(x))/2], bounds=[(0, self.f(x))], options = {"disp": display})
                y_opt = res.x[0]
                g_values[i] = y_opt
                V_new[i] = self.U(x,y_opt) + self.beta*V(y_opt)
                if np.isnan(V(y_opt)) or np.isnan(self.U(x,y_opt)):
                    print("NAN detected!")
                    sys.exit()
            difference = np.max(np.abs((V_old - V_new)/V_old)) 
            print("Difference: ", difference)
            if difference < eps:
                print("Process converged!")
                break
            V_old = np.copy(V_new)
        self.V = V_new
        self.g = g_values
        self.grid = interp_nodes
        return self.V, self.g


    def VFI_interpolate_log_barrier(self, interp_nodes, eps, mu, stop_criteria, t0=10.0):
        """
        Value Function Iteration, directly solving it.
        """
        V_old = np.zeros(len(interp_nodes)) 
        g_values = np.zeros(len(interp_nodes)) # policy values.
        stop = False
        while not stop:
            #mu = initial_mu / it
            def V(y):
                return interp(y, interp_nodes, V_old)
            V_new = np.zeros(len(interp_nodes)) # Could be before...
            for i in range(len(interp_nodes)):
                x = interp_nodes[i]
                def f(y_arr):
                    y = y_arr[0]
                    return -(self.U(x,y)  + self.beta*V(y))
                def phi(y_arr):
                    y = y_arr[0]
                    return -(np.log(y) + np.log(self.f(x) - y))
                solver = BarrierSolver(f, phi, mu)                
                y0 = (0 + self.f(x))/2
                y_opt = solver.solve(stop_criteria, y0, t0)[0]
                g_values[i] = y_opt
                V_new[i] = self.U(x,y_opt) + self.beta*V(y_opt)
                if np.isnan(V(y_opt)) or np.isnan(self.U(x,y_opt)):
                    print("NAN detected!")
                    sys.exit()
            difference = np.max(np.abs((V_old - V_new)/V_old)) 
            print("Difference: ", difference)
            if difference < eps:
                print("Process converged!")
                break
            V_old = np.copy(V_new)
        self.V = V_new
        self.g = g_values
        self.grid = interp_nodes
        return self.V, self.g


    def VFI_interpolate_log_barrier_bound(self, interp_nodes, eps, initial_mu = 1.0):
        """
        Value Function Iteration, directly solving it.
        """
        V_old = np.zeros(len(interp_nodes)) 
        g_values = np.zeros(len(interp_nodes)) # policy values.
        stop = False
        it = 1
        while not stop:
            mu = initial_mu / it
            def V(y):
                return interp(y, interp_nodes, V_old)
            V_new = np.zeros(len(interp_nodes)) # Could be before...
            for i in range(len(interp_nodes)):
                x = interp_nodes[i]
                def opt_fun(y_arr):
                    y = y_arr[0]
                    return -(self.U(x,y) + mu*np.log(y) + mu*np.log(self.f(x) - y) + self.beta*V(y))
                display = True
                res = minimize(opt_fun, [(0 + self.f(x))/2], bounds=[(0, self.f(x))], options = {"disp": display})
                #res = minimize(opt_fun, [(0 + self.f(x))/2], options = {"disp": display})
                #res = minimize(opt_fun, [(0 + self.f(x))/2])
                y_opt = res.x[0]
                g_values[i] = y_opt
                V_new[i] = self.U(x,y_opt) + self.beta*V(y_opt)
                if np.isnan(V(y_opt)) or np.isnan(self.U(x,y_opt)):
                    print("NAN detected!")
                    sys.exit()
            #print("g: ", g_values)
            #print("V: ", V_new)
            difference = np.max(np.abs((V_old - V_new)/V_old)) 
            print("Difference: ", difference)
            if difference < eps:
                print("Process converged!")
                break
            V_old = np.copy(V_new)
            it += 1
        self.V = V_new
        self.g = g_values
        self.grid = interp_nodes
        return self.V, self.g


            

    def get_investment_plan(self, initial_capital, n_periods=100):
        investments = np.zeros(n_periods)
        def V(k):
            return interp(k, self.grid, self.g)
        investments[0] = initial_capital
        for i in range(1, len(investments)):
            investments[i] = V(investments[i-1])
        return investments

