# -*- coding: utf-8 -*-

import numpy as np
from scipy import interp
from scipy.optimize import minimize



"""
Serves to model a solution to the following DP 

V(x) = sup_{y \in G(x)} U(x,y) + \beta V(y)

We assume that G(x) is specified as a list of scalar functions {c_i}
where the restriction is c_i(x) >= 0.

"""



class InequalityConstraintSet:
    
    def __init__(self, cs):
        self.cs = cs
    
    def is_feasible(self, x):
        for i in range(len(self.cs)):
            if self.cs[i](x) < 0:
                return False
        return True
    
    def get_log_barrier(self):
        def barrier(x):
            accum = 0.0
            for i in range(len(self.cs)):
                accum += np.log(self.cs[i](x))
        return barrier



class DeterministicBellman1D:
    """
    - U is a bivariate "utility-value" function.
    - G is an Inequality Constraint Set generator function.
    - beta is the discount factor.
    """
    def __init__(self, U, G, beta):
        self.U = U
        self.G = G
        self.beta = beta
    
    """
    Simple Value Function Iteration.
    Considers a regular grid where the evaluation is made.
    Input:
        - eps determines the convergence criteria.
        - start, stop and num_points are used to construct the grid.
    """
    def VFI_simple_solver(self, eps, initial_mu, grid):
        V_values = np.zeros(len(grid)) 
        g_values = np.zeros(len(grid)) # policy values.
        mu = initial_mu
        i = 1
        while True:
            def V(y):
                return interp(y, grid, V_values)
            new_V = np.zeros(len(grid))
            for i in range(len(grid)):
                x = grid[i]
                barrier = self.G(x).get_log_barrier()
                # Define the objective function, with the barrier.
                def opt_fun(y_arr):
                    y = y_arr[0]
                    return -(self.U(x,y) + mu*barrier(y) + self.beta*V(y))
                # Minimize the objective function. Start in the middle of the grid.
                #print(barrier(1.0))
                res = minimize(opt_fun, [(grid[0]+grid[-1])/2])
                y_opt = res.x[0]
                g_values[i] = y_opt
                new_V[i] = self.U(x,y_opt) + self.beta*V(y_opt)
            if np.max(np.abs(V_values - new_V)) < eps:
                break
            i += 1
            mu = initial_mu/i
            V_values = new_V
        self.V = new_V
        self.g = g_values
        return self.V, self.g

