# -*- coding: utf-8 -*-

import numpy as np
import sys
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
            return accum
        return barrier


def brute_force_minimizer(f, search_grid):
    values = np.zeros(len(search_grid))
    for i in range(len(search_grid)):
        value = f([search_grid[i]])
        if np.isnan(value):
            value = np.inf
        values[i] = value
    index = np.argmin(values)
    #print("bfm values ", values)
    #print("index min: ", index)
    #print("grid min: ", search_grid[index])
    #print("value min: ", f([search_grid[index]]))
    return search_grid[index]


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
    def VFI_simple_solver_no_barrier(self, eps, grid, search_grid):
        V_old = np.zeros(len(grid)) 
        g_values = np.zeros(len(grid)) # policy values.
        it = 1
        anomaly = False
        while not anomaly:
            print("Iteration ", it)
            def V(y):
                return interp(y, grid, V_old)
            V_new = np.zeros(len(grid)) # Could be before...
            for i in range(len(grid)):
                #print("grid point ",i)
                x = grid[i]
                # Define the objective function, with the barrier.
                def opt_fun(y_arr):
                    y = y_arr[0]
                    #print ("y inside: ", y)
                    #print("U inside: ", self.U(x,y))
                    #print("barrier inside: ", barrier(y))
                    #print("V inside: ", V(y))
                    #print("x inside ", x)
                    return -(self.U(x,y) + self.beta*V(y))
                # Minimize the objective function.
                #print("Start point: ",start_point)
                y_opt = brute_force_minimizer(opt_fun ,search_grid)
                g_values[i] = y_opt
                #print("Y_opt: ",y_opt)
                V_new[i] = self.U(x,y_opt) + self.beta*V(y_opt) # Devuelve nan en el y_opt...
                #print("x: ", x)
                #print("Y_opt: ", y_opt)
                #print("U: ", self.U(x,y_opt)) 
                #print("V: ", V(y_opt))# V es nan...
                if np.isnan(V(y_opt)) or np.isnan(self.U(x,y_opt)):
                    print("NAN detected!")
                    sys.exit()
            #print("g", g_values)
            #print("V_old" , V_old)
            #print("V_new", V_new)
            difference = np.max(np.abs(V_old - V_new))
            print("Difference: ", difference)
            if difference < eps:
                print("Process converged!")
                break
            it += 1            
            V_old = np.copy(V_new)
        self.V = V_new
        self.g = g_values
        print("Values obtained: ", self.V)
        print("Policies obtained: ", self.g)
        return self.V, self.g




    """
    Simple Value Function Iteration.
    Considers a regular grid where the evaluation is made.
    Input:
        - eps determines the convergence criteria.
        - start, stop and num_points are used to construct the grid.
    """
    def VFI_simple_solver(self, eps, initial_mu, grid, search_grid):
        V_old = np.zeros(len(grid)) 
        g_values = np.zeros(len(grid)) # policy values.
        mu = initial_mu
        it = 1
        anomaly = False
        while not anomaly:
            print("Iteration ", it)
            def V(y):
                return interp(y, grid, V_old)
            V_new = np.zeros(len(grid)) # Could be before...
            for i in range(len(grid)):
                #print("grid point ",i)
                x = grid[i]
                barrier = self.G(x).get_log_barrier()
                # Define the objective function, with the barrier.
                def opt_fun(y_arr):
                    y = y_arr[0]
                    #print ("y inside: ", y)
                    #print("U inside: ", self.U(x,y))
                    #print("barrier inside: ", barrier(y))
                    #print("V inside: ", V(y))
                    #print("x inside ", x)
                    return -(self.U(x,y) + mu*barrier(y) + self.beta*V(y))
                # Minimize the objective function.
                #print("Start point: ",start_point)
                display = True
                y_opt = brute_force_minimizer(opt_fun ,search_grid)
                g_values[i] = y_opt
                #print("Y_opt: ",y_opt)
                V_new[i] = self.U(x,y_opt) + self.beta*V(y_opt) # Devuelve nan en el y_opt...
                #print("x: ", x)
                #print("Y_opt: ", y_opt)
                #print("U: ", self.U(x,y_opt)) 
                #print("V: ", V(y_opt))# V es nan...
                if np.isnan(V(y_opt)) or np.isnan(self.U(x,y_opt)):
                    print("NAN detected!")
                    sys.exit()
            #print("g", g_values)
            #print("V_old" , V_old)
            #print("V_new", V_new)
            difference = np.max(np.abs(V_old - V_new))
            print("Difference: ", difference)
            if difference < eps:
                print("Process converged!")
                break
            it += 1
            mu = initial_mu/it
            V_old = np.copy(V_new)
        self.V = V_new
        self.g = g_values
        print("Values obtained: ", self.V)
        print("Policies obtained: ", self.g)
        return self.V, self.g


    """
    Simple Value Function Iteration.
    Considers a regular grid where the evaluation is made.
    Input:
        - eps determines the convergence criteria.
        - start, stop and num_points are used to construct the grid.
    """
    def VFI_search_solver(self, eps, initial_mu, grid, search_grid):
        V_old = np.zeros(len(grid)) 
        g_values = np.zeros(len(grid)) # policy values.
        mu = initial_mu
        it = 1
        anomaly = False
        while not anomaly:
            print("Iteration ", it)
            def V(y):
                return interp(y, grid, V_old)
            V_new = np.zeros(len(grid)) # Could be before...
            for i in range(len(grid)):
                #print("grid point ",i)
                x = grid[i]
                barrier = self.G(x).get_log_barrier()
                # Define the objective function, with the barrier.
                def opt_fun(y_arr):
                    y = y_arr[0]
                    #print ("y inside: ", y)
                    #print("U inside: ", self.U(x,y))
                    #print("barrier inside: ", barrier(y))
                    #print("V inside: ", V(y))
                    #print("x inside ", x)
                    return -(self.U(x,y) + mu*barrier(y) + self.beta*V(y))
                # Minimize the objective function.
                # We will look for an initially feasible point.
                start_point = None
                for j in range(len(search_grid)):
                    valuation = opt_fun([search_grid[j]])
                    if  valuation != None and not np.isnan(valuation) and np.isfinite(valuation):
                        start_point = search_grid[j]
                        #print("Start point found!: ", start_point)
                        break
                #print("Valuation ", valuation)
                if start_point == None:
                    print("Process stop! No feasible starting point found.")
                    anomaly = True
                    break
                #print("Start point: ",start_point)
                display = True
                res = minimize(opt_fun, [start_point], bounds=[(search_grid[0], search_grid[-1])], options = {"disp": display})
                y_opt = res.x[0]
                #y_opt = brute_force_minimizer(opt_fun ,search_grid)
                g_values[i] = y_opt
                #print("Y_opt: ",y_opt)
                V_new[i] = self.U(x,y_opt) + self.beta*V(y_opt) # Devuelve nan en el y_opt...
                #print("x: ", x)
                #print("Y_opt: ", y_opt)
                #print("U: ", self.U(x,y_opt)) 
                #print("V: ", V(y_opt))# V es nan...
                if np.isnan(V(y_opt)) or np.isnan(self.U(x,y_opt)):
                    print("NAN detected!")
                    sys.exit()
            #print("g", g_values)
            #print("V_old" , V_old)
            #print("V_new", V_new)
            difference = np.max(np.abs(V_old - V_new))
            print("Difference: ", difference)
            if difference < eps:
                print("Process converged!")
                break
            it += 1
            mu = initial_mu/it
            V_old = np.copy(V_new)
        self.V = V_new
        self.g = g_values
        return self.V, self.g
