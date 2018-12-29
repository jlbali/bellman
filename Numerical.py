#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 19:20:58 2018

@author: lbali
"""

import numpy as np
from scipy.optimize import minimize
import sys

"""
Some ideas based on
http://www.stat.cmu.edu/~ryantibs/convexopt-F15/lectures/15-barr-method.pdf

"""

class BarrierSolver:
    
    
    def __init__(self, f, phi, mu):
        assert mu > 1
        self.f = f
        self.phi = phi
        self.mu = mu
        
    
    def slow_solve(self, stop_criteria, x0, t0):
        t = t0
        xk = x0
        traj = []
        grad_traj = []
        while t <= stop_criteria:
            #print("t: ", t)
            #print("xk: ", xk)
            def opt_fun(x):
                return t*self.f(x) + self.phi(x)
            display = False
            #res = minimize(opt_fun, xk, method="BFGS", options={'gtol': 1e-05, 'norm': np.inf, 'eps': 1.4901161193847656e-20, 'maxiter': None, 'disp': display, 'return_all': False})
            #res = minimize(opt_fun, xk, method="SLSQP", options={'eps': 1.4901161193847656e-10,  'disp': display})
            #res = minimize(opt_fun, xk, method="BFGS", options={'gtol': 1e-20, 'norm': np.inf, 'maxiter': None, 'disp': display, 'return_all': False})
            #res = minimize(opt_fun, xk)
            grad_eps = 1.0e-4 # Feo, con valores altos se queda en el x0 como optimo, con valores bajos se va completamente de rango, muy raro, es casi al reves que antes.
            while True:
                #res = minimize(opt_fun, xk, method="BFGS", options={'gtol': 1e-05, 'norm': np.inf, 'eps': grad_eps, 'maxiter': None, 'disp': display, 'return_all': False})
                res = minimize(opt_fun, xk, method="SLSQP", options={'eps': grad_eps,  'disp': display})
                x = res.x
                y = opt_fun(x)
                #print("y" , y)
                #print("xk" , xk)
                #print("grad_eps: ", grad_eps)
                if not np.isnan(y) and not np.isinf(y):
                    xk = x
                    break
                grad_eps = grad_eps / 2.0
                if grad_eps == 0.0:
                    sys.exit(1)
            traj.append(xk)
            grad_traj.append(grad_eps)
            t = t*self.mu
        #print("Traj: ", traj)
        #print("grad_traj: ", grad_traj)
        return xk
        
    def solve(self, stop_criteria, x0, t0):
        t = t0
        xk = x0
        traj = []
        print("Initial: ", x0)
        while t <= stop_criteria:
            #print("t: ", t)
            #print("xk: ", xk)
            def opt_fun(x):
                return t*self.f(x) + self.phi(x)
            display = False
            #res = minimize(opt_fun, xk, method="BFGS", options={'gtol': 1e-05, 'norm': np.inf, 'eps': 1.4901161193847656e-20, 'maxiter': None, 'disp': display, 'return_all': False})
            res = minimize(opt_fun, xk, method="SLSQP", options={'eps': 1.4901161193847656e-18,  'disp': display})
            #res = minimize(opt_fun, xk, method="BFGS", options={'gtol': 1e-20, 'norm': np.inf, 'maxiter': None, 'disp': display, 'return_all': False})
            #res = minimize(opt_fun, xk)
            xk = res.x
            print("xk: ", xk)
            traj.append(xk)
            t = t*self.mu
        print("Traj: ", traj)
        return xk



