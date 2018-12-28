#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 19:20:58 2018

@author: lbali
"""

import numpy as np
from scipy.optimize import minimize



class BarrierSolver:
    
    
    def __init__(self, f, phi, mu):
        assert mu > 1
        self.f = f
        self.phi = phi
        self.mu = mu
        
    
    def solve(self, stop_criteria, x0, t0):
        t = t0
        xk = x0
        while t <= stop_criteria:
            #print("t: ", t)
            #print("xk: ", xk)
            def opt_fun(x):
                return t*self.f(x) + self.phi(x)
            display = False
            #res = minimize(opt_fun, xk, method="BFGS", options={'gtol': 1e-05, 'norm': np.inf, 'eps': 1.4901161193847656e-20, 'maxiter': None, 'disp': display, 'return_all': False})
            res = minimize(opt_fun, xk, method="SLSQP", options={'eps': 1.4901161193847656e-20,  'disp': display})
            xk = res.x
            t = t*self.mu
        return xk
        



