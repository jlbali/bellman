#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:45:03 2018

@author: lbali
"""

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from OptimalGrowth import *


def test1():
    alfa = 1/3
    gamma = 0
    beta = 0.9
    delta = 1
    U = build_utility(gamma)
    F = build_prod_function(alfa)
    optim_growth = OptimalGrowth(U,F, delta, beta)
    grid = np.linspace(0.001, 0.999, 50)
    optim_growth.set_initial_capital(1.0)
    V,g = optim_growth.VFI(grid, 0.001)
    plt.plot(grid, V)
    plt.show()
    plt.plot(grid,g)
    plt.show()
    

test1()

# Ver bienm gat varias indefiniciones por llamados a logaritmos...

# Considerar el uso de barreras de maximo en cero quizas...

