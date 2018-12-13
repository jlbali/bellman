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
    #delta = 0
    U = build_utility(gamma)
    F = build_prod_function(alfa)
    optim_growth = OptimalGrowth(U,F, delta, beta)
    #grid = np.linspace(0.0, 1.0, 50) # No puede empezar en capital 0, indefiniciones por el log...
    grid = np.linspace(0.0001, 1.0, 50) # No puede empezar en capital 0, indefiniciones por el log...
    search_grid = np.linspace(0.0, 1.0, 500) # No puede empezar en capital 0, indefiniciones por el log...
    V,g = optim_growth.VFI(grid, search_grid, 0.001)
    plt.plot(grid, V)
    plt.show()
    plt.plot(grid,g)
    plt.show()
    investments = optim_growth.get_investment_plan(0.5) # Plan de inversiones con capital inicial 0.5
    plt.plot(range(len(investments)), investments)
    plt.show()
    print("Investments: ", investments) # Se estaciona muy rapido el plan de inversion...
    

test1()

# Ver bienm gat varias indefiniciones por llamados a logaritmos...

# Considerar el uso de barreras de maximo en cero quizas...



# Se esta evaluando en punto que no son feasibles.

# Se optiene siempre el mismo y_opt, que es el primero no cero... Raro.
# Seguir analizando...

# Raro, salen policies con cero... sin importar el delta de depreciacion.
