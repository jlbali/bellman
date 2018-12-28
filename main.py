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
import warnings

def test1():
    alfa = 1/3
    gamma = 0
    beta = 0.9
    delta = 1
    #delta = 0
    U = build_power_utility(gamma)
    F = build_prod_function(alfa)
    optim_growth = OptimalGrowth(U,F, delta, beta)
    #grid = np.linspace(0.0, 1.0, 50) # No puede empezar en capital 0, indefiniciones por el log...
    grid = np.linspace(0.0001, 1.0, 50) # No puede empezar en capital 0, indefiniciones por el log...
    search_grid = np.linspace(0.0, 1.0, 500) # No puede empezar en capital 0, indefiniciones por el log...
    V,g = optim_growth.VFI_Bellman(grid, search_grid, 0.001)
    plt.plot(grid, V)
    plt.show()
    plt.plot(grid,g)
    plt.show()
    investments = optim_growth.get_investment_plan(0.5) # Plan de inversiones con capital inicial 0.5
    plt.plot(range(len(investments)), investments)
    plt.show()
    print("Investments: ", investments) # Se estaciona muy rapido el plan de inversion...
    
# Using the General Bellman framework.
# PELIGRO! La version de optimizados del general framework que usa en el fondo no tiene restricciones..
# Salen naturalmente de alguna manera por la restriccion logaritmica de la utilidad, que hace que no se tomen en cuenta.
def test2():
    alfa = 1/3
    #sigma = 1
    sigma = 0.5 # No parece tener mucho efecto en el plan de inversion.
    beta = 0.9
    delta = 1
    U = build_sigma_utility(sigma)
    F = build_prod_function(alfa)
    optim_growth = OptimalGrowth(U,F, delta, beta)
    #grid = np.linspace(0.0, 1.0, 50) # No puede empezar en capital 0, indefiniciones por el log...
    grid = np.linspace(0.0001, 1.0, 50) # No puede empezar en capital 0, indefiniciones por el log...
    search_grid = np.linspace(0.0, 1.0, 500) # No puede empezar en capital 0, indefiniciones por el log...
    V,g = optim_growth.VFI_Bellman(grid, search_grid, 0.001)
    plt.plot(grid, V)
    plt.show()
    plt.plot(grid,g)
    plt.show()
    investments = optim_growth.get_investment_plan(0.5) # Plan de inversiones con capital inicial 0.5
    plt.plot(range(len(investments)), investments)
    plt.show()
    print("Investments: ", investments) # Se estaciona muy rapido el plan de inversion...


# "Manually" solving the Optimal Growth without the General Bellman framework.
def test3():
    alfa = 1/3
    sigma = 1
    #sigma = 0.5 # No parece tener mucho efecto en el plan de inversion.
    beta = 0.9
    delta = 1
    U = build_sigma_utility(sigma)
    F = build_prod_function(alfa)
    optim_growth = OptimalGrowth(U,F, delta, beta)
    #grid = np.linspace(0.0, 1.0, 50) # No puede empezar en capital 0, indefiniciones por el log...
    grid = np.linspace(0.0001, 1.0, 50) # No puede empezar en capital 0, indefiniciones por el log...
    V,g = optim_growth.VFI_manual(grid,  0.001)
    plt.plot(grid, V)
    plt.show()
    plt.plot(grid,g)
    plt.plot(np.linspace(0,1,50), np.linspace(0,1,50))
    plt.show()
    investments = optim_growth.get_investment_plan(0.5) # Plan de inversiones con capital inicial 0.5
    plt.plot(range(len(investments)), investments)
    plt.show()
    print("Investments: ", investments) # Se estaciona muy rapido el plan de inversion...
# Hay un pounto fijo cerca de 0.163, tiene sentido que se estacione por esa zona.

# Comparison of both schemes (manual solving and with Deterministic Bellman)...
def test4():
    alfa = 1/3
    sigma = 1
    #sigma = 0.5 # No parece tener mucho efecto en el plan de inversion.
    beta = 0.9
    delta = 1
    U = build_sigma_utility(sigma)
    F = build_prod_function(alfa)
    optim_growth = OptimalGrowth(U,F, delta, beta)
    grid = np.linspace(0.0001, 1.0, 50) # No puede empezar en capital 0, indefiniciones por el log...
    search_grid = np.linspace(0.0, 1.0, 500) # No puede empezar en capital 0, indefiniciones por el log...
    V1,g1 = optim_growth.VFI_manual(grid,  0.001)
    V2,g2 = optim_growth.VFI_Bellman(grid, search_grid, 0.001)
    plt.plot(grid, g1)
    plt.plot(grid, g2)
    plt.show()
    

# Comparison of both schemes (manual solving and with Deterministic Bellman)...
# Por ahora, los nodos de interpolacion coinciden con los de busqueda...
def test5():
    warnings.filterwarnings("ignore")
    alfa = 1/3
    sigma = 1
    #sigma = 0.5 # No parece tener mucho efecto en el plan de inversion.
    beta = 0.9
    delta = 1
    U = build_sigma_utility(sigma)
    F = build_prod_function(alfa)
    optim_growth = OptimalGrowth(U,F, delta, beta)
    grid = np.linspace(0.0001, 1.0, 50) # No puede empezar en capital 0, indefiniciones por el log...
    V1,g1 = optim_growth.VFI_interpolate(grid, grid,  0.001)
    V2,g2 = optim_growth.VFI_grid_search(grid,  0.001)
    V3,g3 = optim_growth.VFI_interpolate_log_barrier(grid, grid, 0.001, 1.0) # No anda bien...
    V4,g4 = optim_growth.VFI_interpolate_log_barrier_bound(grid, grid, 0.001, 1.0)
    plt.plot(grid, g1)
    plt.plot(grid, g2)
    #plt.plot(grid, g3) # <-- Anda mal, devuelve negativos...
    plt.plot(grid, g4) # Da bastante distinto a los dos anteriores...
    plt.show()



#test1()
#test2()
#test3()
#test4() # Perfecta coincidencia.

test5()

# Ver bienm gat varias indefiniciones por llamados a logaritmos...

# Considerar el uso de barreras de maximo en cero quizas...



# Se esta evaluando en punto que no son feasibles.

# Se optiene siempre el mismo y_opt, que es el primero no cero... Raro.
# Seguir analizando...

# Raro, salen policies con cero... sin importar el delta de depreciacion.


# Por ahora, esta marcada para usar la version sin barrera.

# No es conveniente usar el no_barrier ya que en el fondo no está imponiendo ningùn
# tipo de restricción!!

# Test 3 genera un warning... Igualmente, esta andando igual que el test2. Quizas es buena marca...
