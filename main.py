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
    nodes = np.linspace(0.0001, 1.0, 10)
    V1,g1 = optim_growth.VFI_interpolate(grid,  0.001)
    V2,g2 = optim_growth.VFI_grid_search(grid,  0.001)
    V3,g3 = optim_growth.VFI_interpolate_log_barrier(grid, 0.001, 1.5, 1e8, 1.0) # No hace mucha diferencia, termina devolviendo lo mismo....
    #V4,g4 = optim_growth.VFI_interpolate_log_barrier(grid, 0.001, 1.0e-1)
    #V5,g5 = optim_growth.VFI_interpolate_log_barrier(nodes, 0.001, 1.0e-1)
    #V4,g4 = optim_growth.VFI_interpolate_log_barrier_bound(grid, grid, 0.001, 1.0)
    plt.plot(grid, g1)
    plt.plot(grid, g2)
    plt.plot(grid, g3) # <-- Da distinto a g1 y g2.
    #plt.plot(grid, g4) # <-- Da distinto a g1 y g2.
    #plt.plot(nodes, g5) # <-- Da distinto a g1 y g2.
    plt.show()

# Para el log barrier se redujo la cantidad de nodos...
# Es lento al principio pero despues toma velocidad...
def test6():
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
    nodes = np.linspace(0.0001, 1.0, 10)
    V1,g1 = optim_growth.VFI_interpolate(grid,  0.001)
    V2,g2 = optim_growth.VFI_grid_search(grid,  0.001)
    V3,g3 = optim_growth.VFI_interpolate_log_barrier(nodes, 0.001, 2.0, 1e8, 1.0) # No hace mucha diferencia, termina devolviendo lo mismo....
    #V4,g4 = optim_growth.VFI_interpolate_log_barrier(grid, 0.001, 1.0e-1)
    #V5,g5 = optim_growth.VFI_interpolate_log_barrier(nodes, 0.001, 1.0e-1)
    #V4,g4 = optim_growth.VFI_interpolate_log_barrier_bound(grid, grid, 0.001, 1.0)
    plt.plot(grid, g1)
    plt.plot(grid, g2)
    plt.plot(nodes, g3) # <-- Da distinto a g1 y g2.
    #plt.plot(grid, g4) # <-- Da distinto a g1 y g2.
    #plt.plot(nodes, g5) # <-- Da distinto a g1 y g2.
    plt.show()


def test7():
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
    nodes = np.linspace(0.0001, 1.0, 10)
    V1,g1 = optim_growth.VFI_interpolate(nodes,  0.001)
    V2,g2 = optim_growth.VFI_grid_search(nodes,  0.001)
    V3,g3 = optim_growth.VFI_interpolate_log_barrier(nodes, 0.001, 2.0, 1e8, 1.0) # No hace mucha diferencia, termina devolviendo lo mismo....
    #V4,g4 = optim_growth.VFI_interpolate_log_barrier(grid, 0.001, 1.0e-1)
    #V5,g5 = optim_growth.VFI_interpolate_log_barrier(nodes, 0.001, 1.0e-1)
    #V4,g4 = optim_growth.VFI_interpolate_log_barrier_bound(grid, grid, 0.001, 1.0)
    plt.plot(nodes, g1)
    plt.plot(nodes, g2)
    plt.plot(nodes, g3) 
    #plt.plot(grid, g4) # <-- Da distinto a g1 y g2.
    #plt.plot(nodes, g5) # <-- Da distinto a g1 y g2.
    plt.show()



# Como el de antes pero todo hecho en el grid fino...
def test8():
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
    nodes = np.linspace(0.0001, 1.0, 10)
    V1,g1 = optim_growth.VFI_interpolate(grid,  0.001)
    V2,g2 = optim_growth.VFI_grid_search(grid,  0.001)
    V3,g3 = optim_growth.VFI_interpolate_log_barrier(grid, 0.001, 2.0, 1e8, 1.0) # No hace mucha diferencia, termina devolviendo lo mismo....
    #V4,g4 = optim_growth.VFI_interpolate_log_barrier(grid, 0.001, 1.0e-1)
    #V5,g5 = optim_growth.VFI_interpolate_log_barrier(nodes, 0.001, 1.0e-1)
    #V4,g4 = optim_growth.VFI_interpolate_log_barrier_bound(grid, grid, 0.001, 1.0)
    plt.plot(grid, g1)
    plt.plot(grid, g2)
    plt.plot(grid, g3) 
    #plt.plot(grid, g4) # <-- Da distinto a g1 y g2.
    #plt.plot(nodes, g5) # <-- Da distinto a g1 y g2.
    plt.show()


# Raro!! Incluso quedando la phi con cero multiplicado, devuelve la misma solucion!!!

#test1()
#test2()
#test3()
#test4() # Perfecta coincidencia.

#test5()
#test6()

#test7()
test8()


# Es lento el procedimiento al comienzo, pero despues toma velocidad y ahora si parece andar bien.
# No se estanca en y0 igual a y_opt y tampoco parece devolver valores fuera del rango...
# El esquema adaptativo del gradiente, aunque a veces lento (en especial al comienzo), parece andar bien.

# Si el epsilon para la aproximacion del jacobiano es muy chico, el y_opt se vuelve el y_0 inicial.
# Si es "grande", el y_opt se vuelve cualquier valor, incluso negativo. Raro pues deberia ser expulsado por la barrera.

# La barrera logaritmica es un problema, da distinto que el resto...

# interp_nodes y grid deben ser lo mismo, por el tema del V... (por ahora)

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
