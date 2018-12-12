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
from Bellman import *


def test1():
    alfa = 1/3
    gamma = 0
    U = build_utility(gamma)
    F = build_prod_function(alfa)
    
    
