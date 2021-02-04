#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 17:54:03 2020

@author: alexansdremonod
"""
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy
from scipy.stats import shapiro, bartlett, ttest_ind

def droite_de_Henry(data, variables):
    for i in variables:
        sm.qqplot(data[i], dist=scipy.stats.distributions.norm, line='s')
        plt.title("Droite de Henry : {}".format(i), fontsize='x-large')

def test_shapiro(data, variable, seuil):
    for i in variable:
        stat, p = shapiro(data[i])
        print('stat=%.3f, p=%.3F\n'% (stat, p))
        if p > seuil:
            print("{} : suit la loi normale au seuil de {seuil}%".format(i, seuil=seuil*100))
        else:
            print("{} : ne suit pas la loi normale au seuil de {seuil}%".format(i, seuil=seuil*100))

def test_bartlett(variable, *args, seuil):
    stat, p = bartlett(variable)
    if p > seuil:
        print("Les variances sont homogènes")
    else:
        print("Les variances ne sont pas homogènes")
