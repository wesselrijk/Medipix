#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import peakutils
from peakutils.plot import plot as pplot

from scipy.optimize import curve_fit

def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

filenames = glob.glob('./*.txt')

h = 1
f1 = 1
f2 = 1
f3 = 1
f4 = 1

for idx in range(len(filenames)):
    df = pd.read_csv(filenames[idx], header=None, sep='\t', names=['THL','Energy'])
    # First line
    #df.ix[0]
    # Last line
    #df.ix[(len(df)-1)]

    x = df['THL']
    f = df['Energy']
    f_prime = np.zeros(len(df))

    for i in range(len(x)):
        if (i == (len(x)-(2*h))):
            break
        if (i > 2*h):
            f1 =   -f[i + 2*h]
            f2 =  8*f[i +   h]
            f3 = -8*f[i -   h]
            f4 =    f[i - 2*h]
            f_prime[i] = ((f1+f2+f3+f4) / (12*h) )

    y = -f_prime

    max_x = x[y.argmax()]

    indexes = peakutils.indexes(y, thres=0.5, min_dist=1)

    mean = sum(x*y)/sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
    popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

    fit_y_max = max(Gauss(x,*popt))
    fit_x_max = ( x[Gauss(x,*popt).argmax()] )

    plt.plot(x, y, '-')
    plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
    plt.plot(x[indexes], y[indexes], 'ro')

    print((x[indexes]+4).values)
    #plt.figtext(0.8, 0.85, 'Peaks: ' + str((x[indexes]+4).values),
    plt.figtext(0.8, 0.85, 'Gauss max: %s, %s' % (fit_x_max, fit_y_max),
        wrap=True,
        bbox={'facecolor':'white',
                'alpha':0.9,
                'pad':10})

    max_txt = 'Max: ' + (str(max_x+4)[:5])
    plt.yscale('linear')
    plt.ylim([ 1, (1.2*max(y)) ])
    plt.title(filenames[idx])
    plt.show()
