import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import ode
import scipy.optimize

import seaborn as sns

import matplotlib
colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
black = matplotlib.rcParams['axes.labelcolor']

tcellcolor = '#0E1E97'
tcellcoloralt = '#0e7b97'
pmhccolor = colors[3]
colors = [tcellcolor, tcellcoloralt, pmhccolor]

import sys
sys.path.append('..')
import plotting
from lib import *
        
plt.style.use('../paper.mplstyle')

def T7(T0, K):
    ts = [0.0, 3.5, 7.0]
    xs = odeint(fcompfull, [T0, C], ts, args=(alpha, mu, K, delta))
    return xs[-1, 0]

# parameters as in Fig.3D (slowly increasing antigen availability)
alpha = 1.2
mu = -0.5
delta = 0.0
C = 10
T0s = np.logspace(0, 3, 50)
Ks = np.logspace(0, 3, 50)

# alternative parameters as in Fig.1
#alpha = 1.5
#mu = 1.2
#delta = 0.22
#C = 10**6.7
#T0s = np.logspace(0, 6.5, 50)
#Ks = np.logspace(0, 6.5, 50)


foldexpansions = np.zeros((len(T0s), len(Ks)))
for i, T0 in enumerate(T0s):
    for j, K in enumerate(Ks):
        foldexpansions[i, j] = T7(T0, K)/T0

fig, ax = plt.subplots(figsize=(2.75, 2.25))
X, Y = np.meshgrid(Ks, T0s)
plt.pcolor(X, Y, np.log10(foldexpansions), cmap='viridis')
cbar = plt.colorbar()
cbar.set_label('log$_{10}$ fold expansion')
CS = ax.contour(X, Y, foldexpansions, colors='w', levels=[3, 10, 100, 400])
ax.set_xscale('log')
ax.set_yscale('log')
plt.clabel(CS, CS.levels, inline=False, inline_spacing=2, fmt='%g')
ax.set_xlabel('$K$')
ax.set_ylabel('$T(0)$')
fig.tight_layout()
plt.show()
fig.savefig('figS1.svg')
fig.savefig('figS1.png', dpi=300)
