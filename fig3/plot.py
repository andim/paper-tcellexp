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

fig, axarr = plt.subplots(figsize=(3.55, 2.8), ncols=2, nrows=2, sharex=True, sharey=True)
axes = axarr.flatten()
ts = np.linspace(0, 6.5)

alpha = 1.2
mu = 1.1
delta = 0.0
K1 = 1e0
K2 = 1e1
C = 1e4

linestyles = ['-', '-', ':']


ax = axes[0]
T1 = 1e1
T2 = 1e1
xs = odeint(ftwospecifities, [T2, T1, C], ts, args=(alpha, mu, K1, K2, delta))
ls = [ax.plot(ts, xs[:, i], linestyles[i], c=colors[i])[0] for i in range(3)]
xs = odeint(fcompfull, [T2, C], ts, args=(alpha, mu, K1, delta))
l1, = ax.plot(ts, xs[:, 0], color=ls[0].get_color(), ls='--')
xs = odeint(fcompfull, [T1, C], ts, args=(alpha, mu, K2, delta))
l2, = ax.plot(ts, xs[:, 0], color=ls[1].get_color(), ls='--')
#ls.extend([l1, l2])
l = ax.legend([ls[0], l1, ls[1], l2], ['$T_1$', 'only $T_1$', '$T_2$', 'only $T_2$'], ncol=2, loc='lower left')
ax.legend([ls[2]], ['pMHCs'], loc='upper center')
ax.add_artist(l)
#ax.legend([l1, l2], ['$T_1$ alone', '$T_2$ alone'], loc='upper center')

ax = axes[1]
T1 = 1e2
T2 = 1e1
xs = odeint(ftwospecifities, [T1, T2, C], ts, args=(alpha, mu, K1, K2, delta))
ls = [ax.plot(ts, xs[:, i], linestyles[i], c=colors[i])[0] for i in range(3)]
xs = odeint(fcompfull, [T1, C], ts, args=(alpha, mu, K1, delta))
l1, = ax.plot(ts, xs[:, 0], color=ls[0].get_color(), ls='--')
xs = odeint(fcompfull, [T2, C], ts, args=(alpha, mu, K2, delta))
l2, = ax.plot(ts, xs[:, 0], color=ls[1].get_color(), ls='--')
ax.set_yscale('log')

if True:

    #alpha = 1.2
    mu = -3.0
    delta = 0.0
    T1 = 1e1
    T2 = 1e1
    C = 1e1

    ax = axes[2]
    xs = odeint(ftwospecifities, [T2, T1, C], ts, args=(alpha, mu, K1, K2, delta))
    ls = [ax.plot(ts, xs[:, i], linestyles[i], c=colors[i])[0] for i in range(3)]
    xs = odeint(fcompfull, [T2, C], ts, args=(alpha, mu, K1, delta))
    ax.plot(ts, xs[:, 0], color=ls[0].get_color(), ls='--')
    xs = odeint(fcompfull, [T1, C], ts, args=(alpha, mu, K2, delta))
    ax.plot(ts, xs[:, 0], color=ls[1].get_color(), ls='--')

    #alpha = 2.4
    mu = -0.5

    ax = axes[3]
    xs = odeint(ftwospecifities, [T1, T2, C], ts, args=(alpha, mu, K1, K2, delta))
    ls = [ax.plot(ts, xs[:, i], linestyles[i], c=colors[i])[0] for i in range(3)]
    xs = odeint(fcompfull, [T1, C], ts, args=(alpha, mu, K1, delta))
    ax.plot(ts, xs[:, 0], color=ls[0].get_color(), ls='--')
    xs = odeint(fcompfull, [T2, C], ts, args=(alpha, mu, K2, delta))
    ax.plot(ts, xs[:, 0], color=ls[1].get_color(), ls='--')

for ax in axes:
    #ax.grid()
    ax.set_ylim(1e0, 1e4)
    ax.set_xlim(min(ts), max(ts))
    ax.set_yscale('log')
    ax.set_yticks(np.logspace(0, 3, 4))
    ax.set_xticks(np.arange(0, 7, 2))
for ax in axarr[:, 0]:
    #ax.set_ylabel('T cell | pMHC number')
    ax.set_ylabel('Number')
for ax in axarr[1, :]:
    ax.set_xlabel('Time in days')
plotting.label_axes(axes, labelstyle='%s', xy=(-0.13, 1.0), fontweight='bold', fontsize=10, va='top')
fig.tight_layout(pad=0.25)
fig.savefig('fig3.svg')
fig.savefig('fig3.png', dpi=300)
