import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import projgrad
from scipy.stats.mstats import gmean

import matplotlib
colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
black = matplotlib.rcParams['axes.labelcolor']

tcellcolor = '#0E1E97'
tcellcoloralt = '#0e7b97'

import plotting
from lib import *

from optparse import OptionParser
parser = OptionParser()
parser.add_option("-t", "--talk", action="store_true", dest="talk", default=False,
                  help="optimize figure for talks")
(options, args) = parser.parse_args()
talk = options.talk
if talk:
    plt.style.use('../talk.mplstyle')
else:
    plt.style.use('../paper.mplstyle')

def fcompfull_dC(x, t, alpha, K, dC, delta=0.0):
    "dC: function called as dC(C, t) giving rhs for dC/dt"
    T, C = x
    B = 0.5*(T+C+K - ((T+C+K)**2 - 4*T*C)**.5)
    return [alpha*B-delta*T, dC(C, t)]

def make_dnu(x, xt, factor):
    x /= np.sum(x)
    dC = lambda C, t: 0 if ((t > xt[-1]) or (t <= xt[0])) else x[xt.searchsorted(t)-1]*factor
    return dC

color_index = [0, 2, 1]

def plot_kinetics(Cfactor, T0, alpha, mu, K, delta, tau, axes=None, lspmhc=':', arrows=True):
    if axes is None:
        figsize = (5.5, 7.0) if talk else (2.75, 3.5)
        fig, axes = plt.subplots(figsize=figsize, nrows=3, sharex=True)
    ts = np.linspace(0, 7, 200)
    T = 4.0
    fold = 5.0
    dnu = make_dnu(fold**np.arange(1, 5), np.arange(5), Cfactor)
    C0dCs = [('Pulse', 0.0, lambda C, t: Cfactor*np.heaviside(-t+1, 0.5)),
             ('Constant', 0.0, lambda C, t: Cfactor*np.heaviside(-t+T, 0.5)/T),
             ('Exponential', 0.0, lambda C, t: dnu(C, t))]
    T6 = []
    for i, (name, C0, dC) in enumerate(C0dCs):
        color = colors[color_index[i]]
        axes[0].plot(ts, [dC(0.0, t)/1e6 for t in ts], c=color, ls=lspmhc, label=name)
        xs = odeint(fcompfull_dC, [T0, C0], ts, args=(alpha, K, lambda C, t: -mu*C + dC(C, t), delta), max_step=0.001)
        T6.append(xs[ts.searchsorted(6), 0])
        axes[2].plot(ts, xs[:, 0], c=color)
        axes[1].plot(ts, xs[:, 1], ls=lspmhc, c=color)#, label='pMHCs')
    axes[2].set_xlabel('Time in days')
    axes[0].set_ylabel('Antigen input\nin $10^6$/day')
    axes[1].set_ylabel('pMHC\nnumber')
    axes[2].set_ylabel('T cell\nnumber')
    axes[2].set_yscale('log')
    axes[1].set_yscale('log')
    axes[1].set_ylim(70.0)
    axes[0].set_ylim(0.0, 3.0)
    if not talk:
        axes[0].legend()
    if arrows:
        axes[2].annotate('', xy=(6.0, T6[1]), xytext=(6.0, T6[0]),
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color=colors[2]))
        axes[2].annotate('', xy=(6.0, T6[2]), xytext=(6.0, T6[0]),
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color=colors[1]))
        axes[2].annotate('%g'%round(T6[1]/T6[0]), xy=(5.75, np.exp(0.5*(np.log(T6[1])+np.log(T6[0])))), va='center', ha='right',
                         color=colors[2])
        axes[2].annotate('%g'%round(T6[2]/T6[0]), xy=(6.25, np.exp(0.5*(np.log(T6[1])+np.log(T6[0])))), va='center', ha='left', color=colors[1])
    for ax in axes:
        ax.set_xlim(0, max(ts))

    return axes


def Tday6(x, xt, Cfactor=2e6, T0=1e2, alpha=2.47, mu=3.1,
                 K=1e1, delta=0.23, T=4.0, tau=1.0):
    ts = list(xt)
    ts.extend([6.0])
    ts = np.array(ts)
    dC = make_dnu(x, xt, Cfactor)
    C0 = 0.0
    xs = odeint(fcompfull_dC, [T0, C0], ts, args=(alpha, K, lambda C, t: -mu*C + dC(C, t), delta),
                max_step=0.01, nsteps=1e6)
    return xs[-1, 0]/T0

def objective(x, xt, **kwargs):
    return -Tday6(x, xt, **kwargs), scipy.optimize.approx_fprime(x, lambda x: -Tday6(x, xt, **kwargs), epsilon=(x*1e-4+1e-8))
    
N = 4
xt = np.linspace(0, 4, N+1)
x0 = np.ones(N)/N
def callback(f, x):
    print(x)
res = projgrad.minimize(objective, x0, args=(xt,), disp=True, nboundupdate=1, callback=callback, algo='slow', maxiters=20)

N = 4
xt = np.linspace(0, 4, N+1)
exp = scipy.optimize.minimize_scalar(lambda a: -Tday6(a**np.arange(N)/np.sum(a**np.arange(N)), xt),
                                     method='brent', bracket=(0.1, 20.0))
print(exp)

fig, axes = plt.subplots(figsize=(1.75, 3.1), nrows=3, sharex=True)
plot_kinetics(Cfactor=2e6, T0=1e2, alpha=2.47, mu=3.1, K=1e1, delta=0.23, tau=0.5, axes=axes, lspmhc='-', arrows=False)
axes[2].set_xticks(np.arange(0, 8, 2))
axes[2].set_yticks(10**np.arange(2, 7, 2))
axes[1].set_yticks(10**np.arange(2, 7, 2))
plotting.label_axes(axes, xy=(-0.65, 0.95), labelstyle='%s', fontweight='bold')
fig.tight_layout(pad=0.1)

fig.savefig('fig4ABC%s.png'%('talk' if talk else ''), dpi=300)
fig.savefig('fig4ABC%s.svg'%('talk' if talk else ''))


fig, axes = plt.subplots(figsize=(1.75, 2.9), nrows=2)
Cfactor = 2e6
T6s = []
ax = axes[0]
protocols = [(res.x, 'Optimal'), (5.0**np.arange(N), 'Experiment')]
colorsh = ['k', colors[1]]
for i, (x, label) in enumerate(protocols):
    x /= np.sum(x)
    T6 = Tday6(x, xt)
    print(T6)
    T6s.append(T6)
    dC = lambda C, t: -1 if t > xt[-1] else x[xt.searchsorted(t)-1]*(N/4.0)*Cfactor
    ts = np.linspace(1e-3, 6, 1000)
    ax.plot(ts, [dC(0, t) for t in ts], color=colorsh[i],
            label=label)

for T6 in T6s[1:]:
    print('fold expansion %g%%'%round((T6-T6s[0])/T6s[0]*100))
ax.legend(ncol=1, loc='upper left')
ax.set_xlim(0.0)
ax.set_xticks(np.arange(0, 9, 2))
ax.set_ylim(2e3, 7e6)
ax.set_yscale('log')
ax.set_xlabel('Time in days')
ax.set_ylabel('Antigen input\n in 1/day')

ax = axes[1]
folds = np.logspace(-1.1, 1.1)
ax.plot(folds, [Tday6(fold**np.arange(N)/np.sum(fold**np.arange(N)), xt)/(-res.fun) for fold in folds],
        c=colors[5])
ax.plot([5.0], [Tday6(fold**np.arange(N)/np.sum(fold**np.arange(N)), xt)/(-res.fun) for fold in [5.0]],
        'o', c=colors[1])
ax.set_xscale('log')
ax.set_xticks([0.2, 1, 5.0])
ax.set_xlim(min(folds), max(folds))
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel('Fold change per day')
ax.set_ylabel('Fold expansion\n rel. to optimal')

if not talk:
    plotting.label_axes(axes, labels='DE', xy=(-0.55, 1.0), labelstyle='%s', fontweight='bold')

fig.tight_layout(pad=0.1)

fig.savefig('fig4DE%s.png'%('talk' if talk else ''), dpi=300)
fig.savefig('fig4DE%s.svg'%('talk' if talk else ''))
