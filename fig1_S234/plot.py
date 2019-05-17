import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.stats.mstats import gmean

import matplotlib
colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
black = matplotlib.rcParams['axes.labelcolor']

tcellcolor = '#0E1E97'
tcellcoloralt = '#0e7b97'

import sys
sys.path.append('..')
import plotting
from lib import *

talk = False
plt.style.use('../paper.mplstyle')

dftimecourse = pd.read_csv('../data/quiel-timecourse.csv')
dftimecoursegmean = dftimecourse.groupby(['days after immunization', 'transferred cells']).agg([gmean, 'count'])
dftimecoursegmean = dftimecoursegmean.unstack()['cells']

dfexpansion = pd.read_csv('../data/quiel.csv', index_col=0)

ms=10 if talk else 5
tcellcolors = np.array([tcellcolor, tcellcoloralt])
def plot_timecourse(ax):
    dftimecourse.plot.scatter(x='days after immunization', y='cells', ax=ax,
                      c=tcellcolors[(dftimecourse['transferred cells']>10000).astype(int)],
                              s=ms, alpha=0.5, edgecolor='none')
    dftimecoursegmean['gmean'].plot(kind='line', marker='x', ls='None', ax=ax, ms=ms, color=tcellcolors)
    ax.legend_.remove()
    ax.set_yscale('log')
    ax.set_xticks(range(0, 11, 2))
    ax.set_xlim(0.0, 10.5)
    ax.set_ylim(4e1, 4e6)
    ax.set_xlabel('Time in days')
    ax.set_ylabel('Number')

def plot_foldexpansion(ax):
    ax.errorbar(dfexpansion.index, dfexpansion['factor of expansion'],
                yerr=dfexpansion['sem'],
                ms=ms,
                fmt='x', c=tcellcolor, label='Experiment')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Fold expansion')
    ax.set_xlabel('Initial number of T cells')

def plot_prediction(ax, tstar, Tstar):
    linekwargs = dict(c='.7', ls='--', lw=0.75)
    ax.axvline(tstar, **linekwargs)
    ax.axhline(Tstar, **linekwargs)
    ax.set_xticks(range(0, 11, 2))
    ax.set_xlabel('Time in days')
    ax.set_ylabel('number')
    ax.set_yscale('log')

def func(params, K=0.0, sigma=1.0):
    logC0, alpha, mu, delta = params
    C0 = 10**logC0
    residuals = []

    for i, T0 in enumerate(dfexpansion.index):
        xs = odeint(fcomp, [T0, C0], np.linspace(0, 7, 3), args=(alpha, mu, K, delta))
        y = dfexpansion['factor of expansion'].iloc[i]*T0
        residuals.append(dfexpansion['N'].iloc[i]**.5*(np.log(xs[-1, 0]) - np.log(y))/sigma**2)

    timecourse = np.asarray(dftimecoursegmean['gmean'])
    ts = [0.0]
    ts.extend(list(dftimecoursegmean['gmean'].index))
    for i, T0 in enumerate([3e2, 3e4]):
        xs = odeint(fcomp, [T0, C0], ts, args=(alpha, mu, K, delta))
        residuals.extend(dftimecoursegmean['count'].iloc[:, i]**.5*(np.log(xs[1:, 0]) - np.log(timecourse[:, i]))/sigma**2)

    return np.array(residuals)

logC0 = np.log10(3e6)
alpha = 1.4
mu = 1.15
delta = 0.15
params = logC0, alpha, mu, delta
optparams, pcov, infodict = scipy.optimize.leastsq(func, params, args=(0.0, 1.0), full_output=True)[:3]

s_sq = np.sum(infodict['fvec']**2) / (infodict['fvec'].size - optparams.size)
pcov = pcov * s_sq
optse = pcov.diagonal()**.5
print(optparams, optse)
paramnames = ['logC0', 'logK', 'alpha', 'mu', 'delta']
[': '.join((p[0], str_quant(*p[1:], scientific=False))) for p in zip(paramnames, optparams, optse)]

def chi2(K):
    residuals = func(optparams, K=K, sigma=1.12)
    return np.sum(residuals**2), len(residuals)-len(optparams)
chi2opt, df = chi2(0)
# When is Delta chi2 = 1 for K?
Kmax = scipy.optimize.bisect(lambda K: chi2(K)[0]-chi2opt-1, 0, 1e4)
print(chi2opt/df, Kmax)

logC0, alpha, mu, delta = optparams
C0 = 10**logC0
K = 0.0

ts = np.linspace(0, 10.5, 100)

fig, axes = plt.subplots(figsize=(8.0, 3.5) if talk else (4.0, 1.75), ncols=2)

ax = axes[1]
plot_timecourse(ax)
ls = []
for i, T0 in enumerate([3e2, 3e4]):
    xs = odeint(fcomp, [T0, C0], ts, args=(alpha, mu, K, delta))
    l, = ax.plot(ts, xs[:, 0], c=ax.lines[i].get_c(), label='T cell')
    ls.append(l)
lpMHC, = ax.plot(ts, xs[:, 1], c=colors[3], ls=':', label='pMHCs')
handles = [tuple(ls), lpMHC]
labels = ['T cells', 'pMHCs']
ax.legend(handles, labels, handler_map={tuple: plotting.OffsetHandlerTuple()}, loc='lower left', bbox_to_anchor=(0.1, 0.0))

ax = axes[0]

T0s = np.logspace(0, 5, 100)
fes = []
for i, T0 in enumerate(T0s):
    xs = odeint(fcomp, [T0, C0], np.linspace(0, 7), args=(alpha, mu, K, delta))
    fes.append(xs[-1, 0]/T0)
plot_foldexpansion(ax)
ax.plot(T0s, fes, '-', c=tcellcolor, label='Model')
ax.legend(loc='lower left')
ax.set_xticks(10**np.arange(0, 6, 2))

fig.tight_layout()
fig.savefig('fig1.svg')
fig.savefig('fig1.png', dpi=300)


logC0, alpha, mu, delta = optparams
C0 = 10**logC0
K = 1e4

ts = np.linspace(0, 10.5, 100)


#### S3 ####
fig, ax = plt.subplots(figsize=(2.75, 2.25))
T0s = np.logspace(0, 5, 100)
fes = []
for i, T0 in enumerate(T0s):
    xs = odeint(fcomp, [T0, C0], np.linspace(0, 7), args=(alpha, mu, K, delta))
    fes.append(xs[-1, 0]/T0)
plot_foldexpansion(ax)
ax.plot(T0s, fes, '-', c=tcellcolor, label='Model')
ax.legend(loc='lower left')

fig.tight_layout()
fig.savefig('figS3.svg')
fig.savefig('figS3.png', dpi=300)


#### S4 ####

logC0, alpha, mu, delta = optparams
C0 = 10**logC0
K = 1e2

ts = np.linspace(0, 10.5, 100)

fig, ax = plt.subplots(figsize=(2.75, 2.25))

T0s = [10, 100]
delays = np.linspace(0, 4.2, 10)
for T0 in T0s:
    fes = []
    for i, delay in enumerate(delays):
        Cdelay = C0 * np.exp(-mu*delay)
        xs = odeint(fcomp, [T0, Cdelay], np.linspace(0, 7), args=(alpha, mu, K, delta))
        fes.append(xs[-1, 0]/T0)
    ax.plot(delays, fes, '-', label=T0)
ax.legend(loc='lower left')
ax.set_yscale('log')
ax.set_xlim(min(delays), max(delays))
ax.set_xlabel('Time delay in days')
ax.set_ylabel('Fold expansion')
ax.set_ylim(3e1, 1.5e3)
ax.legend(title=r'$T(t_{\rm delay})$')
fig.tight_layout()
fig.savefig('figS4.svg')
fig.savefig('figS4.png', dpi=300)

#### S2 ####
fig, ax = plt.subplots(figsize=(2.75, 2.25))

T0 = 1e0
K = 0.0
xs = odeint(fcomp, [T0, C0], ts, args=(alpha, mu, K, delta))

exponent = ((alpha-delta)/(alpha+mu-delta))
tstar = np.log(C0/T0)/(alpha+mu-delta)
Tstar = T0 * (C0/T0)**exponent

ax.plot(ts, xs[:, 0], c=tcellcolor, label='T cells')
ax.plot(ts, xs[:, 1], c=colors[3], ls=':', label='pMHCs')
ax.plot([tstar], [Tstar], 'o', color=tcellcolor, ms=4)
ax.set_ylim(1, 5e6)
ax.set_xlim(min(ts), max(ts))
ax.set_xticks(range(0, 11, 2))
ax.set_xlabel('Time in days')
ax.set_ylabel('Number')
ax.set_yscale('log')
ax.legend()
fig.tight_layout()
fig.savefig('figS2.svg')
fig.savefig('figS2.png', dpi=300)
