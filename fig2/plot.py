import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize

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

df = pd.read_csv('../data/zehn.csv', index_col=0)
df['p'] = df['percentage']/100
df['ratio'] = df['p']/(1.0-df['p'])


dftc = pd.read_csv('../data/zehn-timecourse.csv')
dftc['p'] = dftc['percentage']/100.0
dftc['ratio'] = dftc['p']/(1.0-dftc['p'])
# ec for reference ligand = 8 pM
ecN4 = 8

def func(params):
    alpha, mu, delta, logC0, T0 = params
    C0 = 10**logC0
    residuals = []
    for peptide, dfg in dftc.groupby('peptide'):
        ec50 = df.loc[peptide]['ec50']
        K = ec50
        Ts = odeint(fsaturation, [T0, C0], np.asarray(dfg['day']), args=(alpha, mu, K, delta))[:, 0]
        y = np.asarray(dfg['ratio'])
        residuals.extend(np.log(y) - np.log(Ts))
    return residuals


alpha = 2.5
mu = 3.0
delta = 0.5
logC0 = 3
T0 = 1e-2

params = (alpha, mu, delta, logC0, T0)
optparams, pcov, infodict = scipy.optimize.leastsq(func, params, full_output=True)[:3]

# calculate and print errors
paramnames = 'alpha', 'mu', 'delta', 'logC0', 'T0'
s_sq = np.sum(infodict['fvec']**2) / (infodict['fvec'].size - optparams.size)
pcov = pcov * s_sq
optse = pcov.diagonal()**.5
optparams_units = optparams.copy()
# absolute value for C concentration
#optparams_units[3] += np.log10(ecN4)
# renormalize T cell units to percentage at day 4 = 0.01
T4 = 0.01/0.99
optparams_units[4] /= T4
optse[4] /= T4
print('fitted parameters:')
print('\n'.join([': '.join((p[0], str_quant(*p[1:], scientific=False))) for p in zip(paramnames, optparams_units, optse)]))


fig, axes = plt.subplots(figsize=(7, 3.5) if talk else (3.45, 1.7), ncols=2)
alpha, mu, delta, logC0, T0 = optparams
C0 = 10**logC0

strains = list(df.index[:-1])
dfm = dftc.merge(df[['ec50']], left_on='peptide', right_index=True)
minratio = dfm[dfm['day']==4]['ratio'].min()
dfm = dfm[dfm['day']==7]

def log_10_product(x, pos):
    return '%g' % (x)
formatter = plt.FuncFormatter(log_10_product)


ax = axes[1]
Tfactor = 1/T0
ts = np.linspace(4.0, 9.25, 50)
for strain in strains:
    dfg = dftc[dftc['peptide']==strain]
    l, = ax.plot(dfg['day'], dfg['ratio']/minratio, 'o', c=colors[strains.index(strain)+1], label=strain)
    ec50 = df.loc[strain]['ec50']
    K = ec50
    xs = odeint(fsaturation, [T0, C0], ts, args=(alpha, mu, K, delta))
    ax.plot(ts, xs[:, 0]/T0, c=l.get_c())
#ax.set_xlim(4.0, max(ts))
ax.set_yscale('log')
ax.yaxis.set_major_formatter(formatter)
ax.set_xlabel('Time in days')
ax.set_ylabel('T cell number')


ax = axes[0]
ax.scatter(dfm['ec50'], dfm['ratio']/minratio, c=colors[1:dfm['ratio'].shape[0]+1], label='Experiment')
ts = [4.0, 7.0]
T6s = []
Ks = np.logspace(-.5, 3.5)
for K in Ks:
    xs = odeint(fsaturation, [T0, C0], ts, args=(alpha, mu, K, delta))
    T6s.append(xs[-1, 0])
ax.plot(Ks, np.array(T6s)/T0, c=colors[0], label='Model')
leg = ax.legend(loc='lower left')
leg.legendHandles[1].set_color('.2')
ax.set_xscale('log')
ax.set_yscale('log')
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
ax.set_xlim(5e-1, 2e3)
ax.set_xlabel('rel. pMHC concentration\nfor half max. response')
ax.set_ylabel('Fold expansion')

for ax in axes:
    ax.set_ylim(0.5, 3e2)
if not talk:
    plotting.label_axes(axes, labelstyle='%s', xy=(-0.3, 0.94), fontweight='bold', fontsize=10)

fig.tight_layout(pad=0.25, w_pad=0.0)
fig.savefig('fig2%s.png'%('talk' if talk else ''), dpi=300)
fig.savefig('fig2%s.svg'%('talk' if talk else ''))
