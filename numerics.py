# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 14:05:50 2026

@author: ranieri dugo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

import os
os.chdir('')
import functions as f

## Figure 1
df = pd.DataFrame(columns=['H', 'hit1', 'hit2'])
df['H'] = np.linspace(0.01, 0.99, num = 11)
for i, H in enumerate(df['H']):
    df.loc[i, 'hit1'] = f.hitratio_fgn(1, [0, 1], H)
    df.loc[i, 'hit2'] = f.hitratio_fgn(1, [0, 0.5, 1, 2, 3], H)
plt.figure(figsize=(7, 5))
plt.plot(df["H"], df["hit1"], marker="o", label="n = 1")
plt.plot(df["H"], df["hit2"], marker="o", label="n = 4")
plt.xlabel("H")
plt.ylabel("Hit Ratio")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

## Figure 2
df = pd.DataFrame(columns=['delta1', 'hit1', 'hit2'])
df['delta1'] = np.linspace(0.0001, 10, 500)
for i, d in enumerate(df['delta1']):
    df.loc[i, 'hit1'] = f.hitratio_fgn(h = 1, vd = [0, d], H = 0.15)
    df.loc[i, 'hit2'] = f.hitratio_fgn(h = 1, vd = [0, d], H = 0.65)

plt.figure(figsize=(7, 5))
plt.plot(df["delta1"], df["hit1"], label = "H = 0.15")
plt.plot(df["delta1"], df["hit2"], label = "H = 0.65")
plt.xlabel(r"$\delta_1/h$")
plt.ylabel("Hit Ratio")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

## Table 1
df = pd.DataFrame(columns=['n', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'suc', 'r'])
df['n'] = range(2, 7)

warnings.filterwarnings("ignore", category=UserWarning, module="scipy.optimize._differentiable_functions")
for H in [0.65, 0.15]:
    for n in df['n']:
        res = f.optimize_duration(1, n, H)
        df.loc[df['n'] == n, ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'suc', 'r']] = [
            *list(res['durations']) + [0] * (6 - n),
            res['success'],
            res['hitratio']
        ]
    pd.set_option('display.max_columns', None)
    print(df)


## Figure 3
vd_opt = np.r_[0, f.optimize_duration(h = 1, n = 6, H = 0.65)['durations']]
df = pd.DataFrame(columns=['scaler', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6'])
df['scaler'] = np.linspace(0.5, 2, 15)

for i in range(1, 7):
    for j, scaler in enumerate(df['scaler']):
        vd_adj = vd_opt.copy()
        vd_adj[i] = vd_opt[i] * scaler
        df.loc[j, f'd{i}'] = f.hitratio_fgn(h = 1, vd = vd_adj, H = 0.65)

plt.figure(figsize=(8, 5))
for i, col in enumerate(df.columns[1:]):
    plt.plot(df["scaler"], df[col], marker="o", label = rf'$i={i}$')
plt.xlabel(r"$\delta_i/\delta_i^\star$")
plt.ylabel("Hit Ratio")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

## Figure 4
vlam = [0, 0.25, 0.5, 0.75, 1]
vth = np.linspace(0, 0.8, num = 10)
plt.figure(figsize=(8, 5))
for lam in vlam:
    y = f.adj_perf_strat(vth, lam, h = 1, vd = [0, 1], H = 0.65, sigma = 1)
    plt.plot(vth, y, label=rf"$\lambda = {lam}$")
plt.xlabel(r"$\theta$")
plt.ylabel(r"$\tilde R_\lambda\left(\theta\right)$")
plt.legend()
plt.ylim(- 0.15, 0.2)
plt.xlim(0, 0.8)
plt.grid(True)
plt.show()

## Figure 5
vH = [0.6, 0.7, 0.8]
vlam = np.linspace(0, 1, num = 10)
dfs = [
    pd.DataFrame({
        "lam": vlam,
        "theta_opt": np.nan,
        "p_zero": np.nan
    })
    for H in vH
]

for i, H in enumerate(vH):
    for j, lam in enumerate(vlam):
        theta_opt = f.optimize_threshold(lam = lam, h = 1, vd = [0, 1], H = H, ub = 1)['theta']
        dfs[i].loc[j, 'theta_opt'] = theta_opt
        dfs[i].loc[j, 'p_zero'] = f.p_zero(theta_opt, h = 1, vd = [0, 1], H = H, sigma = 1)


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, df in enumerate(dfs):
    H = vH[i]
    axes[0].plot(
        df["lam"],
        df["theta_opt"],
        marker="o",
        label=fr"$H={H}$"
    )
    axes[1].plot(
        df["lam"],
        df["p_zero"],
        marker="o",
        label=fr"$H={H}$"
    )
axes[0].set_xlabel(r"$\lambda$")
axes[0].set_ylabel(r"$\theta^*$")
axes[0].legend()
axes[1].set_xlabel(r"$\lambda$")
axes[1].set_ylabel(r"$p_0$")
axes[1].legend()
plt.tight_layout()
plt.show()

## Figure 6
vH = [0.1, 0.2, 0.3, 0.4, 0.51, 0.6, 0.7, 0.8, 0.9, 0.99]
df = pd.DataFrame({
    "H": vH,
    "theta_opt": np.nan,
    "p_zero": np.nan
})

for j, H in enumerate(vH):
    theta_opt = f.optimize_threshold(lam = 0.1, h = 1, vd = [0, 1], H = H)['theta']
    df.loc[j, 'theta_opt'] = theta_opt
    df.loc[j, 'p_zero'] = f.p_zero(theta_opt, h = 1, vd = [0, 1], H = H, sigma = 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(
    df["H"],
    df["theta_opt"],
    marker="o"
)
axes[0].set_xlabel(r"$H$")
axes[0].set_ylabel(r"$\theta^*$")
axes[1].plot(
    df["H"],
    df["p_zero"],
    marker="o"
)
axes[1].set_xlabel(r"$H$")
axes[1].set_ylabel(r"$p_0$")
plt.tight_layout()
plt.show()

## Figure 7
vlam = [0.1, 0.5]
vH = [0.1, 0.2, 0.3, 0.4, 0.51, 0.6, 0.7, 0.8, 0.9]

dfs = [
    pd.DataFrame({
        "H": vH,
        "perf_00": np.nan,
        "perf_l0": np.nan,
        "perf_lt": np.nan
    })
    for lam in vlam
]
for j, H in enumerate(vH):
    tmp = f.adj_perf_strat(
        theta = 0, lam = 0, h = 1, vd = [0, 1], H = H, sigma = 1)
    for i, lam in enumerate(vlam):
        theta_opt = f.optimize_threshold(lam = lam, h = 1, vd = [0, 1], H = H)['theta']
        dfs[i].loc[j, 'perf_00'] = tmp
        dfs[i].loc[j, 'perf_l0'] = f.adj_perf_strat(
            theta = 0, lam = lam, h = 1, vd = [0, 1], H = H, sigma = 1)
        dfs[i].loc[j, 'perf_lt'] = f.adj_perf_strat(
            theta = theta_opt, lam = lam, h = 1, vd = [0, 1], H = H, sigma = 1)

ylow = [- 0.1, - 0.4]
yup = [0.9, 1]
fig, axes = plt.subplots(1, 2, figsize = (12, 5), sharey = False)
for i, ax in enumerate(axes):
    df = dfs[i]
    ax.plot(df["H"], df["perf_00"], marker="o", label=r"$\mathrm{perf}_{00}$")
    ax.plot(df["H"], df["perf_l0"], marker="o", label=r"$\mathrm{perf}_{l0}$")
    ax.plot(df["H"], df["perf_lt"], marker="o", label=r"$\mathrm{perf}_{lt}$")
    ax.set_ylim(ylow[i], yup[i])
    ax.set_xlabel(r"$H$")
    ax.set_title(fr"$\lambda = {vlam[i]:.3f}$")
    ax.grid(True, alpha=0.3)
    ax.legend()
axes[0].set_ylabel("Risk-Adjusted Performance")
plt.tight_layout()
plt.show()