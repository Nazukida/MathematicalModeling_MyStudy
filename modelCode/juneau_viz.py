# -*- coding: utf-8 -*-
"""
High-quality visualizations for the Juneau model.

Creates publication-ready figures: seasonal visitor curve, stacked contributions,
and a U heatmap across (c1, c2) policy space. Saves PNGs to ./figures.
"""
from __future__ import annotations
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import TwoSlopeNorm

# ensure project root is on sys.path so we can import modelCode
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modelCode.juneau_model import JuneauModel, JuneauParams


def setup_style():
    sns.set(style='whitegrid')
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['legend.frameon'] = False


class FigureSaver:
    def __init__(self, save_dir='./figures'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save(self, fig, name):
        path = os.path.join(self.save_dir, name)
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print('Saved:', path)


def plot_seasonal_N(model: JuneauModel, decisions: dict, title_suffix='', saver: FigureSaver = None):
    """绘制一年中每日游客数的季节性曲线（自然 vs 政策调整）"""
    setup_style()
    t = model.t
    n0 = model.N0(t)
    N_policy = model.N(decisions['c1'], decisions['c2'])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, n0, color='#2E86AB', lw=1.5, label='Natural demand $N_0(t)$')
    ax.plot(t, N_policy, color='#F18F01', lw=2.0, label='Policy-adjusted $N(t)$')

    # highlight peak season
    ax.axvspan(121, 270, color='#D6EAF8', alpha=0.25)
    ax.text(195, ax.get_ylim()[1]*0.9, 'Peak season', ha='center', va='center', fontsize=9)

    ax.set_xlabel('Day of Year')
    ax.set_ylabel('Daily visitors')
    ax.set_title(f'Seasonal Visitor Pattern {title_suffix}')
    ax.legend()

    # tasteful grid and style
    ax.grid(alpha=0.25)

    if saver:
        saver.save(fig, 'seasonal_visitors.png')
    return fig


def plot_contributions(model: JuneauModel, decisions: dict, saver: FigureSaver = None):
    """绘制 P, E, S 三部分对 U 的贡献（年度总和）并用堆叠条形图显示"""
    setup_style()
    vals = model.compute_PES((decisions['c1'], decisions['c2'], decisions['I'], decisions['gamma1'], decisions['x1'], decisions['x2']))

    P = vals['P']
    E = vals['E']
    S = vals['S']

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = np.array([P, E, S]) / 1e6  # convert to millions
    labels = ['Economic (P)', 'Environmental (E)', 'Social (S)']
    colors = ['#2E86AB', '#27AE60', '#A23B72']

    ax.barh(0, parts[0], color=colors[0])
    ax.barh(0, parts[1], left=parts[0], color=colors[1])
    ax.barh(0, parts[2], left=parts[0]+parts[1], color=colors[2])

    ax.set_xlim(min(0, parts.sum()*0.0), parts.sum()*1.05)
    ax.set_yticks([])
    ax.set_xlabel('Contribution to U (Million USD)')
    ax.set_title('Annual Contribution Breakdown to Total Social Utility')

    # annotate
    left = 0.0
    for v, lab in zip(parts, labels):
        ax.text(left + v/2, 0, f'{lab}\n{v:,.1f}M', ha='center', va='center', color='white', fontsize=9, weight='bold')
        left += v

    if saver:
        saver.save(fig, 'contributions_breakdown.png')
    return fig


def plot_U_heatmap(model: JuneauModel, c1_range=(5000,30000), c2_range=(0,8000), nx=80, ny=60, saver: FigureSaver = None):
    """在(c1,c2)空间绘制 U 总量热力图（用于政策可视化）"""
    setup_style()
    c1_vals = np.linspace(c1_range[0], c1_range[1], nx)
    c2_vals = np.linspace(c2_range[0], c2_range[1], ny)
    U_grid = np.zeros((ny, nx))

    # evaluate grid
    for i, c2 in enumerate(c2_vals):
        for j, c1 in enumerate(c1_vals):
            vals = model.compute_PES((c1, c2, 1000.0, 0.5, 20.0, 10.0))
            U_grid[i, j] = vals['U']

    # normalize to millions for plotting
    U_m = U_grid / 1e6

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(U_m, ax=ax, cmap='magma', cbar_kws={'label': 'U (Million USD)'},
                xticklabels=np.linspace(c1_range[0], c1_range[1], 5).astype(int),
                yticklabels=np.linspace(c2_range[0], c2_range[1], 5).astype(int))

    ax.set_xlabel('c1 (peak capacity)')
    ax.set_ylabel('c2 (off-peak target)')
    ax.set_title('Total Social Utility U across policy space (sampled)')

    if saver:
        saver.save(fig, 'U_heatmap.png')
    return fig


def demo_generate():
    params = JuneauParams()
    model = JuneauModel(params)
    saver = FigureSaver('./figures')

    # choose two example policies (baseline and a policy)
    baseline = {'c1': 30000.0, 'c2': 2000.0, 'I': 1000.0, 'gamma1': 0.5, 'x1': 0.0, 'x2': 0.0}
    policy = {'c1': 20000.0, 'c2': 4000.0, 'I': 1200.0, 'gamma1': 0.6, 'x1': 25.0, 'x2': 10.0}

    print('Generating figures into ./figures ...')
    plot_seasonal_N(model, baseline, title_suffix='(Baseline)', saver=saver)
    plot_seasonal_N(model, policy, title_suffix='(Policy)', saver=saver)
    plot_contributions(model, policy, saver=saver)
    plot_U_heatmap(model, saver=saver)


if __name__ == '__main__':
    demo_generate()
