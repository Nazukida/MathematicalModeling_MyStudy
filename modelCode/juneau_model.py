# -*- coding: utf-8 -*-
"""
Juneau tourism model implementation

基于用户提供的论文模型重构，风格参照 `test.py`。

提供：参数类 `JuneauParams`、模型类 `JuneauModel`、求解示例。
"""
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict
import warnings
from scipy.optimize import minimize

warnings.filterwarnings('ignore')


@dataclass
class JuneauParams:
    """模型参数，使用论文给定的默认值"""
    # Tourist flow parameters
    A: float = 16822.0
    B: float = 5514.0

    # Profit per tourist (USD)
    p: float = 250.0

    # Carbon / environmental
    carbon_cost: float = 12.56  # USD per tourist (66.13kg * $190/ton -> ~12.56)

    # Ecosystem resilience
    ERI_max: float = 2e5
    beta: float = 1e-4

    # Investment logistic params
    alpha1: float = 1e-4
    alpha2: float = 1e-4
    Gamma1m: float = 1e8
    Gamma2m: float = 1e8
    Gamma10: float = 1e4
    Gamma20: float = 1e4
    I10: float = 0.0
    I20: float = 0.0

    # Social baseline S1 yearly (paper gives ~7,774,865)
    S1_yearly: float = 7_774_865.0
    beta2: float = 0.0  # if >0, negative social impact per tourist (can be tuned)

    # Constraints thresholds
    E0: float = 0.0
    S0: float = 0.0

    # Days
    days: int = 365


class JuneauModel:
    """实现模型方程并提供目标函数供优化器调用"""

    def __init__(self, params: JuneauParams = None):
        self.p = params if params is not None else JuneauParams()
        # precompute day indices 1..365
        self.t = np.arange(1, self.p.days + 1)

    def N0(self, t: np.ndarray) -> np.ndarray:
        """自然需求 N0(t) = max{-A cos(2π/365 t) + B, 0}"""
        A, B = self.p.A, self.p.B
        vals = -A * np.cos(2 * math.pi / 365.0 * t) + B
        return np.maximum(vals, 0.0)

    def f_policy(self, t: np.ndarray, x1: float, x2: float) -> np.ndarray:
        """政策税/补贴函数 f(t) 在 x1 和 x2 之间的正弦/余弦变动"""
        # f(t) = (x1-x2)/2 * cos(2π/365 t + π) + (x1+x2)/2
        return (x1 - x2) / 2.0 * np.cos(2 * math.pi / 365.0 * t + math.pi) + (x1 + x2) / 2.0

    def N(self, c1: float, c2: float) -> np.ndarray:
        """政策调整后的每日游客数 N(t)"""
        n0 = self.N0(self.t)
        # peak season indices
        peak_mask = (self.t >= 121) & (self.t <= 270)
        N = np.empty_like(n0)
        # peak: min(N0, c1)
        N[peak_mask] = np.minimum(n0[peak_mask], c1)
        # off-peak: max(N0, c2)
        N[~peak_mask] = np.maximum(n0[~peak_mask], c2)
        return N

    def Gamma(self, I_alloc: float, alpha: float, Gm: float, G0: float, I0: float) -> float:
        """Logistic return function Γ(I) (daily contribution assumed constant across days)
        Uses the provided formula in the reconstruction.
        """
        # avoid overflow
        denom = 1.0 + (Gm / G0 - 1.0) * math.exp(-alpha * (I_alloc - I0)) if G0 != 0 else 1.0
        return Gm / denom

    def ERI(self, N_t: np.ndarray) -> np.ndarray:
        ERI_max = self.p.ERI_max
        beta = self.p.beta
        return ERI_max / (1.0 + beta * N_t)

    def E_cost(self, N_t: np.ndarray) -> np.ndarray:
        return N_t * self.p.carbon_cost

    def compute_PES(self, decision: Tuple[float, float, float, float, float, float]) -> Dict[str, float]:
        """给定决策变量，计算 P, E, S 及 U

        decision = (c1, c2, I, gamma1, x1, x2)
        返回字典包含日累加和及总和
        """
        c1, c2, I, gamma1, x1, x2 = decision
        gamma1 = float(np.clip(gamma1, 0.0, 1.0))
        gamma2 = 1.0 - gamma1

        N_t = self.N(c1, c2)
        f_t = self.f_policy(self.t, x1, x2)

        # Economic profit P: sum_t [ N(t)*p + (f(t) - I) ]
        P_daily = N_t * self.p.p + (f_t - I)
        P = float(np.sum(P_daily))

        # Environmental E: sum_t ( -E_cost + ERI + Gamma1(I_alloc) )
        E_cost_t = self.E_cost(N_t)
        ERI_t = self.ERI(N_t)
        # allocate investment to environment: gamma1 * I
        Gamma1_daily = self.Gamma(gamma1 * I, self.p.alpha1, self.p.Gamma1m, self.p.Gamma10, self.p.I10)
        E = float(np.sum(-E_cost_t + ERI_t + Gamma1_daily))

        # Social S: sum_t ( S1_yearly/365 + S2(t) + Gamma2(I_alloc) )
        S1_daily = self.p.S1_yearly / self.p.days
        # negative social impact per day: beta2 * N(t)
        S2_t = - self.p.beta2 * N_t
        Gamma2_daily = self.Gamma(gamma2 * I, self.p.alpha2, self.p.Gamma2m, self.p.Gamma20, self.p.I20)
        S = float(np.sum(S1_daily + S2_t + Gamma2_daily))

        U = P + E + S

        return {
            'P': P,
            'E': E,
            'S': S,
            'U': U,
            'N_mean': float(np.mean(N_t)),
            'N_total': float(np.sum(N_t))
        }

    def objective(self, x: np.ndarray) -> float:
        """优化器调用的目标函数（负的 U，用于最小化）

        x = [c1, c2, I, gamma1, x1, x2]
        对约束使用惩罚项
        """
        dec = (float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]))
        vals = self.compute_PES(dec)
        U = vals['U']

        penalty = 0.0
        # constraints: c1 >= c2, x1 >= x2, E>=E0, S>=S0, 0<=gamma1<=1, 0<=I<=B*p
        if dec[0] < dec[1]:
            penalty += 1e9 * (dec[1] - dec[0])
        if dec[4] < dec[5]:
            penalty += 1e9 * (dec[5] - dec[4])
        # E and S thresholds
        if vals['E'] < self.p.E0:
            penalty += 1e8 * (self.p.E0 - vals['E'])
        if vals['S'] < self.p.S0:
            penalty += 1e8 * (self.p.S0 - vals['S'])
        # I bound: using B (params.B) * p as upper bound
        I_upper = self.p.B * self.p.p
        if dec[2] < 0:
            penalty += 1e8 * (-dec[2])
        if dec[2] > I_upper:
            penalty += 1e8 * (dec[2] - I_upper)

        return -U + penalty


def optimize_model(model: JuneauModel, x0=None):
    """使用约束最小化 (SLSQP) 求解，目标为最大化 U（我们最小化 -U）"""
    # variables: c1, c2, I, gamma1, x1, x2
    # bounds: reasonable ranges
    bounds = [
        (0.0, 50000.0),  # c1
        (0.0, 50000.0),  # c2
        (0.0, model.p.B * model.p.p),  # I (upper bound B * p)
        (0.0, 1.0),      # gamma1
        (-1e4, 1e4),     # x1
        (-1e4, 1e4)      # x2
    ]

    if x0 is None:
        x0 = np.array([20000.0, 2000.0, 1000.0, 0.5, 10.0, 5.0])

    # constraints: keep linear feasibility constraints only.
    # Nonlinear E/S constraints removed because SLSQP may require a feasible
    # starting point; we instead enforce them via penalty in `objective`.
    cons = (
        {'type': 'ineq', 'fun': lambda x: x[0] - x[1]},
        {'type': 'ineq', 'fun': lambda x: x[4] - x[5]},
    )

    res = minimize(fun=model.objective, x0=x0, bounds=bounds, constraints=cons, method='SLSQP', options={'maxiter': 200, 'disp': False})
    return res


if __name__ == '__main__':
    params = JuneauParams()
    model = JuneauModel(params)

    print('Running a quick optimization (this may take a few seconds)...')
    res = optimize_model(model)
    if res.success:
        c1, c2, I, gamma1, x1, x2 = res.x
        print('Optimization success')
        print(f'c1={c1:.1f}, c2={c2:.1f}, I={I:.1f}, gamma1={gamma1:.3f}, x1={x1:.2f}, x2={x2:.2f}')
        vals = model.compute_PES(tuple(res.x))
        print(f"P={vals['P']:.2f}, E={vals['E']:.2f}, S={vals['S']:.2f}, U={vals['U']:.2f}")
    else:
        print('Optimization failed: ', res.message)
