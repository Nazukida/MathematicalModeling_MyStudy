"""
============================================================
Multi-objective Nonlinear Programming Model for Sustainable Tourism
============================================================
This script implements the mathematical model described in the paper:
"Economy, Ecology, and Social Welfare: A Win-Win Approach for Sustainable Tourism in Juneau" (Team #2501687).

Objective: Maximize Total Social Utility (U) by optimizing tourist numbers, government investment, and taxation.
"""

import numpy as np
from scipy.optimize import minimize

# ============================================================
# Parameter Definitions
# ============================================================
class Parameters:
    def __init__(self):
        self.A = 16822  # Tourist flow amplitude
        self.B = 5514   # Tourist flow baseline
        self.p = 100    # Profit per tourist
        self.SCC = 190  # Social Cost of Carbon ($/ton)
        self.e = 66.13  # Emissions per tourist (kg)
        self.ERI_max = 2e5  # Max ecosystem resilience
        self.beta = 1e-4  # Resilience decay coefficient
        self.Gamma_1m = 1e8  # Max environmental investment return
        self.Gamma_2m = 1e8  # Max social investment return
        self.alpha_1 = 1e-4  # Environmental investment efficiency
        self.alpha_2 = 1e-4  # Social investment efficiency
        self.Gamma_10 = 1e4  # Base environmental return
        self.Gamma_20 = 1e4  # Base social return

# ============================================================
# Decision Variables
# ============================================================
class DecisionVariables:
    def __init__(self, c1, c2, I, gamma1, x1, x2):
        self.c1 = c1  # Peak season tourist cap
        self.c2 = c2  # Off-peak tourist target
        self.I = I    # Government investment
        self.gamma1 = gamma1  # Environmental investment proportion
        self.gamma2 = 1 - gamma1  # Social investment proportion
        self.x1 = x1  # Peak tax adjustment
        self.x2 = x2  # Off-peak subsidy adjustment

# ============================================================
# Model Equations
# ============================================================
def natural_demand(t, A, B):
    """Calculate natural tourist demand."""
    return max(-A * np.cos(2 * np.pi * t / 365) + B, 0)

def actual_demand(t, c1, c2, A, B):
    """Calculate actual tourist demand based on policy."""
    N0 = natural_demand(t, A, B)
    return min(N0, c1) if 121 <= t <= 270 else max(N0, c2)

def economic_profits(params, decision_vars):
    """Calculate total economic profits."""
    total_profit = 0
    for t in range(1, 366):
        N_t = actual_demand(t, decision_vars.c1, decision_vars.c2, params.A, params.B)
        f_t = (decision_vars.x1 - decision_vars.x2) / 2 * np.cos(2 * np.pi * t / 365 + np.pi) + (decision_vars.x1 + decision_vars.x2) / 2
        total_profit += N_t * params.p + (f_t - decision_vars.I)
    return total_profit

def environmental_level(params, decision_vars):
    """Calculate environmental level."""
    total_environment = 0
    for t in range(1, 366):
        N_t = actual_demand(t, decision_vars.c1, decision_vars.c2, params.A, params.B)
        E_cost = N_t * params.e * params.SCC / 1000
        ERI_t = params.ERI_max / (1 + params.beta * N_t)
        Gamma_1 = params.Gamma_1m / (1 + (params.Gamma_1m / params.Gamma_10 - 1) * np.exp(-params.alpha_1 * (decision_vars.gamma1 * decision_vars.I - params.Gamma_10)))
        total_environment += -E_cost + ERI_t + Gamma_1
    return total_environment

def social_welfare(params, decision_vars):
    """Calculate social welfare level."""
    total_social = 0
    for t in range(1, 366):
        N_t = actual_demand(t, decision_vars.c1, decision_vars.c2, params.A, params.B)
        S1_t = 7.774e6 / 365  # Constant employment benefit
        S2_t = 0.1 * N_t  # Negative social impact (example coefficient)
        Gamma_2 = params.Gamma_2m / (1 + (params.Gamma_2m / params.Gamma_20 - 1) * np.exp(-params.alpha_2 * (decision_vars.gamma2 * decision_vars.I - params.Gamma_20)))
        total_social += S1_t - S2_t + Gamma_2
    return total_social

def total_utility(params, decision_vars):
    """Objective function: Maximize total utility."""
    P = economic_profits(params, decision_vars)
    E = environmental_level(params, decision_vars)
    S = social_welfare(params, decision_vars)
    return P + E + S

# ============================================================
# Optimization
# ============================================================
def optimize_model(params):
    """Optimize the model to maximize total utility."""
    def objective(x):
        decision_vars = DecisionVariables(c1=x[0], c2=x[1], I=x[2], gamma1=x[3], x1=x[4], x2=x[5])
        utility = total_utility(params, decision_vars)
        print(f"Testing variables: {x}, Utility: {utility}")  # Debugging line
        return -utility  # Negative for maximization

    # Constraints
    def cons_c1_c2(x):
        return x[0] - x[1]  # c1 >= c2
    def cons_x1_x2(x):
        return x[4] - x[5]  # x1 >= x2
    def cons_env(x):
        decision_vars = DecisionVariables(c1=x[0], c2=x[1], I=x[2], gamma1=x[3], x1=x[4], x2=x[5])
        E = environmental_level(params, decision_vars)
        return E  # E >= 0
    def cons_soc(x):
        decision_vars = DecisionVariables(c1=x[0], c2=x[1], I=x[2], gamma1=x[3], x1=x[4], x2=x[5])
        S = social_welfare(params, decision_vars)
        return S  # S >= 0

    constraints = [
        {'type': 'ineq', 'fun': cons_c1_c2},
        {'type': 'ineq', 'fun': cons_x1_x2},
        {'type': 'ineq', 'fun': cons_env},
        {'type': 'ineq', 'fun': cons_soc},
    ]

    # Initial guesses
    x0 = [20000, 15000, 1e6, 0.5, 100, 50]

    # Bounds for decision variables
    bounds = [
        (15000, 25000),  # c1
        (10000, 20000),  # c2
        (0, params.B * params.p),  # I
        (0, 1),  # gamma1
        (50, 150),  # x1
        (0, 100)   # x2
    ]

    result = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')
    print(f"Optimization result: {result}")  # Debugging line
    return result

# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    params = Parameters()
    result = optimize_model(params)

    if result.success:
        print("Optimization Successful!")
        print("Optimal Decision Variables:")
        print(f"c1 (Peak Cap): {result.x[0]:.2f}")
        print(f"c2 (Off-Peak Target): {result.x[1]:.2f}")
        print(f"I (Investment): {result.x[2]:.2f}")
        print(f"gamma1 (Environmental Allocation): {result.x[3]:.2f}")
        print(f"x1 (Peak Tax): {result.x[4]:.2f}")
        print(f"x2 (Off-Peak Subsidy): {result.x[5]:.2f}")
    else:
        print("Optimization Failed.")
