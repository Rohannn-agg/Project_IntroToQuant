"""Multi-Objective Evolutionary Optimizers: NSGA-II and MOEA/D.

Implements Pareto optimization across Expected Return, Volatility, and CVaR,
and 4 Pareto selection weighting schemes (uniform, linear, sigmoid, geometric).

References:
    - Deb, Pratap, Agarwal, Meyarivan (2002), "A Fast and Elitist Multiobjective
      Genetic Algorithm: NSGA-II", IEEE Transactions on Evolutionary Computation.
    - Zhang and Li (2007), "MOEA/D: A Multiobjective Evolutionary Algorithm
      Based on Decomposition", IEEE Transactions on Evolutionary Computation.
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np
from .objectives import evaluate_objectives, project_to_simplex


def fast_non_dominated_sort(objective_values: np.ndarray) -> List[List[int]]:
    """Partitions population into non-dominated Pareto fronts: F_1, F_2, ...

    Args:
        objective_values: shape (P, M) where all M objectives are to be MINIMIZED.
    """
    P = len(objective_values)
    domination_counts = np.zeros(P, dtype=int)
    dominated_solutions = [[] for _ in range(P)]
    fronts: List[List[int]] = [[]]

    for p in range(P):
        for q in range(P):
            if p == q:
                continue
            # p dominates q if p is <= in all objectives and < in at least one
            p_val = objective_values[p]
            q_val = objective_values[q]
            if np.all(p_val <= q_val) and np.any(p_val < q_val):
                dominated_solutions[p].append(q)
            elif np.all(q_val <= p_val) and np.any(q_val < p_val):
                domination_counts[p] += 1

        if domination_counts[p] == 0:
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return [f for f in fronts if len(f) > 0]


def calculate_crowding_distance(objective_values: np.ndarray, front: List[int]) -> np.ndarray:
    """Calculates crowding distances for solutions on a single Pareto front."""
    L = len(front)
    if L <= 2:
        return np.full(L, np.inf)

    distances = np.zeros(L, dtype=np.float64)
    front_objs = objective_values[front]  # shape (L, M)
    M = front_objs.shape[1]

    for m in range(M):
        sorted_idx = np.argsort(front_objs[:, m])
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf
        
        obj_range = front_objs[sorted_idx[-1], m] - front_objs[sorted_idx[0], m]
        if obj_range <= 1e-12:
            continue
            
        for i in range(1, L - 1):
            distances[sorted_idx[i]] += (
                front_objs[sorted_idx[i + 1], m] - front_objs[sorted_idx[i - 1], m]
            ) / obj_range

    return distances


class NSGA2PortfolioOptimizer:
    """NSGA-II implementation specialized for constrained portfolio optimization."""

    def __init__(
        self,
        pop_size: int = 60,
        n_generations: int = 40,
        crossover_prob: float = 0.90,
        mutation_prob: float = 0.15,
        max_asset_weight: float = 0.20,
        seed: int = 42,
    ):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_asset_weight = max_asset_weight
        self.rng = np.random.default_rng(seed)

    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        historical_returns: np.ndarray,
        cvar_alpha: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Runs NSGA-II to discover the 3-D Pareto optimal frontier.

        Returns:
            pareto_weights: (K, N) array of Pareto optimal portfolio weights
            pareto_objectives: (K, 3) array of objective values [ -Return, Volatility, CVaR ]
        """
        N = len(expected_returns)
        
        # 1. Initialize random population on the simplex
        raw_pop = self.rng.dirichlet(np.ones(N), size=self.pop_size)
        population = np.array([project_to_simplex(ind, self.max_asset_weight) for ind in raw_pop])

        # Evaluate initial population
        objs = np.array([
            evaluate_objectives(w, expected_returns, cov_matrix, historical_returns, cvar_alpha)
            for w in population
        ])

        # 2. Main Generation Loop
        for gen in range(self.n_generations):
            # Tournament Selection & Offspring Generation
            offspring = []
            while len(offspring) < self.pop_size:
                # Tournament
                i1, i2 = self.rng.integers(0, self.pop_size, size=2)
                p1 = population[i1] if objs[i1, 1] < objs[i2, 1] else population[i2]
                
                j1, j2 = self.rng.integers(0, self.pop_size, size=2)
                p2 = population[j1] if objs[j1, 1] < objs[j2, 1] else population[j2]

                # Simulated Binary Crossover (SBX)
                if self.rng.random() < self.crossover_prob:
                    u = self.rng.random()
                    eta_c = 15.0
                    beta = (2.0 * u) ** (1.0 / (eta_c + 1.0)) if u <= 0.5 else (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))
                    c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
                    c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # Polynomial Mutation
                for child in (c1, c2):
                    if self.rng.random() < self.mutation_prob:
                        noise = self.rng.normal(0, 0.05, size=N)
                        child += noise
                    offspring.append(project_to_simplex(child, self.max_asset_weight))

            offspring = np.array(offspring[: self.pop_size])
            offspring_objs = np.array([
                evaluate_objectives(w, expected_returns, cov_matrix, historical_returns, cvar_alpha)
                for w in offspring
            ])

            # 3. Elitist Selection: Combine parents and offspring (2N -> N)
            combined_pop = np.vstack([population, offspring])
            combined_objs = np.vstack([objs, offspring_objs])

            fronts = fast_non_dominated_sort(combined_objs)
            new_pop: List[np.ndarray] = []
            new_objs: List[np.ndarray] = []

            for front in fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend(combined_pop[front])
                    new_objs.extend(combined_objs[front])
                else:
                    # Partial front fill based on crowding distance
                    cd = calculate_crowding_distance(combined_objs, front)
                    sorted_by_cd = np.argsort(cd)[::-1]
                    spots_left = self.pop_size - len(new_pop)
                    chosen_idx = [front[idx] for idx in sorted_by_cd[:spots_left]]
                    new_pop.extend(combined_pop[chosen_idx])
                    new_objs.extend(combined_objs[chosen_idx])
                    break

            population = np.array(new_pop)
            objs = np.array(new_objs)

        # Extract Rank-1 (Pareto optimal) frontier
        rank1_front = fast_non_dominated_sort(objs)[0]
        return population[rank1_front], objs[rank1_front]


class MOEADPortfolioOptimizer:
    """MOEA/D with Tchebycheff decomposition approach."""

    def __init__(
        self,
        n_subproblems: int = 50,
        n_neighbors: int = 10,
        max_generations: int = 30,
        max_asset_weight: float = 0.20,
        seed: int = 42,
    ):
        self.n_subproblems = n_subproblems
        self.n_neighbors = min(n_neighbors, n_subproblems)
        self.max_generations = max_generations
        self.max_asset_weight = max_asset_weight
        self.rng = np.random.default_rng(seed)

    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        historical_returns: np.ndarray,
        cvar_alpha: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Runs MOEA/D decomposition across 3 objectives."""
        N = len(expected_returns)
        # 1. Generate uniform reference weight vectors across 3 objectives
        raw_weights = self.rng.dirichlet(np.ones(3), size=self.n_subproblems)
        lambda_weights = raw_weights / np.sum(raw_weights, axis=1, keepdims=True)

        # 2. Compute neighborhood matrix based on Euclidean distance of reference vectors
        dist_matrix = np.linalg.norm(lambda_weights[:, None, :] - lambda_weights[None, :, :], axis=2)
        neighborhoods = np.argsort(dist_matrix, axis=1)[:, : self.n_neighbors]

        # 3. Initial population
        population = np.array([
            project_to_simplex(self.rng.dirichlet(np.ones(N)), self.max_asset_weight)
            for _ in range(self.n_subproblems)
        ])
        objs = np.array([
            evaluate_objectives(w, expected_returns, cov_matrix, historical_returns, cvar_alpha)
            for w in population
        ])

        # Reference point z* (ideal point: min value for each objective)
        z_star = np.min(objs, axis=0)

        # 4. Evolutionary updates
        for gen in range(self.max_generations):
            for i in range(self.n_subproblems):
                # Pick two distinct parents from neighborhood
                p1_idx, p2_idx = self.rng.choice(neighborhoods[i], size=2, replace=False)
                p1, p2 = population[p1_idx], population[p2_idx]

                # Crossover
                child = project_to_simplex(0.5 * (p1 + p2) + self.rng.normal(0, 0.02, N), self.max_asset_weight)
                child_obj = np.array(evaluate_objectives(child, expected_returns, cov_matrix, historical_returns, cvar_alpha))

                # Update ideal point z*
                z_star = np.minimum(z_star, child_obj)

                # Tchebycheff objective update across neighborhood
                for j in neighborhoods[i]:
                    lam = lambda_weights[j]
                    current_te = np.max(lam * np.abs(objs[j] - z_star))
                    new_te = np.max(lam * np.abs(child_obj - z_star))
                    if new_te < current_te:
                        population[j] = child
                        objs[j] = child_obj

        # Filter non-dominated front
        fronts = fast_non_dominated_sort(objs)
        rank1 = fronts[0] if len(fronts) > 0 else list(range(len(objs)))
        return population[rank1], objs[rank1]


def select_pareto_portfolio(
    pareto_weights: np.ndarray,
    pareto_objs: np.ndarray,
    scheme: str = "linear",
    regime_state: int = 0,
) -> np.ndarray:
    """Selects a single compromise portfolio from the Pareto front using

    one of 4 weighting schemes: 'uniform', 'linear', 'sigmoid', or 'geometric'.
    
    Regime-aware: During turbulent regimes (state=1), shifts weight towards
    lower CVaR and lower volatility.
    """
    K = len(pareto_weights)
    if K == 1:
        return pareto_weights[0]

    # Objectives: f1 = -Return, f2 = Volatility, f3 = CVaR
    returns = -pareto_objs[:, 0]
    vols = pareto_objs[:, 1]
    cvars = pareto_objs[:, 2]

    # Calculate Sharpe approximation for each Pareto point
    sharpes = returns / np.maximum(vols, 1e-6)

    if scheme == "uniform":
        weights = np.ones(K) / K

    elif scheme == "linear":
        # Linearly weighted by Sharpe ratio
        min_s = np.min(sharpes)
        shifted_s = sharpes - min_s + 0.1
        weights = shifted_s / np.sum(shifted_s)

    elif scheme == "sigmoid":
        # Sigmoid function centered around median Sharpe
        # In turbulent regime, penalizes CVaR heavily
        if regime_state == 1:  # Crisis / Turbulent regime
            score = -cvars
        else:
            score = sharpes
        z = (score - np.mean(score)) / max(np.std(score), 1e-6)
        sig = 1.0 / (1.0 + np.exp(-z))
        weights = sig / np.sum(sig)

    elif scheme == "geometric":
        # Geometrically decaying weights favoring the highest Sharpe points
        rank_order = np.argsort(sharpes)[::-1]
        decay = 0.85
        raw_geom = decay ** np.arange(K)
        weights = np.zeros(K)
        weights[rank_order] = raw_geom
        weights /= np.sum(weights)

    else:
        raise ValueError(f"Unknown Pareto weighting scheme: {scheme}")

    # Return weighted average composite portfolio
    composite_w = np.sum(pareto_weights * weights[:, None], axis=0)
    return composite_w / np.sum(composite_w)
