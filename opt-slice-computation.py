#!/usr/bin/env python
# coding: utf-8

"""Generic proportional-fair slice optimization for any number of links.

Model notes
-----------
- A "link" can represent any STA<->AP communication edge.
- For N links, decision variables are created for every non-empty active-set subset,
  i.e., 2^N - 1 variables.
- Each variable x_S denotes the resource fraction assigned when exactly links in set
  S are active.
- The optimizer maximizes sum(log(rate_i)) over a chosen set of target links.

rate_i = beta_i * sum_{S: i in S} x_S * alpha_{i,S}

where alpha_{i,S} is the normalized throughput factor of link i when S is active.
"""

from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize


Subset = Tuple[int, ...]  # sorted tuple of 0-based link indices


def non_empty_subsets(n_links: int) -> List[Subset]:
    """Return all non-empty subsets over [0, n_links)."""
    return [
        subset
        for r in range(1, n_links + 1)
        for subset in combinations(range(n_links), r)
    ]


def bitmask_to_indices(mask: str) -> Tuple[int, ...]:
    """Convert bitmask like '1011' into active link indices (0-based)."""
    return tuple(i for i, bit in enumerate(mask) if bit == "1")


def build_alpha(
    beta: Sequence[float],
    measurements: Dict[Subset, Sequence[float]],
) -> Dict[Subset, np.ndarray]:
    """Build normalized alpha factors from measured throughputs.

    Parameters
    ----------
    beta:
        Isolated throughput per link (length N).
    measurements:
        Mapping from subset S to a length-N throughput vector where non-members of S
        can be 0. For members i in S, alpha_{i,S} = throughput_{i,S} / beta_i.

    Returns
    -------
    Dict[Subset, np.ndarray]
        For each subset, an array length N where valid factors are filled for members
        and zeros elsewhere.
    """
    n_links = len(beta)
    beta_arr = np.asarray(beta, dtype=float)

    alpha: Dict[Subset, np.ndarray] = {}

    for subset, throughput_vec in measurements.items():
        subset = tuple(sorted(subset))
        throughput_arr = np.asarray(throughput_vec, dtype=float)
        if throughput_arr.shape != (n_links,):
            raise ValueError(
                f"Subset {subset}: expected throughput vector length {n_links}, "
                f"got {throughput_arr.shape}."
            )

        factors = np.zeros(n_links, dtype=float)
        for i in subset:
            if beta_arr[i] <= 0:
                raise ValueError(f"beta[{i}] must be positive.")
            factors[i] = throughput_arr[i] / beta_arr[i]
        alpha[subset] = factors

    # Ensure singleton subsets exist (default alpha=1 for own link).
    for i in range(n_links):
        singleton = (i,)
        if singleton not in alpha:
            factors = np.zeros(n_links, dtype=float)
            factors[i] = 1.0
            alpha[singleton] = factors

    return alpha


def objective(
    x: np.ndarray,
    subsets: Sequence[Subset],
    alpha: Dict[Subset, np.ndarray],
    beta: np.ndarray,
    target_links: Iterable[int],
) -> float:
    """Negative proportional-fair utility for SciPy minimization."""
    target_links = tuple(target_links)

    rates = np.zeros_like(beta)
    for x_j, subset in zip(x, subsets):
        factors = alpha.get(subset)
        if factors is None:
            continue
        rates += x_j * factors * beta

    # Penalize invalid/non-positive rates to keep log well-defined.
    min_rate = np.min(rates[list(target_links)])
    if min_rate <= 0:
        return 1e12 + abs(min_rate) * 1e6

    return -np.sum(np.log(rates[list(target_links)]))


def simplex_constraint(x: np.ndarray) -> float:
    """Constraint enforcing sum(x)=1."""
    return np.sum(x) - 1.0


def optimize_slices(
    beta: Sequence[float],
    measurements: Dict[Subset, Sequence[float]],
    active_mask: str = None,
    active_links: Sequence[int] = None,
    seed: int = 7,
):
    """Solve the generic slice optimization problem.

    Exactly one of `active_mask` or `active_links` should be provided.
    """
    n_links = len(beta)
    beta_arr = np.asarray(beta, dtype=float)

    if active_mask is not None and active_links is not None:
        raise ValueError("Provide only one of active_mask or active_links.")
    if active_mask is None and active_links is None:
        active_links = tuple(range(n_links))
    elif active_mask is not None:
        if len(active_mask) != n_links:
            raise ValueError(
                f"active_mask length ({len(active_mask)}) must match n_links ({n_links})."
            )
        active_links = bitmask_to_indices(active_mask)
    else:
        active_links = tuple(sorted(active_links))

    if not active_links:
        raise ValueError("At least one active link is required.")

    subsets = non_empty_subsets(n_links)
    alpha = build_alpha(beta_arr, measurements)

    rng = np.random.default_rng(seed)
    x0 = rng.random(len(subsets))
    x0 /= x0.sum()

    bounds = [(0.0, 1.0)] * len(subsets)
    constraints = [{"type": "eq", "fun": simplex_constraint}]

    sol = minimize(
        objective,
        x0,
        args=(subsets, alpha, beta_arr, active_links),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return sol, subsets, active_links


if __name__ == "__main__":
    # Example equivalent to original 4-link case, but through generic data structures.
    beta = [101.71, 103.8, 78.06, 122.04]

    # Throughputs per active subset (0-based link indexing).
    measurements = {
        (0, 1): [50, 50, 0, 0],
        (0, 2): [101, 0, 39, 0],
        (0, 3): [101, 0, 0, 122],
        (1, 2): [0, 30, 30, 0],
        (1, 3): [0, 25, 0, 80],
        (2, 3): [0, 0, 29, 80],
        (0, 1, 2): [100, 2, 37, 0],
        (0, 1, 3): [80, 20, 0, 50],
        (0, 2, 3): [80, 0, 20, 50],
        (1, 2, 3): [0, 20, 30, 50],
        (0, 1, 2, 3): [50, 2, 10, 40],
    }

    # Active target links; here all links are active.
    sol, subsets, active_links = optimize_slices(
        beta=beta,
        measurements=measurements,
        active_mask="1111",
    )

    print("Active links:", active_links)
    print("Solver success:", sol.success)
    print("Objective value:", sol.fun)
    print("Top allocations (subset -> x):")

    x = sol.x
    top_idx = np.argsort(-x)[:10]
    for idx in top_idx:
        subset = subsets[idx]
        print(f"  {subset}: {x[idx]:.6f}")
