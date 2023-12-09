import numpy as np
import random
from scipy.optimize import linprog

def generate_bids(n, m, p_bar):
    """
    Generate a sequence of random bids.
    :param n: Total number of bids.
    :param m: Number of items.
    :param p_bar: Ground truth price vector.
    :return: Array of bids.
    """
    bids = []
    for _ in range(n):
        # Generate a vector ak whose each entry is either zero or one at random
        ak = np.random.choice([0, 1], size=m)
        # Calculate bid price with randomization considered
        # Changing the Gauss variable should increase sensitivity
        # TODO(@mteffeteller): Re-run experiment with new Gauss
        # pi_k = np.dot(p_bar, ak) + np.random.normal(0, np.sqrt(0.2))
        pi_k = np.dot(p_bar, ak) + random.gauss(0, .02)  
        bids.append((ak, pi_k))
    return bids

def solve_offline_lp(bids, m, b_i):
    # Negative for maximization
    c = -np.array([pi_k for _, pi_k in bids])
    # Transpose to match dimensions
    A = np.array([a_k for a_k, _ in bids]).T
    b = b_i * np.ones(m)    

    # Solving the LP
    # TODO(@mteffeteller): solve this with > 1 solver for thoroughness
    result = linprog(c, A_ub=A, b_ub=b, bounds=(0, 1), method='highs')

    if result.success:
        # Revenue (negate because of maximization)
        # The current value of the objective function
        return -result.fun
    else:
        raise ValueError("Offline LP did not converge")

def solve_partial_lp_dual(bids, k, n, b_i):
    # Objective function: maximize sum(pi_j * x_j) for j=1 to kfb_
    c = -np.array([pi_k for _, pi_k in bids[:k]])

    # The inequality constraint matrix
    # Constraints: sum(a_ij * x_j) <= (k/n) * b_i for all i
    # Get dual price by taking the transpose of the original problem
    A_T = np.array([a_k for a_k, _ in bids[:k]]).T

    # The inequality constraint vector
    # Each element represents an upper bound on the corresponding value of A_ub @ x.
    b = (k / n) * np.array(b_i)

    # Bounds for decision variables: 0 <= x_j <= 1
    x_bounds = [(0, 1) for _ in range(k)]

    # Solve the linear program
    # 'highs' is the algorithm, but others can be used
    result = linprog(c, A_ub=A_T, b_ub=b, bounds=x_bounds, method='highs')

    if result.success:
        # The dual variable corresponding to the inequality constraints A_ub * x <= b_ub
        return result.get('slack')
    else:
        raise ValueError("failed to find a solution")
