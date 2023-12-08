import numpy as np
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
        a_k = np.random.choice([0, 1], size=m)  # Generate a_k
        pi_k = np.dot(p_bar, a_k) + np.random.normal(0, np.sqrt(0.2))  # Calculate bid price
        bids.append((a_k, pi_k))
    return bids

def solve_offline_lp(bids, m, b_i):
    c = -np.array([pi_k for _, pi_k in bids])  # Negative for maximization
    A = np.array([a_k for a_k, _ in bids]).T  # Transpose to match dimensions
    b = b_i * np.ones(m)    

    # Solving the LP
    result = linprog(c, A_ub=A, b_ub=b, bounds=(0, 1), method='highs')

    if result.success:
        return -result.fun  # Revenue (negate because of maximization)
    else:
        raise ValueError("Offline LP did not converge")

def solve_partial_lp_dual(bids, k, n, b_i):
    # Objective function: maximize sum(pi_j * x_j) for j=1 to k
    c = -np.array([pi_k for _, pi_k in bids[:k]])

    # Constraints: sum(a_ij * x_j) <= (k/n) * b_i for all i
    A = np.array([a_k for a_k, _ in bids[:k]]).T
    b = (k / n) * np.array(b_i)
    
    # Bounds for decision variables: 0 <= x_j <= 1
    x_bounds = [(0, 1) for _ in range(k)]

    # Solve the linear program
    result = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds, method='highs')

    if result.success:
        # The dual variable corresponding to the inequality constraints A_ub * x <= b_ub
        return result.get('slack')
    else:
        raise ValueError("failed to find a solution")
