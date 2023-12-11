import common
import numpy as np
from scipy.optimize import linprog


def solve_lp_get_dual_multi(bids, k, n, b_i, algo):
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
    # result = linprog(c, A_ub=A_T, b_ub=b, bounds=x_bounds, method='highs')
    result = linprog(c, A_ub=A_T, b_ub=b, bounds=x_bounds, method=f'{algo}')

    if result.success:
        # The dual variable corresponding to the inequality constraints A_ub * x <= b_ub
        # return result.get('slack')
        return  -1*result.ineqlin.get('marginals'), result.get('fun')
    else:
        raise ValueError("failed to find a solution")

# Problem 1
def run_slpm(bids, k, n, b_i, algo):
    revenue = 0
    remaining_capacity = np.array(b_i)
    
    # Solve the partial LP for the first k bids to get dual prices
    y_bar, current_rev = solve_lp_get_dual_multi(bids, k, n, b_i, algo)
    b_i -= y_bar

    for i, (a_k, pi_k) in enumerate(bids):
        if i >= k:
            # Allocate based on the decision rule using y_bar
            if pi_k > np.dot(a_k, y_bar) and all(remaining_capacity - a_k >= 0):
                revenue += pi_k
                remaining_capacity -= a_k

    return revenue

def sensitivity_ratio(generated_revenue, offline_revenue, k):
  """Calculates the sensitivity ratio given the generated revenue, offline revenue, and number of bids considered k."""
  ratio = generated_revenue / (offline_revenue)
  ratio = '{:.2f}'.format(ratio)
  return ratio


def simulate(n, m, p_bar, k_values):
    # Regenerate bids with the fixed p_bar
    algos = ['highs', 'highs-ds', 'highs-ipm']
    bids_fixed = common.generate_bids(n, m, p_bar)

    b_i = np.ones(m) * 1000  # Bid cap for all i

    # Get the various revenues generated up to k bids
    revenues_highs = {}
    for k in k_values:
        rev = run_slpm(bids_fixed, k, n, b_i, 'highs')
        revenues_highs[k] = rev

    revenues_highs_ds = {}
    for k in k_values:
        rev = run_slpm(bids_fixed, k, n, b_i, 'highs-ds')
        revenues_highs_ds[k] = rev

    revenues_highs_ipm = {}
    for k in k_values:
        rev = run_slpm(bids_fixed, k, n, b_i, 'highs-ipm')
        revenues_highs_ipm[k] = rev

    # Get the single revenue of offline LP
    # This should not be beat since it has all information
    offline_revenue = common.solve_offline_lp(bids_fixed, m, b_i)

    print(f"Offline revenue: {offline_revenue}")

    highs_ratios = []
    for k, revenue in revenues_highs.items():
        ratio = sensitivity_ratio(revenue, offline_revenue, k)
        ratio = ratio[2:]
        highs_ratios.append(round(int(ratio), 2))

    print(f'highs: {highs_ratios}%')

    highs_ds_ratios = []
    for k, revenue in revenues_highs_ds.items():
        ratio = sensitivity_ratio(revenue, offline_revenue, k)
        ratio = ratio[2:]
        highs_ds_ratios.append(round(int(ratio), 2))

    print(f'highs_ds: {highs_ds_ratios}%')

    highs_ipm_ratios = []
    for k, revenue in revenues_highs_ipm.items():
        ratio = sensitivity_ratio(revenue, offline_revenue, k)
        ratio = ratio[2:]
        highs_ipm_ratios.append(round(int(ratio), 2))

    print(f'highs_ipm: {highs_ipm_ratios}%')


if __name__ == "__main__":
    bid_count = 10000 
    items = 10
    k_values = [50, 75, 100, 150, 200, 250, 350, 500, 1000, 5000]  # Different k values to test

    # Fixed ground truth price vector (p_bar) - set to ones for simplicity
    price_vector = np.ones(items)  # Vector of ones

    simulate(bid_count, items, price_vector, k_values)
 