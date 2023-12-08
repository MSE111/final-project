import common
import numpy as np
from scipy.optimize import linprog

# Problem 1
def run_slpm(bids, k, n, b_i):
    revenue = 0
    remaining_capacity = np.array(b_i)
    
    # Solve the partial LP for the first k bids to get dual prices
    y_bar = common.solve_partial_lp_dual(bids, k, n, b_i)

    # Adjust capacity for each item based on the dual prices
    b_i -= y_bar

    for i, (a_k, pi_k) in enumerate(bids):
        if i >= k:
            # Allocate based on the decision rule using y_bar
            if pi_k > np.dot(a_k, y_bar) and all(remaining_capacity - a_k >= 0):
                revenue += pi_k
                remaining_capacity -= a_k

    return revenue

def run(n, m, p_bar, k_values):
    # Regenerate bids with the fixed p_bar
    bids_fixed = common.generate_bids(n, m, p_bar)

    b_i = np.ones(m) * 1000  # Bid cap for all i

    revenues = {}
    for k in k_values:
        revenue = run_slpm(bids_fixed, k, n, b_i)
        revenues[k] = revenue

    for slpm_revenue, k in zip(revenues.values(), k_values):
        print(f"SLPM revenue: {slpm_revenue} at k={k}")

    offline_revenue_value = common.solve_offline_lp(bids_fixed, m, b_i)

    print(f"Offline revenue: {offline_revenue_value}")

if __name__ == "__main__":
    n = 10000  # Total number of bids
    m = 10     # Number of items
    k_values = [50, 100, 200, 1000, 5000]  # Different k values to test

    # Fixed ground truth price vector (p_bar) - set to ones for simplicity
    price_vector = np.ones(m)  # Vector of ones

    run(n, m, price_vector, k_values)
 