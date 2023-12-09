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

def simulate(n, m, p_bar, k_values):
    # Regenerate bids with the fixed p_bar
    bids_fixed = common.generate_bids(n, m, p_bar)

    b_i = np.ones(m) * 1000  # Bid cap for all i

    # Get the various revenues generated up to k bids
    revenues = {}
    for k in k_values:
        revenue = run_slpm(bids_fixed, k, n, b_i)
        revenues[k] = revenue

    for k, revenue in revenues.items():
        print(f"SLPM revenue: {revenue} at k={k}")

    # Get the single revenue of offline LP
    # This should not be beat since it has all information
    offline_revenue_value = common.solve_offline_lp(bids_fixed, m, b_i)

    print(f"Offline revenue: {offline_revenue_value}")

if __name__ == "__main__":
    bid_count = 10000 
    items = 10
    k_values = [50, 100, 200, 1000, 5000]  # Different k values to test

    # Fixed ground truth price vector (p_bar) - set to ones for simplicity
    price_vector = np.ones(items)  # Vector of ones

    simulate(bid_count, items, price_vector, k_values)
 