import common
import numpy as np

# Problem 2
def run_slpm_revised(bids, k_values, n, m, b_i):
    k_values = [50, 100, 200]  # Different k values to test
    revenue = 0
    remaining_capacity = np.array(b_i)
    y_bar = None

    for i, (a_k, pi_k) in enumerate(bids):
        # Update dual prices at specified points
        if i == 0 or i in k_values:
            print('i = k at ', i)
            y_bar = common.solve_partial_lp_dual(bids, i + 1, n, remaining_capacity)

        # Allocate based on the decision rule using y_bar
        if y_bar is not None and pi_k > np.dot(a_k, y_bar) and all(remaining_capacity - a_k >= 0):
            revenue += pi_k
            remaining_capacity -= a_k

    return revenue


def run_simulation_problem_2(n, m, p_bar, k_values):
    # Regenerate bids with the fixed p_bar
    bids_fixed = common.generate_bids(n, m, p_bar)

    b_i = np.ones(m) * 1000  # Bid cap for all i

    updated_revenues = {}
    for k in k_values:
        revenue = run_slpm_revised(bids_fixed, k, n, m, b_i)
        updated_revenues[k] = revenue
    
    for k, revenue in updated_revenues.items():
        print(f"SLPM revenue: {revenue} at k={k}")

    offline_revenue_value = common.solve_offline_lp(bids_fixed, m, b_i)

    print(f"Offline revenue: {offline_revenue_value}")

if __name__ == "__main__":
    # Simulation parameters
    n = 10000  # Total number of bids
    m = 10     # Number of items
    k_values = [50, 100, 200]  # Different k values to test

    # # Fixed ground truth price vector (p_bar) - set to ones for simplicity
    p_bar_fixed = np.ones(m)  # Vector of ones

    run_simulation_problem_2(n, m, p_bar_fixed, k_values)
    # SLPM revenue: 10013.070557996842 at k=50
    # SLPM revenue: 10015.335887968598 at k=100
    # SLPM revenue: 10010.26340969867 at k=200
    # Offline revenue: 11281.89486664792