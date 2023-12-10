import common
import numpy as np
import q1

# Problem 2
def dynamic_slpm(bids, n, b_i):
    interval = 50
    steps = 2
    remaining_capacity = np.array(b_i)
    dual = None
    revenue = 0
    for i, (a_k, pi_k) in enumerate(bids):
        # Update dual prices at specified steps
        if i == interval:
            print(f'i: = {i} \n dual: {dual}')
            dual, current_revenue = common.solve_lp_get_dual(bids, i + 1, n, remaining_capacity)
            interval*= steps
        # Allocate based on the decision rule using y_bar
        if dual is not None and pi_k > np.dot(a_k, dual) and all(remaining_capacity - a_k >= 0):
            revenue += pi_k
            remaining_capacity -= a_k
    revenue += -(current_revenue)
    return revenue


def run_simulation_problem_2(n, m, p_bar):
    # Regenerate bids with the fixed p_bar
    bids_fixed = common.generate_bids(n, m, p_bar)

    b_i = np.ones(m) * 1000  # Bid cap for all i
    dynamic_rev = dynamic_slpm(bids_fixed, n, b_i)
    print(f'Dynamic SLPM revenue: {dynamic_rev}')

    offline_revenue_value = common.solve_offline_lp(bids_fixed, m, b_i)

    print(f"Offline revenue: {offline_revenue_value}")

if __name__ == "__main__":
    # Simulation parameters
    n = 10000  # Total number of bids
    m = 10     # Number of items
    k_values = [50, 100, 200]  # Different k values to test

    # # Fixed ground truth price vector (p_bar) - set to ones for simplicity
    p_bar_fixed = np.ones(m)  # Vector of ones

    run_simulation_problem_2(n, m, p_bar_fixed)

    # i: = 50 
    # dual: None
    # i: = 100 
    # dual: [1.00026286 1.00279371 0.99972416 0.99329563 1.00696832 0.98529363
    # 1.01031407 0.99965332 1.01005201 1.01862897]
    # ...
    # i: = 6400 
    # dual: [1.00526488 1.00716264 1.00307658 1.00416318 1.00431024 1.00561206
    # 1.00351953 1.00194944 1.00306047 1.00441075]
    # Dynamic SLPM revenue: 10014.324607038357
    # Offline revenue: 10057.733449063195