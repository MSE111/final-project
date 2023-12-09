import common
import numpy as np

def ahlda(bids, k, n, m, b_i, opt):
    """
    Action-history-dependent Learning Algorithm.
    """
    revenue = 0
    remaining_capacity = np.array(b_i)
    performance = []

    for i in range(1, k + 1):
        # Update dual prices
        y_bar = common.solve_partial_lp_dual(bids, i, n, remaining_capacity)

        # Make decision for bid i
        a_k, pi_k = bids[i - 1]
        if pi_k > np.dot(a_k, y_bar) and all(remaining_capacity - a_k >= 0):
            revenue += pi_k
            remaining_capacity -= a_k
        
        # Compute performance metric
        performance.append(revenue - (i / n) * opt)

    return performance

def performance_metric_slpm_revised(bids, k_values, n, m, b_i, opt):
    """
    Run the revised SLPM algorithm with dynamic dual price updates and compute performance metrics.
    """
    performance = {}
    for k in k_values:
        revenue = 0
        remaining_capacity = np.array(b_i)
        y_bar = None
        performance_k = []

        for i, (a_k, pi_k) in enumerate(bids):
            if i >= k:  # Only consider bids after k
                break

            # Update dual prices at specified points or at the start
            if i in k_values or i == 0:
                y_bar = common.solve_partial_lp_dual(bids, i + 1, n, remaining_capacity)

            # Allocate based on the decision rule using y_bar
            if y_bar is not None and pi_k > np.dot(a_k, y_bar) and all(remaining_capacity - a_k >= 0):
                revenue += pi_k
                remaining_capacity -= a_k

            # Compute performance metric
            performance_k.append(revenue - (i / n) * opt)

        performance[k] = performance_k

    return performance

if __name__ == "__main__":
    # Parameters
    n = 10000
    m = 10
    p_bar_fixed = np.ones(m)
    b_i = np.ones(m) * 1000
    k_values = [50, 100, 200]

    # Generate bids
    bids = common.generate_bids(n, m, p_bar_fixed)

    # Solve offline problem for OPT
    opt = common.solve_offline_lp(bids, m, b_i)

    # AHDLA algorithm performance
    ahlda_performance = {k: ahlda(bids, k, n, m, b_i, opt) for k in k_values}

    print(ahlda_performance[200][:10])  # Display first 10 performance metrics for k=200

    # SLPM algorithm performance
    slpm_performance = performance_metric_slpm_revised(bids, k_values, n, m, b_i, opt)

    # Display the first 10 performance metrics for k=200 from SLPM
    print(slpm_performance[200][:10])