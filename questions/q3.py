import common
import numpy as np

def ahlda(bids, k, n, b_i, opt):
    """
    Action-history-dependent Learning Algorithm.
    """
    incremental_rev = 0
    remaining_capacity = np.array(b_i)
    performance = []

    for i in range(1, k + 1):
        # Update dual prices
        y_bar, rev = common.solve_lp_get_dual(bids, i, n, remaining_capacity)

        # Make decision for bid i
        a_k, pi_k = bids[i - 1]
        if pi_k > np.dot(a_k, y_bar) and all(remaining_capacity - a_k >= 0):
            incremental_rev += pi_k
            remaining_capacity -= a_k
        
        # Compute performance metric
        if not performance:
            last_rev = 0
        if len(performance) > 0:
            last_rev = performance[-1]
        performance.append(last_rev + (incremental_rev - (i / n) * opt)*-1)
        # performance.append((incremental_rev - (i / n) * opt)*-1)

    return performance

def slpm_problem_3(bids, k_values, n, b_i, opt):
    """
    SLPM algorithm with dynamic shadow price updates
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
                y_bar, rev = common.solve_lp_get_dual(bids, i + 1, n, remaining_capacity)

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
    p_bar = np.ones(m)
    b_i = np.ones(m) * 1000
    k_values = [50, 100, 200]
    k_step = 50

    # Generate bids
    bids = common.generate_bids(n, m, p_bar)

    # Solve offline problem for OPT
    opt = common.solve_offline_lp(bids, m, b_i)

    # AHDLA algorithm performance
    ahd_performance = {}
    for k in k_values:
        ahd_performance[k] = ahlda(bids, k, n, b_i, opt)

    print('Incremental AHD Algorithm Performance:')
    print(ahd_performance[200][:10])

    # SLPM algorithm performance
    slpm_performance = slpm_problem_3(bids, k_values, n, b_i, opt)

    print('Incremental Algorithm Performance:')
    print(slpm_performance[200][:10])
