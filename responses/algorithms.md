Algorithms for consideration:

1. Highs Simplex: This algorithm iteratively explores the vertices of the feasible region until it finds the optimal solution
2. Highs Dual Simplex: This algorithm focuses on the dual problem and iterates through its vertices to find the optimal solution.
3. Highs Interior Point: This algorithm takes advantage of barrier functions to find the optimal solution within the interior of the feasible region.

Output from code:

```
highs: [86, 84, 90, 93, 88, 87, 91, 95, 96, 54]
highs_ds: [67, 84, 89, 93, 87, 86, 91, 94, 94, 54]
highs_ipm: [67, 83, 89, 93, 86, 86, 90, 93, 94, 53]
```

Key findings:
- After running the simulation on the same n=10,000 dataset across
  3 different algorithms (highs "simplex", highs-ds "dual simplex", 
  high-ipm "interior point") we can observe that the highs simplex method
  consistently achieves revenue that best approaches the optimal result (the
  offline revenue)

- Highs-ds and highs-ipm algorithms exhibit similar performance: While both highs-ds and highs-ipm algorithms achieve lower revenue than the highs simplex method, their performance is relatively consistent with each other. This indicates that both algorithms may be suitable for scenarios where achieving the absolute highest revenue is not critical, but speed or other factors might be more important.