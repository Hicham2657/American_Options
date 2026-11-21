# American Options

My solutions to two lab sessions on American option pricing from the American Options course at Ensimag, in Python. Both price an American put.

## Contents

| Notebook | Topic | What it does |
|---|---|---|
| `american_option_crr_tree.ipynb` | Binomial tree | Prices an American put in the Cox-Ross-Rubinstein model by backward induction, measures the $1/N$ convergence of the price, and checks that the delta hedging strategy replicates the option exactly along simulated paths. |
| `rogers_dual_american_options.ipynb` | Dual method | Computes Monte Carlo upper bounds on the price with Rogers' dual formulation, using either the European put martingale or a discrete delta hedge, and compares them with Bermudan and American prices from a binomial tree. |

## References

The exercise statements  (énoncés) come from the course.

- L. C. G. Rogers, *Monte Carlo valuation of American options*, Mathematical Finance, 2002.
