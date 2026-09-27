## Forecasting with Fractional Brownian Motion: A Financial Perspective

This repository is based on the paper Forecasting with Fractional Brownian Motion: A Financial Perspective (Garcin, 2022).

The paper proposes a framework for forecasting high-frequency financial returns using a fractional Brownian motion (fBm) hypothesis to model log-prices. It also provides a number of results that are useful for constructing and evaluating a trading strategy based on the resulting forecasts. The results derived in the paper allow the analyst to optimize, to some extent, both the forecasting rule and the trading strategy.

This repository implements the formulas derived in the paper and replicates the numerical results presented therein, providing practical tools for generating forecasts, evaluating their performance, and assessing the proposed trading strategy.

Garcin, M. (2022). Forecasting with fractional Brownian motion: a financial perspective. Quantitative Finance, 22(8), 1495–1512. https://doi.org/10.1080/14697688.2022.2071758

## Content of the repository
`functions.py` — a collection of the formulas derived in the paper and the related optimization routines;\
`numerics.py` — a replication of the figures and tables reporting the numerical results in the paper.
