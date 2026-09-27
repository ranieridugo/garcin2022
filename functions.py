# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 13:30:15 2026

@author: ranieri dugo

Functions for reproducing the results in:

    Forecasting with fractional Brownian motion: a financial perspective
    Matthieu Garcin
    Quantitative finance, 2022
    
The code implements:
    1. Covariance and variance functions for fractional Gaussian noise.
    2. Forecasting of future increments from past observations.
    3. Forecast evaluation and optimal lag selection.
    4. A ternary trading strategy based on the forecast.
    5. Risk-adjusted strategy performance.

Notation follows the paper where possible.
"""

import math
import numpy as np
from scipy import stats
from scipy.optimize import Bounds, LinearConstraint, minimize, minimize_scalar

# ------------------------------------------------------------------------------
# 1. Covariance and variance functions for fractional Gaussian noise.
# ------------------------------------------------------------------------------

def cov_fgn(t, s, v, u, H, sigma = 1):
    """Covariance between two non-overlapping or overlapping increments of an fBm.

    Parameters
    ----------
    t, s : float or ndarray
        End and start times of the first increment, X(t) - X(s).
    v, u : float or ndarray
        End and start times of the second increment, X(v) - X(u).
    H : float
        Hurst exponent, H in (0, 1).
    sigma : float, optional
        Scale / volatility parameter of the fBm (default is 1).

    Returns
    -------
    float or ndarray
        Covariance E[(X(t) - X(s))(X(v) - X(u))].

    Notes
    -----
    This function implements Equation 2 of Garcin (2022):
        E[(X(t) - X(s))(X(v) - X(u))] = (sigma^2 / 2) * (|u - t|^{2H} + |v - s|^{2H} 
                                                        - |v - t|^{2H} - |u - s|^{2H}).
    It is the foundational covariance function used to construct both the 
    predictor covariance matrix Sigma_S (via `mcov_fgn`) and the cross-covariance 
    vector Sigma_RS (via `vcov_fgn`).
    """
    exp = 2 * H
    return (
        sigma ** 2 / 2 * (abs(u - t) ** exp + abs(v - s) ** exp -
                          abs(v - t) ** exp - abs(u - s) ** exp)
        )

def var_fgn(t, s, H, sigma = 1):
    """Variance of a fractional Brownian motion (fBm) increment over a time interval.

    Parameters
    ----------
    t : float or ndarray
        End time of the interval.
    s : float or ndarray
        Start time of the interval.
    H : float
        Hurst exponent, H in (0, 1).
    sigma : float, optional
        Scale / volatility parameter of the fBm (default is 1).

    Returns
    -------
    float or ndarray
        Variance Var(X(t) - X(s)) = sigma^2 * |t - s|^{2H}.

    Notes
    -----
    In the framework of Garcin (2022), this function computes the variance 
    of asset return increments over a duration |t - s|. For a forecast horizon 
    h = |t - s|, the variance of the future return R_{t, t+h} is:
        Var(R_{t, t+h}) = sigma^2 * h^{2H}.
    """
    return sigma ** 2 * abs(s - t) ** (2 * H)

def mcov_fgn(vd, H, sigma = 1):
    """Covariance matrix of observed historical return increments (Sigma_S).

    Parameters
    ----------
    vd : array-like
        Ordered vector of observation time lags including d_0 = 0:
            vd = [d_0, d_1, ..., d_n],
        with 0 = d_0 < d_1 < ... < d_n.
    H : float
        Hurst exponent, H in (0, 1).
    sigma : float, optional
        Scale / volatility parameter of the fBm (default is 1).

    Returns
    -------
    ndarray
        Symmetric (n x n) covariance matrix Sigma_S of past returns S, where 
        S_i = X(t - d_{i-1}) - X(t - d_i) for i = 1, ..., n.

    Notes
    -----
    In Garcin (2022), this corresponds to the matrix Sigma_S (or Sigma_{yy}), 
    representing the variance-covariance matrix of predictor inputs S. 
    Entry C[i, j] computes the covariance between incremental returns over 
    lags [d_{i-1}, d_i] and [d_{j-1}, d_j] using `cov_fgn`.
    """ 
    nd = len(vd)
    vd1 = vd[1 : ]
    vd2 = vd[: - 1]
    C = np.zeros((nd - 1, nd - 1))
    for i in range(len(vd1)):
        for j in range(i, len(vd2)):
            # t - di
            di = vd1[i] 
            # t - dj
            dj = vd2[i]
            # t - dk
            dk = vd1[j] 
            # t - dl
            dl = vd2[j]
            C[i, j] = cov_fgn(dk, dl, di, dj, H, sigma)
    C = C + C.T - np.diag(np.diag(C))
    return C

def vcov_fgn(h, vd, H, sigma = 1):
    """Cross-covariance vector between future return and past return increments (Sigma_RS).

    Parameters
    ----------
    h : float
        Forecast horizon (h > 0).
    vd : array-like
        Ordered vector of observation time lags including d_0 = 0:
            vd = [d_0, d_1, ..., d_n],
        with 0 = d_0 < d_1 < ... < d_n.
    H : float
        Hurst exponent, H in (0, 1).
    sigma : float, optional
        Scale / volatility parameter of the fBm (default is 1).

    Returns
    -------
    ndarray
        1D array of length n containing the cross-covariances between the future 
        return R_{t, t+h} = X(t+h) - X(t) and each past return increment 
        S_i = X(t - d_{i-1}) - X(t - d_i) for i = 1, ..., n.

    Notes
    -----
    In Garcin (2022), this corresponds to the row vector Sigma_RS (or Sigma_{zy}), 
    representing the cross-covariance between the target forecast return and the 
    lagged predictor inputs S. It is used alongside Sigma_S to determine the linear 
    filter weights and the explained predictive variance a^2.
    """
    return cov_fgn(np.array(vd[1 : ]), np.array(vd[ : - 1]), 0, - h, H, sigma)

# ------------------------------------------------------------------------------
# 2. Forecasting of future increments from past observations.
# ------------------------------------------------------------------------------

def forecast_fgn(h, vd, vy, H, sigma = 1):
    """Conditional return forecast given past observed return increments.

    Parameters
    ----------
    h : float
        Forecast horizon (h > 0).
    vd : array-like
        Ordered vector of observation time lags including d_0 = 0:
            vd = [d_0, d_1, ..., d_n],
        with 0 = d_0 < d_1 < ... < d_n.
    vy : array-like
        Vector of observed past return increments S = [S_1, ..., S_n]^T, where 
        S_i = X(t - d_{i-1}) - X(t - d_i).
    H : float
        Hurst exponent, H in (0, 1).
    sigma : float, optional
        Scale / volatility parameter of the fBm (default is 1).

    Returns
    -------
    float or ndarray
        Conditional expectation E[R_{t, t+h} | S] = Sigma_RS * Sigma_S^{-1} * S.

    Notes
    -----
    Implements the linear Gaussian projection for predicting future return
    R_{t, t+h} = X(t+h) - X(t) based on the history of observed returns S.
    """
    mxy = vcov_fgn(h, vd, H, sigma)
    myy = mcov_fgn(vd, H, sigma)
    return mxy @ np.linalg.inv(myy) @ vy

def forecast_var(h, vd, H, sigma = 1):
    """Explained variance of the linear predictor (a^2).

    Parameters
    ----------
    h : float
        Forecast horizon (h > 0).
    vd : array-like
        Ordered vector of observation time lags including d_0 = 0:
            vd = [d_0, d_1, ..., d_n],
        with 0 = d_0 < d_1 < ... < d_n.
    H : float
        Hurst exponent, H in (0, 1).
    sigma : float, optional
        Scale / volatility parameter of the fBm (default is 1).

    Returns
    -------
    float
        Predictive variance a^2 = Sigma_RS * Sigma_S^{-1} * Sigma_RS^T.

    Notes
    -----
    In Garcin (2022), a^2 is the variance of the forecast E[R_{t, t+h} | S]. 
    This key scalar quantity summarizes the total predictive information 
    extracted from the lag structure vd and directly determines both the 
    theoretical hit ratio and risk-adjusted strategy performance.
    """
    mxy = vcov_fgn(h, vd, H, sigma)
    myy = mcov_fgn(vd, H, sigma)
    return mxy @ np.linalg.inv(myy) @ mxy.T

def residual_var(h, vd, H, sigma = 1):
    """Conditional variance of the forecast error (mean squared error).

    Parameters
    ----------
    h : float
        Forecast horizon (h > 0).
    vd : array-like
        Ordered vector of observation time lags including d_0 = 0:
            vd = [d_0, d_1, ..., d_n],
        with 0 = d_0 < d_1 < ... < d_n.
    H : float
        Hurst exponent, H in (0, 1).
    sigma : float, optional
        Scale / volatility parameter of the fBm (default is 1).

    Returns
    -------
    float
        Residual variance Var(R_{t, t+h} | S) = sigma^2 * h^{2H} - a^2.

    Notes
    -----
    In Garcin (2022), this measures the unexplained variance (or prediction MSE) 
    remaining after conditioning on past observed returns S. It equals the 
    unconditional return variance Var(R_{t, t+h}) minus the explained 
    predictive variance a^2.
    """
    return sigma ** 2 * h ** (2 * H) - forecast_var(h, vd, H, sigma)

# ------------------------------------------------------------------------------
# 3. Forecast evaluation and optimal lag selection.
# ------------------------------------------------------------------------------

def hitratio_fgn(h, vd, H, sigma = 1):
    """Theoretical non-conditional hit ratio for an fBm return predictor.

    Parameters
    ----------
    h : float
        Forecast horizon (h > 0).
    vd : array-like
        Ordered vector of observation time lags including d_0 = 0:
            vd = [0, delta_1, delta_2, ..., delta_n],
        with 0 = d_0 < delta_1 < ... < delta_n.
    H : float
        Hurst exponent, H in (0, 1).
    sigma : float, optional
        Scale / volatility parameter of the fBm (default is 1).

    Returns
    -------
    float
        Theoretical non-conditional probability rho in (0.5, 1) of correctly 
        predicting the directional sign of the future return R_{t, t+h}.

    Notes
    -----
    Implements Theorem 1 of Garcin (2022):
        rho = 1 - (1 / pi) * arctan( sqrt( (sigma^2 * h^{2H}) / (Sigma_RS * Sigma_S^{-1} * Sigma_RS^T) - 1 ) )
    where:
    - Sigma_S is the (n x n) covariance matrix of past returns S.
    - Sigma_RS is the (1 x n) cross-covariance vector between future return 
      R_{t, t+h} and past returns S.
    """
    mxy = vcov_fgn(h, vd, H, sigma)
    myy = mcov_fgn(vd, H, sigma)
    arg = sigma ** 2 * h ** (2 * H) / (mxy @ np.linalg.inv(myy) @ mxy)
    return 1 - 1 / np.pi * np.arctan(np.sqrt(arg - 1))

def optimize_duration_full(h, n, H, sigma = 1, x0 = None):
    """Optimize time lags maximizing hit ratio over full n-dimensional space.

    Parameters
    ----------
    h : float
        Forecast horizon (h > 0).
    n : int
        Number of historical lag durations to include in the predictor (n >= 1).
    H : float
        Hurst exponent, H in (0, 1).
    sigma : float, optional
        Volatility parameter of the fBm (default is 1).
    x0 : array-like, optional
        Initial guess for the full lag vector of length n. If None, symmetric
        initial values around h are generated.

    Returns
    -------
    dict or float
        If n == 1, returns h directly (analytical solution).
        Otherwise returns dictionary with:
        - 'durations' : ndarray of length n
            Optimal ordered lag vector [delta_1*, delta_2*, ..., delta_n*].
        - 'success' : bool
            Optimization success flag from scipy.optimize.minimize.
        - 'hitratio' : float
            Maximized theoretical non-conditional hit ratio.

    Notes
    -----
    Performs numerical optimization over all n lags subject to the sequential 
    ordering constraint 0 < delta_1 < delta_2 < ... < delta_n.
    For n = 1, it executes an analytical early exit returning delta_1* = h.
    """
    if n == 1:
        return h
    
    if x0 is None:
        stop = h - h / 5 if n % 2 == 0 else h - h / 3
        x0a = np.linspace(h / 5, stop, n // 2)
        x0 = np.concatenate([x0a, [h] if n % 2 else [], h**2 / x0a[::-1]])
    else:
        x0 = np.asarray(x0, dtype=float)
        if x0.shape[0] != n:
            raise ValueError(f"x0 has length {x0.shape[0]}, expected {n}")
    
    def objective(x):
        return - hitratio_fgn(h, np.r_[0, x], H, sigma)
    
    A_order = np.eye(n, k = 1) - np.eye(n)
    order_constraint = LinearConstraint(A_order[:-1], lb = 1e-3, ub = np.inf)
    
    res = minimize(objective, x0, method = 'trust-constr', 
                   constraints = order_constraint, 
                   bounds = Bounds(lb = 1e-3, ub = np.inf))
    
    return {'durations': res.x, 'success': res.success, 'hitratio': - res.fun}

def optimize_duration_reduced(h, n, H, sigma = 1, x0 = None):
    """Optimize time lags maximizing hit ratio using inverse symmetry parameter reduction.

    Parameters
    ----------
    h : float
        Forecast horizon (h > 0).
    n : int
        Number of historical lag durations to include in the predictor (n >= 1).
    H : float
        Hurst exponent, H in (0, 1).
    sigma : float, optional
        Volatility parameter of the fBm (default is 1).
    x0 : array-like, optional
        Initial guess for the reduced decision vector x of length m = floor(n / 2).
        If None, spaced initial values below h are automatically generated.

    Returns
    -------
    dict or float
        If n == 1, returns h directly.
        Otherwise returns dictionary with:
        - 'durations' : ndarray of length n
            Reconstructed full optimal lag vector [delta_1*, ..., delta_n*].
        - 'success' : bool
            Optimization success flag from scipy.optimize.minimize.
        - 'hitratio' : float
            Maximized theoretical non-conditional hit ratio.

    Notes
    -----
    Exploits the theoretical inverse symmetry relation numerically observed in
    the paper's experiment':
        delta_i* * delta_{n+1-i}* = h^2
    This reduces the decision vector dimension to m = floor(n / 2), setting 
    the central lag to h when n is odd, and guaranteeing that all reconstructed 
    lags satisfy the geometric symmetry around h.
    """
    if n == 1:
        return h
        
    m = n // 2
    
    if x0 is None:
        ub = 5 if n % 2 == 0 else 3
        x0 = np.linspace(start = h / 5, stop = h - h / ub, num = m, dtype = float)
    else:
        x0 = np.asarray(x0, dtype=float)
        if x0.shape[0] != m:
            raise ValueError(f"x0 has length {x0.shape[0]}, expected {n}")
    
    def get_full_lags(x):
        if n % 2 == 0: 
            return np.concatenate([x, h ** 2 / x[::-1]])
        else:
            return np.concatenate([x, [h], h ** 2 / x[::-1]])
        
    def objective(x):
        xf = get_full_lags(x)
        return - hitratio_fgn(h, np.r_[0, xf], H, sigma)
    
    constraints = []
    if m > 1:
        A_order = np.eye(m, k = 1) - np.eye(m)
        order_constraint = LinearConstraint(A_order[:-1], lb = 1e-2, ub = np.inf)
        constraints.append(order_constraint)
    res = minimize(objective, x0, method = 'trust-constr',
                   constraints = constraints, 
                   bounds = Bounds(lb = 1e-2, ub = h - 1e-2))
    
    return {'durations': get_full_lags(res.x), 'success': res.success, 'hitratio': -res.fun}

def optimize_duration(h, n, H, sigma = 1, x0 = None, symmetric = False):
    """High-level wrapper to optimize time lags maximizing the hit ratio.

    Parameters
    ----------
    h : float
        Forecast horizon (h > 0).
    n : int
        Number of historical lag durations to include in the predictor (n >= 1).
    H : float
        Hurst exponent, H in (0, 1).
    sigma : float, optional
        Volatility parameter of the fBm (default is 1).
    x0 : array-like, optional
        Initial guess for the optimization vector.
    symmetric : bool, optional
        If True, solves the problem in the reduced m-dimensional space by 
        enforcing delta_i* * delta_{n+1-i}* = h^2 (default is False).
        If False, solves over the full n-dimensional space.

    Returns
    -------
    dict or float
        Optimization result dictionary containing optimal 'durations', 'success', 
        and 'hitratio'.

    Notes
    -----
    Dispatches to `optimize_duration_reduced` when `symmetric=True` and 
    `optimize_duration_full` when `symmetric=False`.
    """
    if symmetric:
        return optimize_duration_reduced(h, n, H, sigma = sigma, x0 = x0)
    else:
        return optimize_duration_full(h, n, H, sigma = sigma, x0 = x0)
    
# ------------------------------------------------------------------------------
# 4. A ternary trading strategy based on the forecast.
# ------------------------------------------------------------------------------

def p_plus(theta, h, vd, H, sigma = 1):
    """Probability of a correct directional prediction under return thresholding.

    Parameters
    ----------
    theta : float
        Filter threshold value (theta >= 0). Positions are only taken when
        the magnitude of the forecast exceeds theta.
    h : float
        Forecast horizon (h > 0).
    vd : array-like
        Ordered vector of observation time lags including d_0 = 0:
            vd = [d_0, d_1, ..., d_n],
        with 0 = d_0 < d_1 < ... < d_n.
    H : float
        Hurst exponent, H in (0, 1).
    sigma : float, optional
        Scale / volatility parameter of the fBm (default is 1).

    Returns
    -------
    float
        Joint probability p_+(theta) = P(hat{R} > theta, R > 0) + P(hat{R} < - theta, R < 0)
        that the predictor takes a position and correctly predicts the direction 
        of the future return.

    Notes
    -----
    Implements the joint probability formula for p_+ in Theorem 2 of Garcin (2022).
    Here, a = sqrt(forecast_var) and b = sqrt(residual_var).
    """
    a = math.sqrt(forecast_var(h, vd, H, sigma))
    b = math.sqrt(residual_var(h, vd, H, sigma))
    mcov = [[1 + (a / b) ** 2, a / b], [a / b, 1]]
    foo = theta / a
    return (
        1/2 - stats.norm.cdf(foo) + 
        stats.multivariate_normal.cdf([0, foo], cov = mcov) +
        stats.multivariate_normal.cdf([0, - foo], cov = mcov)
        )

def p_minus(theta, h, vd, H, sigma = 1):
    """Probability of an incorrect directional prediction under return thresholding.

    Parameters
    ----------
    theta : float
        Filter threshold value (theta >= 0). Positions are only taken when
        the magnitude of the forecast exceeds theta.
    h : float
        Forecast horizon (h > 0).
    vd : array-like
        Ordered vector of observation time lags including d_0 = 0:
            vd = [d_0, d_1, ..., d_n],
        with 0 = d_0 < d_1 < ... < d_n.
    H : float
        Hurst exponent, H in (0, 1).
    sigma : float, optional
        Scale / volatility parameter of the fBm (default is 1).

    Returns
    -------
    float
        Joint probability p_-(theta) = P(hat{R} > theta, R < 0) + P(hat{R} < -theta, R > 0)
        that the predictor takes a position but incorrectly predicts the direction 
        of the future return.

    Notes
    -----
    Implements the joint probability formula for p_- in Theorem 2 of Garcin (2022).
    Together with p_+ and p_0, the probabilities satisfy p_+ + p_- + p_0 = 1.
    """
    a = math.sqrt(forecast_var(h, vd, H, sigma))
    b = math.sqrt(residual_var(h, vd, H, sigma))
    mcov = [[1 + (a / b) ** 2, a / b], [a / b, 1]]
    foo = theta / a
    return (
        3/2 - stats.norm.cdf(foo) - 
        stats.multivariate_normal.cdf([0, foo], cov = mcov) -
        stats.multivariate_normal.cdf([0, - foo], cov = mcov)
        )

def p_zero(theta, h, vd, H, sigma = 1):
    """Probability of no trading action under return thresholding.

    Parameters
    ----------
    theta : float
        Filter threshold value (theta >= 0).
    h : float
        Forecast horizon (h > 0).
    vd : array-like
        Ordered vector of observation time lags including d_0 = 0:
            vd = [d_0, d_1, ..., d_n],
        with 0 = d_0 < d_1 < ... < d_n.
    H : float
        Hurst exponent, H in (0, 1).
    sigma : float, optional
        Scale / volatility parameter of the fBm (default is 1).

    Returns
    -------
    float
        Probability p_0(theta) = P(|hat{R}| <= theta) = 2 * Phi(theta / a) - 1
        that the forecasted return magnitude does not exceed the threshold theta,
        resulting in no position being taken.

    Notes
    -----
    Implements Theorem 2 in Section 4.1 of Garcin (2022).
    When theta = 0, p_0(0) = 0 and all forecasts trigger active trading.
    """
    a = math.sqrt(forecast_var(h, vd, H, sigma))
    return - 1 + 2 * stats.norm.cdf(theta / a)

# ------------------------------------------------------------------------------
# 5. Risk-adjusted strategy performance and optimal threshold.
# ------------------------------------------------------------------------------


def mean_strat(theta, h, vd, H, sigma = 1):
    """Expected return of the thresholded trading strategy E[R^{theta}].

    Parameters
    ----------
    theta : float
        Filter threshold parameter (theta >= 0). Trades are executed only 
        when |E[R_{t, t+h} | S]| > theta.
    h : float
        Forecast horizon (h > 0).
    vd : array-like
        Ordered vector of observation time lags including d_0 = 0:
            vd = [d_0, d_1, ..., d_n].
    H : float
        Hurst exponent, H in (0, 1).
    sigma : float, optional
        Scale / volatility parameter of the fBm (default is 1).

    Returns
    -------
    float
        Expected strategy return E[R^{theta}] = 2 * a * phi(theta / a), 
        where a = sqrt(forecast_var) and phi is the standard normal PDF.

    Notes
    -----
    Implements Theorem 3 of Garcin (2022). As theta increases, 
    only high-confidence predictions are traded, which alters the expected 
    return per trade and trade frequency.
    """
    a = math.sqrt(forecast_var(h, vd, H, sigma))
    return 2 * a * stats.norm.pdf(theta / a) 

def risk_strat(theta, h, vd, H, sigma = 1):
    """Lower absolute semi-deviation (LASD) risk measure of the thresholded trading strategy.

    Parameters
    ----------
    theta : float
        Filter threshold parameter (theta >= 0).
    h : float
        Forecast horizon (h > 0).
    vd : array-like
        Ordered vector of observation time lags including d_0 = 0:
            vd = [d_0, d_1, ..., d_n].
    H : float
        Hurst exponent, H in (0, 1).
    sigma : float, optional
        Scale / volatility parameter of the fBm (default is 1).

    Returns
    -------
    float
        Lower absolute semi-deviation downside risk LASD(theta).

    Notes
    -----
    Implements Theorem 3 of Garcin (2022). The lower absolute 
    semi-deviation measures downside return risk:
        LASD(theta) = E[ max(0, -R^{theta}) ]
    where a = sqrt(forecast_var) and b = sqrt(residual_var).
    """
    a = math.sqrt(forecast_var(h, vd, H, sigma))
    b = math.sqrt(residual_var(h, vd, H, sigma))
    return (
        - 2 * a * stats.norm.cdf(- theta / b) * stats.norm.pdf(theta / a) +
        math.sqrt(2 / math.pi) * sigma * h ** H * 
        stats.norm.cdf(- theta * math.sqrt(1 / a ** 2 + 1 / b ** 2))
    )

def adj_perf_strat(theta, lam, h, vd, H, sigma = 1):
    """Risk-adjusted performance of the thresholded trading strategy.

    Parameters
    ----------
    theta : float
        Filter threshold parameter (theta >= 0).
    lam : float
        Risk-aversion penalty parameter lambda >= 0.
    h : float
        Forecast horizon (h > 0).
    vd : array-like
        Ordered vector of observation time lags including d_0 = 0:
            vd = [d_0, d_1, ..., d_n].
    H : float
        Hurst exponent, H in (0, 1).
    sigma : float, optional
        Scale / volatility parameter of the fBm (default is 1).

    Returns
    -------
    float
        Risk-adjusted performance score:
            P_lambda(theta) = E[R^{theta}] - lambda * LASD(theta).

    Notes
    -----
    Implements the risk-adjusted criterion from Section 4.2 of Garcin (2022), 
    penalizing strategy downside risk by factor lambda.
    """
    a = math.sqrt(forecast_var(h, vd, H, sigma))
    b = math.sqrt(residual_var(h, vd, H, sigma))
    ret = 2 * a * stats.norm.pdf(theta / a) 
    risk = (
        - 2 * a * stats.norm.cdf(- theta / b) * stats.norm.pdf(theta / a) +
        math.sqrt(2 / math.pi) * sigma * h ** H * 
        stats.norm.cdf(- theta * math.sqrt(1 / a ** 2 + 1 / b ** 2))
    )
    return ret - lam * risk

def optimize_threshold(lam, h, vd, H, sigma = 1, ub = 0.1):
    """Find the optimal forecast threshold theta* maximizing risk-adjusted performance.

    Parameters
    ----------
    lam : float
        Risk aversion penalty coefficient lambda >= 0.
    h : float
        Forecast horizon (h > 0).
    vd : array-like
        Ordered vector of observation time lags including d_0 = 0:
            vd = [d_0, d_1, ..., d_n].
    H : float
        Hurst exponent, H in (0, 1).
    sigma : float, optional
        Scale / volatility parameter of the fBm (default is 1).
    ub : float, optional
        Upper bound for threshold search range [0, ub] (default is 0.1).

    Returns
    -------
    dict
        Dictionary containing:
        - 'theta' : float
            Optimal forecast threshold theta*.
        - 'perf' : float
            Maximized risk-adjusted performance value R_lambda(theta*).

    Notes
    -----
    Uses `scipy.optimize.minimize_scalar` with bounded optimization over [0, ub] 
    to maximize the risk-adjusted criterion P_lambda(theta).
    """
    res = minimize_scalar(
            lambda theta: - adj_perf_strat(
                theta, lam, h, vd, H, sigma),
            bounds = (0, ub), method = "bounded")
    return {'theta': res.x, 'perf': res.fun}