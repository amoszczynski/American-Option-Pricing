import numpy as np
import pandas as pd
from scipy import interpolate
import datetime as dt

# risk free rate dataframe
rfr = pd.read_csv("daily-treasury-rates.csv", index_col=0)
rfr.index = pd.to_datetime(rfr.index)
rfr.columns = [1 / 12, 1 / 6, 1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 5, 7, 10, 20, 30]


### RFR CURVES
def _flat_r(N):
    """Gets a flat risk free rate for all times to maturity"""
    return np.array([0.054] * N)


def _curve_r(N, T, calc_day):
    """Interpolates risk free rate using treasury rates"""
    rf = rfr.loc[calc_day]
    y = np.array(rf.values) / 100
    x = rf.index.tolist()

    # cubic spline for nonlinear interpolaton
    cs = interpolate.CubicSpline(x, y)
    return cs(np.linspace(0, T, N))


### DIVIDEND FUNCTION
def time_to_div(day):
    """fraction of years until next dividend"""

    # get offset
    divs = ["09-15", "06-16", "03-17", "12-16"]
    div_dates = sorted(
        [dt.datetime.strptime(f"{day.year}-{date}", "%Y-%m-%d") for date in divs]
    )

    # get next dividend
    for date in div_dates:
        if day <= date:
            return (date - day).days / 365

    return (div_dates[0].replace(year=day.year + 1) - day).days / 365


### PRICER
def american_option(S0, K, calc_day, expiry, sig, payoff, rf, div_info):
    """
    Parameters:
        S0 - spot price
        K - strike price
        calc_day - day of calculation
        expiry - expiration date of option
        sig - annualized vol
        payoff - 'call' or 'put'
        rf - 'flat' or 'curve'
        div_info - [yield, freq, offset]
    """
    # initialization
    N = 1_000
    T = (expiry - calc_day).days / 365
    if rf == "flat":
        r = _flat_r(N)
    else:
        r = _curve_r(N, T, calc_day)

    # dividend init
    div_yield = div_info[0]  # quarterly payout
    div_freq = div_info[1]  # quarterly
    div_offset = div_info[2]  # time in years until first dividend

    # number of dividends that will occur in option lifetime
    num_div = int((T - div_offset) / div_freq) + 1

    # times in years at which dividends will occur
    div_times = [div_offset + n * div_freq for n in range(0, num_div)]

    # initialization
    dT = float(T) / N  # Delta t
    u = np.exp(sig * np.sqrt(dT))  # up factor
    d = 1.0 / u  # down factor

    a = np.exp(r * dT)  # risk free compound return
    p = (a - d) / (u - d)  # risk neutral up probability
    q = 1.0 - p  # risk neutral down probability

    # initialize price vector
    V = np.zeros(N + 1)
    S_T = np.array([(S0 * u**j * d ** (N - j)) for j in range(N + 1)])

    # adjusting price for number of dividends that has occured
    S_T = S_T * (1 - div_yield) ** num_div

    # payoff function
    if payoff == "call":
        V[:] = np.maximum(S_T - K, 0.0)
    elif payoff == "put":
        V[:] = np.maximum(K - S_T, 0.0)

    # backwards iteration
    for i in range(N - 1, -1, -1):
        # the price vector is overwritten at each step
        V[:-1] = np.exp(-r[i] * dT) * (p[i] * V[1:] + q[i] * V[:-1])

        # obtain the price at the previous time step
        S_T = S_T * u

        # if we pass a dividend time, we can divide by (1-div)
        if div_times:
            if i * dT < div_times[-1]:
                S_T = S_T / (1 - div_yield)
                div_times.pop(-1)

        # american early exercise
        if payoff == "call":
            V = np.maximum(V, S_T - K)
        elif payoff == "put":
            V = np.maximum(V, K - S_T)

    return V[0]


### IV SURFACE
def derive_IV(S0, K, calc_day, expiry, payoff, rf, div_info, mid, tol=0.001):
    """
    Calculates Implied Volatility to a certain tolerance using bisection
    Parameters:
        S0 - spot price
        K - strike price
        calc_day - day of calculation
        expiry - expiration date of option
        payoff - 'call' or 'put'
        rf - 'flat' or 'curve'
        div_info - [yield, freq, offset]
        mid - midprice of specific option
        tol - tolerance for biseciton method

    """

    MAX_ITER = 1000

    # f(vol) = p(vol) - midprice
    def f(model_price):
        return model_price - mid

    # bisection method
    interval = [0, 10]  # 10 will essentially guarantee convergence
    mvol = np.mean(interval)
    fx = f(american_option(S0, K, calc_day, expiry, mvol, payoff, rf, div_info))

    i = 0
    while np.abs(fx) > tol and i < MAX_ITER:
        # adjust interval
        if fx > 0:
            interval[1] = mvol
        else:
            interval[0] = mvol

        # adjust parameters for next iteration
        mvol = np.mean(interval)
        fx = f(american_option(S0, K, calc_day, expiry, mvol, payoff, rf, div_info))
        i += 1

    return mvol
