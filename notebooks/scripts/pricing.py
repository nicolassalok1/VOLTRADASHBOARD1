import math
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

DEFAULT_R = 0.02
DEFAULT_Q = 0.0
DEFAULT_SIGMA = 0.2
DEFAULT_T = 1.0

_CACHE_SPY_CLOSE = Path("notebooks/GPT/_cache_spy_close.csv")


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price_call(S: float, K: float, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, T: float = DEFAULT_T) -> float:
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_price_put(S: float, K: float, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, T: float = DEFAULT_T) -> float:
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


def fetch_spy_history(period: str = "1y", interval: str = "1d", cache_path: Path = _CACHE_SPY_CLOSE) -> pd.Series:
    """Fetch SPY close prices with a simple CSV cache under notebooks/GPT/."""
    try:
        if cache_path.exists():
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not cached.empty and "Close" in cached.columns:
                return cached["Close"]
    except Exception:
        pass
    import yfinance as yf

    data = yf.download("SPY", period=period, interval=interval, progress=False)
    if data.empty or "Close" not in data:
        raise RuntimeError("Impossible de récupérer les prix SPY")
    close = data["Close"]
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        close.to_csv(cache_path, index_label="date")
    except Exception:
        pass
    return close


def last_spy_close(period: str = "1y", interval: str = "1d") -> float:
    close = fetch_spy_history(period=period, interval=interval)
    return float(close.iloc[-1])


def payoff_call(spot, strike: float):
    s = np.asarray(spot, dtype=float)
    return np.maximum(s - strike, 0.0)


def payoff_put(spot, strike: float):
    s = np.asarray(spot, dtype=float)
    return np.maximum(strike - s, 0.0)


def payoff_straddle(spot, strike: float):
    return payoff_call(spot, strike) + payoff_put(spot, strike)


def payoff_strangle(spot, k_put: float, k_call: float):
    return payoff_put(spot, k_put) + payoff_call(spot, k_call)


def payoff_call_spread(spot, k_long: float, k_short: float):
    return payoff_call(spot, k_long) - payoff_call(spot, k_short)


def payoff_put_spread(spot, k_long: float, k_short: float):
    return payoff_put(spot, k_long) - payoff_put(spot, k_short)


def payoff_butterfly(spot, k1: float, k2: float, k3: float):
    return payoff_call(spot, k1) - 2.0 * payoff_call(spot, k2) + payoff_call(spot, k3)


def payoff_condor(spot, k1: float, k2: float, k3: float, k4: float):
    return payoff_call(spot, k1) - payoff_call(spot, k2) - payoff_call(spot, k3) + payoff_call(spot, k4)


def payoff_iron_butterfly(spot, k_put_long: float, k_center: float, k_call_long: float):
    return payoff_put(spot, k_put_long) - payoff_put(spot, k_center) - payoff_call(spot, k_center) + payoff_call(spot, k_call_long)


def payoff_iron_condor(spot, k_put_long: float, k_put_short: float, k_call_short: float, k_call_long: float):
    return payoff_put(spot, k_put_long) - payoff_put(spot, k_put_short) - payoff_call(spot, k_call_short) + payoff_call(spot, k_call_long)


def price_straddle_bs(S: float, strike: float, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, T: float = DEFAULT_T) -> float:
    return bs_price_put(S, strike, r=r, q=q, sigma=sigma, T=T) + bs_price_call(S, strike, r=r, q=q, sigma=sigma, T=T)


def price_strangle_bs(S: float, k_put: float, k_call: float, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, T: float = DEFAULT_T) -> float:
    return bs_price_put(S, k_put, r=r, q=q, sigma=sigma, T=T) + bs_price_call(S, k_call, r=r, q=q, sigma=sigma, T=T)


def pricing_strangle_bs(S: float, k_put: float, k_call: float, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, T: float = DEFAULT_T) -> float:
    """Alias explicite pour le pricing d'un strangle via Black-Scholes (somme put+call)."""
    return price_strangle_bs(S, k_put, k_call, r=r, q=q, sigma=sigma, T=T)


def price_call_spread_bs(S: float, k_long: float, k_short: float, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, T: float = DEFAULT_T) -> float:
    return bs_price_call(S, k_long, r=r, q=q, sigma=sigma, T=T) - bs_price_call(S, k_short, r=r, q=q, sigma=sigma, T=T)


def price_put_spread_bs(S: float, k_long: float, k_short: float, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, T: float = DEFAULT_T) -> float:
    return bs_price_put(S, k_long, r=r, q=q, sigma=sigma, T=T) - bs_price_put(S, k_short, r=r, q=q, sigma=sigma, T=T)


def price_butterfly_bs(S: float, k1: float, k2: float, k3: float, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, T: float = DEFAULT_T) -> float:
    return bs_price_call(S, k1, r=r, q=q, sigma=sigma, T=T) - 2.0 * bs_price_call(S, k2, r=r, q=q, sigma=sigma, T=T) + bs_price_call(S, k3, r=r, q=q, sigma=sigma, T=T)


def price_condor_bs(S: float, k1: float, k2: float, k3: float, k4: float, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, T: float = DEFAULT_T) -> float:
    return (
        bs_price_call(S, k1, r=r, q=q, sigma=sigma, T=T)
        - bs_price_call(S, k2, r=r, q=q, sigma=sigma, T=T)
        - bs_price_call(S, k3, r=r, q=q, sigma=sigma, T=T)
        + bs_price_call(S, k4, r=r, q=q, sigma=sigma, T=T)
    )


def price_iron_butterfly_bs(S: float, k_put_long: float, k_center: float, k_call_long: float, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, T: float = DEFAULT_T) -> float:
    return (
        bs_price_put(S, k_put_long, r=r, q=q, sigma=sigma, T=T)
        - bs_price_put(S, k_center, r=r, q=q, sigma=sigma, T=T)
        - bs_price_call(S, k_center, r=r, q=q, sigma=sigma, T=T)
        + bs_price_call(S, k_call_long, r=r, q=q, sigma=sigma, T=T)
    )


def price_iron_condor_bs(S: float, k_put_long: float, k_put_short: float, k_call_short: float, k_call_long: float, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, T: float = DEFAULT_T) -> float:
    return (
        bs_price_put(S, k_put_long, r=r, q=q, sigma=sigma, T=T)
        - bs_price_put(S, k_put_short, r=r, q=q, sigma=sigma, T=T)
        - bs_price_call(S, k_call_short, r=r, q=q, sigma=sigma, T=T)
        + bs_price_call(S, k_call_long, r=r, q=q, sigma=sigma, T=T)
    )


def _build_view(payoff_fn, premium: float, s0: float, args: Iterable, breakevens: Tuple[float, ...], span: float = 0.5, n: int = 300):
    s_grid = np.linspace(s0 * (1.0 - span), s0 * (1.0 + span), n)
    payoff_grid = payoff_fn(s_grid, *args)
    pnl_grid = payoff_grid - premium
    return {
        "s_grid": s_grid,
        "payoff": payoff_grid,
        "pnl": pnl_grid,
        "premium": premium,
        "breakevens": tuple(breakevens),
    }


def view_vanilla_call(s0: float, strike: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = bs_price_call(s0, strike, **kwargs)
    be = strike + premium
    return _build_view(payoff_call, premium, s0, (strike,), (be,), span, n)


def view_vanilla_put(s0: float, strike: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = bs_price_put(s0, strike, **kwargs)
    be = strike - premium
    return _build_view(payoff_put, premium, s0, (strike,), (be,), span, n)


def view_straddle(s0: float, strike: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = price_straddle_bs(s0, strike, **kwargs)
    be_low, be_high = strike - premium, strike + premium
    return _build_view(payoff_straddle, premium, s0, (strike,), (be_low, be_high), span, n)


def view_strangle(s0: float, k_put: float, k_call: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = price_strangle_bs(s0, k_put, k_call, **kwargs)
    be_low, be_high = k_put - premium, k_call + premium
    return _build_view(payoff_strangle, premium, s0, (k_put, k_call), (be_low, be_high), span, n)


def view_call_spread(s0: float, k_long: float, k_short: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = price_call_spread_bs(s0, k_long, k_short, **kwargs)
    be = k_long + premium
    return _build_view(payoff_call_spread, premium, s0, (k_long, k_short), (be,), span, n)


def view_put_spread(s0: float, k_long: float, k_short: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = price_put_spread_bs(s0, k_long, k_short, **kwargs)
    be = k_long - premium
    return _build_view(payoff_put_spread, premium, s0, (k_long, k_short), (be,), span, n)


def view_butterfly(s0: float, k1: float, k2: float, k3: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = price_butterfly_bs(s0, k1, k2, k3, **kwargs)
    be_low, be_high = k1 + premium, k3 - premium
    return _build_view(payoff_butterfly, premium, s0, (k1, k2, k3), (be_low, be_high), span, n)


def view_condor(s0: float, k1: float, k2: float, k3: float, k4: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = price_condor_bs(s0, k1, k2, k3, k4, **kwargs)
    be_low, be_high = k1 + premium, k4 - premium
    return _build_view(payoff_condor, premium, s0, (k1, k2, k3, k4), (be_low, be_high), span, n)


def view_iron_butterfly(s0: float, k_put_long: float, k_center: float, k_call_long: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = price_iron_butterfly_bs(s0, k_put_long, k_center, k_call_long, **kwargs)
    credit = -premium
    be_low, be_high = k_center - credit, k_center + credit
    return _build_view(payoff_iron_butterfly, premium, s0, (k_put_long, k_center, k_call_long), (be_low, be_high), span, n)


def view_iron_condor(s0: float, k_put_long: float, k_put_short: float, k_call_short: float, k_call_long: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = price_iron_condor_bs(s0, k_put_long, k_put_short, k_call_short, k_call_long, **kwargs)
    credit = -premium
    be_low, be_high = k_put_short - credit, k_call_short + credit
    return _build_view(payoff_iron_condor, premium, s0, (k_put_long, k_put_short, k_call_short, k_call_long), (be_low, be_high), span, n)
