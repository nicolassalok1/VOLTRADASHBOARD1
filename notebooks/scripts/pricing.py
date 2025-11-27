import math

DEFAULT_R = 0.02
DEFAULT_Q = 0.0
DEFAULT_SIGMA = 0.2
DEFAULT_T = 1.0


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


def payoff_call(spot: float, strike: float) -> float:
    return max(spot - strike, 0.0)


def payoff_put(spot: float, strike: float) -> float:
    return max(strike - spot, 0.0)


def payoff_strangle(spot: float, k_put: float, k_call: float) -> float:
    return payoff_put(spot, k_put) + payoff_call(spot, k_call)


def price_strangle_bs(S: float, k_put: float, k_call: float, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, T: float = DEFAULT_T) -> float:
    return bs_price_put(S, k_put, r=r, q=q, sigma=sigma, T=T) + bs_price_call(S, k_call, r=r, q=q, sigma=sigma, T=T)


def pricing_strangle_bs(S: float, k_put: float, k_call: float, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, T: float = DEFAULT_T) -> float:
    """Alias explicite pour le pricing d'un strangle via Black-Scholes (somme put+call)."""
    return price_strangle_bs(S, k_put, k_call, r=r, q=q, sigma=sigma, T=T)


def payoff_strangle(spot: float, k_put: float, k_call: float) -> float:
    return payoff_put(spot, k_put) + payoff_call(spot, k_call)
