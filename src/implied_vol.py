import numpy as np
from src.black_scholes import bs_price, bs_vega

SIGMA_MIN = 1e-6
SIGMA_MAX = 5.0


def _bornes_arbitrage(S, K, T, r, option_type):
    strike_actualise = K * np.exp(-r * T)
    if option_type == "call":
        return max(S - strike_actualise, 0.0), S
    return max(strike_actualise - S, 0.0), strike_actualise


def _bissection(prix, S, K, T, r, option_type, tol, max_iter):
    bas, haut = SIGMA_MIN, SIGMA_MAX
    for _ in range(max_iter):
        milieu = 0.5 * (bas + haut)
        if haut - bas < tol:
            return milieu
        ecart = bs_price(S, K, T, r, milieu, option_type) - prix
        if ecart > 0:
            haut = milieu     
        else:
            bas = milieu    
    return 0.5 * (bas + haut)


def implied_vol(prix, S, K, T, r, option_type="call",
                sigma0=0.20, tol=1e-8, max_iter=100):
    borne_inf, borne_sup = _bornes_arbitrage(S, K, T, r, option_type)
    if not (borne_inf < prix < borne_sup):
        return np.nan

    sigma = sigma0
    for _ in range(max_iter):
        ecart = bs_price(S, K, T, r, sigma, option_type) - prix
        vega =bs_vega(S, K, T, r, sigma)
        if vega < 1e-8:
            break                    

        pas = ecart / vega            
        sigma_suivant = sigma - pas
        if not (SIGMA_MIN < sigma_suivant < SIGMA_MAX):
            break                        
        if abs(pas) < tol:
            return sigma_suivant         
        sigma = sigma_suivant

    return _bissection(prix, S, K, T, r, option_type, tol, max_iter)