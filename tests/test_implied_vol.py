import numpy as np
import pytest
from src.black_scholes import bs_price
from src.implied_vol import implied_vol


@pytest.mark.parametrize("sigma", [0.05, 0.15, 0.20, 0.40, 0.80, 1.50])
@pytest.mark.parametrize("K", [80, 100, 120])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_aller_retour(sigma, K, option_type):
    S, T, r = 100, 1.0, 0.05
    prix = bs_price(S, K, T, r, sigma, option_type)
    iv = implied_vol(prix, S, K, T, r, option_type)
    assert iv == pytest.approx(sigma, rel=1e-6)


def test_exemple_hull():
    iv = implied_vol(4.759422, S=42, K=40, T=0.5, r=0.10, option_type="call")
    assert iv == pytest.approx(0.20, abs=1e-5)


def test_prix_sous_borne_inferieure():
    S, K, T, r = 100, 80, 1.0, 0.05
    borne = S - K * np.exp(-r * T)
    assert np.isnan(implied_vol(borne - 1, S, K, T, r, "call"))


def test_prix_au_dessus_de_S():
    assert np.isnan(implied_vol(105, S=100, K=100, T=1.0, r=0.05))


def test_option_tres_hors_monnaie():
    S, K, T, r, sigma = 100, 160, 0.25, 0.05, 0.25
    prix = bs_price(S, K, T, r, sigma, "call")
    iv = implied_vol(prix, S, K, T, r, "call")
    assert iv == pytest.approx(sigma, rel=1e-4)