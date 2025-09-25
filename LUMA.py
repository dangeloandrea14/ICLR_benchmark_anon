from sklearn.metrics.pairwise import laplacian_kernel, chi2_kernel 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
import statistics

def efficiency_score(gold_efficiency, test_efficiency, eff_w=[0.9,0.1], power=3):

    if eff_w is None:
        eff_w = np.ones_like(test_efficiency) / len(test_efficiency)
    else:
        eff_w = np.array(eff_w, dtype=float)

    ratios = np.log1p(test_efficiency) / np.log1p(gold_efficiency)
    scores = np.exp(-(ratios ** power))

    return float(np.dot(eff_w, scores))

def LUMA(gold, m_prime, eff_w = [0.9, 0.1]):

    gold_utility, gold_efficacy, gold_efficiency = gold
    test_utility, test_efficacy, test_efficiency = m_prime

    utility = laplacian_kernel([gold_utility],[test_utility],gamma=4)[0, 0]
    efficacy = laplacian_kernel([gold_efficacy], [test_efficacy],gamma=4)[0, 0]

    efficiency = efficiency_score(gold_efficiency, test_efficiency, eff_w)

    return statistics.harmonic_mean([utility, efficacy, efficiency])

