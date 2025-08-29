# Reconstructed by merge of 1 variants for trj_opt/recommend.py
# Strategy: union of imports, union of classes/functions with longest/most-params preference.

import json, math, random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from .lab_bridge import LabBridge, _load_simple_config
from .objectives import efficiency_percent, series_resistance_area

@dataclass
class RecommendConfig:
    target_eta_percent: float = 26.0
    Jsc_mA_cm2: float = 19.5
    Voc_bulk_V: float = 1.9
    J0_bulk_A_cm2: float = 1e-16
    diode_n: float = 1.0
    T_K: float = 300.0
    Rsh_area_Ohm_cm2: float = 1000000000000.0
    Pin_W_m2: float = 1000.0
    rho_c_max_Ohm_cm2: float = 0.01
    dVoc_max_V: float = 0.02
    geom_factor_sheet: float = 0.5
    Rs_other_Ohm_cm2: float = 0.0
    jitter_rel: float = 0.05
    jitter_abs: Dict[str, float] = None
    mc_samples: int = 32
    risk_beta: float = 0.5
    n_init: int = 64
    n_iter: int = 80
    candidates_per_iter: int = 64
    top_k_exploit: int = 8

def load_config(path: Optional[str]) -> RecommendConfig:
    if path is None:
        return RecommendConfig()
    d = _load_simple_config(path)
    kw = {**RecommendConfig().__dict__, **d}
    return RecommendConfig(**kw)

def propose_candidates(X, S, bounds, n=64, top_k=8):
    cands = []
    if X:
        idx = np.argsort(np.array(S))[::-1][:max(1, top_k)]
        bests = [X[i] for i in idx]
        for b in bests:
            for _ in range(n // 2 // max(1, len(bests))):
                y = {}
                for k, (lo, hi) in bounds.items():
                    span = hi - lo
                    y[k] = float(np.clip(np.random.normal(b[k], 0.05 * span), lo, hi))
                cands.append(y)
    while len(cands) < n:
        cands.append(sample_within(bounds))
    return cands[:n]

def recommend(bridge_path: str, cfg_path: Optional[str], n_return: int=3, seed: int=0) -> pd.DataFrame:
    np.random.seed(seed)
    random.seed(seed)
    bridge = LabBridge.from_file(bridge_path)
    cfg = load_config(cfg_path)
    B = bridge.bounds()
    X = []
    S = []
    R = []
    for _ in range(max(4, cfg.n_init)):
        x0 = sample_within(B)
        sc, summ = robust_score(x0, bridge, cfg, B)
        X.append(x0)
        S.append(sc)
        R.append(summ)
    for _ in range(cfg.n_iter):
        C = propose_candidates(X, S, B, cfg.candidates_per_iter, cfg.top_k_exploit)
        best_c = None
        best_sc = -1000000000.0
        best_sum = None
        for c in C:
            sc, summ = robust_score(c, bridge, cfg, B)
            if sc > best_sc:
                best_sc = sc
                best_c = c
                best_sum = summ
        X.append(best_c)
        S.append(best_sc)
        R.append(best_sum)
    order = np.argsort(np.array(S))[::-1]
    rows = []
    for i in order[:n_return]:
        x = X[i]
        sc = S[i]
        summ = R[i]
        met = score_x(x, bridge, cfg)
        rows.append({**x, **met, **summ, 'meets_26pct': met['eta_percent'] >= cfg.target_eta_percent, 'rho_c_mOhm_cm2': 1000.0 * met['rho_c_Ohm_cm2'], 'DeltaVoc_mV': 1000.0 * met['DeltaVoc_V']})
    return pd.DataFrame(rows)

def robust_score(x, bridge: LabBridge, cfg: RecommendConfig, bounds):
    scores = []
    etas = []
    dv = []
    rc = []
    for _ in range(cfg.mc_samples):
        xj = sample_with_jitter(x, bounds, cfg)
        met = score_x(xj, bridge, cfg)
        scores.append(met['score'])
        etas.append(met['eta_percent'])
        dv.append(met['DeltaVoc_V'])
        rc.append(met['rho_c_Ohm_cm2'])
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    robust = mean - cfg.risk_beta * std
    med = lambda a: float(np.median(np.array(a)))
    return (robust, {'eta_median': med(etas), 'DeltaVoc_median_V': med(dv), 'rho_c_median_Ohm_cm2': med(rc), 'robust_score': robust, 'score_mean': mean, 'score_std': std})

def sample_with_jitter(x, bounds, cfg: RecommendConfig):
    y = {}
    for k, (lo, hi) in bounds.items():
        v = x[k]
        if cfg.jitter_abs and k in (cfg.jitter_abs or {}):
            s = float(cfg.jitter_abs[k])
            vj = np.random.normal(v, s)
        else:
            s = abs(cfg.jitter_rel * v)
            vj = np.random.normal(v, s)
        y[k] = float(np.clip(vj, lo, hi))
    return y

def sample_within(bounds: Dict[str, Tuple[float, float]]):
    return {k: float(np.random.uniform(lo, hi)) for k, (lo, hi) in bounds.items()}

def score_x(x, bridge: LabBridge, cfg: RecommendConfig):
    m = bridge.lab_to_model(x)
    Rs_area = series_resistance_area(m['rho_c_Ohm_cm2'], m.get('sheet_resistance_ohm_sq', None), cfg.geom_factor_sheet, cfg.Rs_other_Ohm_cm2)
    out = efficiency_percent(cfg.Jsc_mA_cm2, cfg.Voc_bulk_V, cfg.J0_bulk_A_cm2, m['J0_interface_A_cm2'], Rs_area, n=cfg.diode_n, T=cfg.T_K, Rsh_area_Ohm_cm2=cfg.Rsh_area_Ohm_cm2, Pin_W_m2=cfg.Pin_W_m2)
    penalty = 0.0
    if m['rho_c_Ohm_cm2'] > cfg.rho_c_max_Ohm_cm2:
        penalty += (m['rho_c_Ohm_cm2'] - cfg.rho_c_max_Ohm_cm2) / max(1e-12, cfg.rho_c_max_Ohm_cm2)
    if out['DeltaVoc_V'] > cfg.dVoc_max_V:
        penalty += (out['DeltaVoc_V'] - cfg.dVoc_max_V) / max(1e-12, cfg.dVoc_max_V)
    score = out['eta_percent'] - 100.0 * penalty
    return {**out, 'rho_c_Ohm_cm2': m['rho_c_Ohm_cm2'], 'J0_interface_A_cm2': m['J0_interface_A_cm2'], 'sheet_resistance_ohm_sq': m.get('sheet_resistance_ohm_sq', float('nan')), 'Rs_area_Ohm_cm2': Rs_area, 'score': float(score)}
import json, math, random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from .lab_bridge import LabBridge, _load_simple_config
from .objectives import efficiency_percent, series_resistance_area
