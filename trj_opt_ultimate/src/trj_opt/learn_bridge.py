# Reconstructed by merge of 1 variants for trj_opt/learn_bridge.py
# Strategy: union of imports, union of classes/functions with longest/most-params preference.

import json, math, random
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from .lab_bridge import LabBridge, _load_simple_config
from .objectives import dvoc_from_J0int

@dataclass
class FitConfig:
    max_iters: int = 1000
    init_sigma: float = 0.25
    sigma_decay: float = 0.995
    seed: int = 0
    huber_delta: float = 0.35
    bootstrap: int = 20

def _huber(z: float, delta: float=0.35) -> float:
    a = abs(z)
    if a <= delta:
        return 0.5 * z * z
    return delta * (a - 0.5 * delta)

def _logratio_err(pred: float, meas: float) -> float:
    if meas is None or meas <= 0:
        return 0.0
    if pred <= 0:
        pred = 1e-30
    return math.log(pred / meas)

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename = {'rho_c_mOhm_cm2': 'meas_rho_c_mOhm_cm2', 'rho_c_Ohm_cm2': 'meas_rho_c_Ohm_cm2', 'DeltaVoc_mV': 'meas_DeltaVoc_mV', 'DeltaVoc_interface_mV': 'meas_DeltaVoc_mV', 'DeltaVoc_V': 'meas_DeltaVoc_V', 'J0_interface_A_cm2': 'meas_J0int_A_cm2', 'sheet_R_ohm_sq': 'meas_sheet_ohm_sq', 'sheet_resistance_ohm_sq': 'meas_sheet_ohm_sq', 't_contact_nm': 't_contact_nm', 'anneal_C': 'anneal_C', 'anneal_min': 'anneal_min', 'plasma_W': 'plasma_W', 'plasma_s': 'plasma_s', 'passivation_nm': 'passivation_nm', 'surface_treat_s': 'surface_treat_s', 'dope_dose_1e13cm2': 'dope_dose_1e13cm2'}
    for k, v in rename.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    if 'meas_rho_c_mOhm_cm2' in df.columns and 'meas_rho_c_Ohm_cm2' not in df.columns:
        df['meas_rho_c_Ohm_cm2'] = 0.001 * df['meas_rho_c_mOhm_cm2'].astype(float)
    if 'meas_DeltaVoc_mV' in df.columns and 'meas_DeltaVoc_V' not in df.columns:
        df['meas_DeltaVoc_V'] = 0.001 * df['meas_DeltaVoc_mV'].astype(float)
    return df

def _predict_from_params(x_row: Dict[str, float], params: Dict[str, Any]) -> Dict[str, float]:
    import math

    def g(name, default=0.0):
        return float(params.get(name, default))

    def gx(key):
        return float(x_row.get(key, params.get(key + '_init', 0.0)))
    t_nm = gx('t_contact_nm')
    T_C = gx('anneal_C')
    t_min = gx('anneal_min')
    plasma_W = gx('plasma_W')
    plasma_s = gx('plasma_s')
    passiv_nm = gx('passivation_nm')
    surf_s = gx('surface_treat_s')
    dope_dose = gx('dope_dose_1e13cm2')
    rho_min = g('rho_c_min_Ohm_cm2', 0.0001)
    rho_max = g('rho_c_max_Ohm_cm2', 0.1)
    k_t = g('rho_k_t', 0.12)
    k_T = g('rho_k_T', 0.02)
    T0 = g('rho_T0_C', 120.0)
    k_tau = g('rho_k_tau', 0.08)
    k_dose = g('rho_k_dose', 0.05)
    f_t = 1.0 - math.exp(-k_t * max(0.0, t_nm))
    f_T = 1.0 - math.exp(-k_T * max(0.0, T_C - T0))
    f_tau = 1.0 - math.exp(-k_tau * max(0.0, t_min))
    f_dose = 1.0 - math.exp(-k_dose * max(0.0, dope_dose))
    contact_gain = f_t * f_T * f_tau * (0.5 + 0.5 * f_dose)
    rho_c = rho_max - (rho_max - rho_min) * contact_gain
    rho_c = max(rho_min, min(rho_max, rho_c))
    J0_min = g('J0int_min_A_cm2', 1e-18)
    J0_max = g('J0int_max_A_cm2', 1e-12)
    k_pass = g('J0_k_pass', 0.25)
    k_surf = g('J0_k_surf', 0.015)
    P_W_star = g('plasma_W_star', 8.0)
    P_t_star = g('plasma_s_star', 15.0)
    P_sigma_W = max(0.001, g('plasma_sigma_W', 3.0))
    P_sigma_t = max(0.001, g('plasma_sigma_s', 6.0))
    penalty_plasma = math.exp(-0.5 * ((plasma_W - P_W_star) / P_sigma_W) ** 2 - 0.5 * ((plasma_s - P_t_star) / P_sigma_t) ** 2)
    pass_gain = 1.0 - math.exp(-k_pass * max(0.0, passiv_nm))
    surf_gain = 1.0 - math.exp(-k_surf * max(0.0, surf_s))
    reduc = 0.6 * pass_gain + 0.4 * surf_gain
    J0_int = J0_max - (J0_max - J0_min) * reduc * penalty_plasma
    J0_int = max(J0_min, min(J0_max, J0_int))
    rho_bulk_Ohm_cm = g('rho_bulk_Ohm_cm', 0.0002)
    t_cm = max(1e-09, t_nm * 1e-07)
    sheet_R_ohm_sq = rho_bulk_Ohm_cm / t_cm
    return {'rho_c_Ohm_cm2': rho_c, 'J0int_A_cm2': J0_int, 'sheet_ohm_sq': sheet_R_ohm_sq}

def _project(typ: str, lo: float, hi: float, val: float) -> float:
    if typ == 'pos':
        val = abs(val)
    return float(max(lo, min(hi, val)))

def _write_yaml_like(path: str, d: Dict[str, Any]) -> None:
    with open(path, 'w') as f:
        for k, v in d.items():
            f.write(f'{k}: {v}\n')

def fit_lab_bridge(lab_yaml: str, constraints_yaml: Optional[str], measurements_csv: str, out_yaml: str, out_report_csv: str, weights: Dict[str, float]=None, fit_cfg: FitConfig=FitConfig()) -> Dict[str, Any]:
    if weights is None:
        weights = {'rho': 1.0, 'j0': 1.0, 'sheet': 0.3}
    np.random.seed(fit_cfg.seed)
    random.seed(fit_cfg.seed)
    params0 = _load_simple_config(lab_yaml)
    constraints = _load_simple_config(constraints_yaml) if constraints_yaml else {}
    J0_bulk = float(constraints.get('J0_bulk_A_cm2', 1e-16))
    T_K = float(constraints.get('T_K', 300.0))
    df = pd.read_csv(measurements_csv)
    df = _normalize_columns(df)
    rows = df.to_dict(orient='records')
    names = [n for n, t, lo, hi, use in PARAM_SPECS if n in params0]
    specs_map = {n: (t, lo, hi) for n, t, lo, hi, use in PARAM_SPECS if n in params0}
    vec = np.array([_project(specs_map[n][0], specs_map[n][1], specs_map[n][2], float(params0[n])) for n in names])

    def objective(v, rows_local):
        params = {}
        for i, n in enumerate(names):
            t, lo, hi = specs_map[n]
            params[n] = _project(t, lo, hi, float(v[i]))
        loss = 0.0
        for r in rows_local:
            pred = _predict_from_params(r, params)
            if 'meas_rho_c_Ohm_cm2' in r and r['meas_rho_c_Ohm_cm2'] is not None:
                loss += weights['rho'] * _huber(_logratio_err(pred['rho_c_Ohm_cm2'], r['meas_rho_c_Ohm_cm2']), fit_cfg.huber_delta)
            J0_meas = r.get('meas_J0int_A_cm2', None)
            if J0_meas is None and 'meas_DeltaVoc_V' in r and (J0_bulk > 0):
                Vt = 0.025852 * (T_K / 300.0)
                J0_meas = float(J0_bulk * (math.exp(r['meas_DeltaVoc_V'] / Vt) - 1.0)) if r['meas_DeltaVoc_V'] is not None else None
            if J0_meas is not None and J0_meas > 0:
                loss += weights['j0'] * _huber(_logratio_err(pred['J0int_A_cm2'], J0_meas), fit_cfg.huber_delta)
            if 'meas_sheet_ohm_sq' in r and r['meas_sheet_ohm_sq'] is not None:
                loss += weights['sheet'] * _huber(_logratio_err(pred['sheet_ohm_sq'], r['meas_sheet_ohm_sq']), fit_cfg.huber_delta)
        return float(loss)
    best_vec = vec.copy()
    best_loss = objective(best_vec, rows)
    sigma = float(fit_cfg.init_sigma)
    for it in range(fit_cfg.max_iters):
        cand = best_vec.copy()
        k = np.random.randint(1, max(2, len(cand) // 3) + 1)
        idxs = np.random.choice(len(cand), size=k, replace=False)
        for i in idxs:
            t, lo, hi = specs_map[names[i]]
            if t == 'pos':
                step = math.exp(np.random.normal(0.0, sigma))
                cand[i] = _project(t, lo, hi, float(cand[i] * step))
            else:
                span = hi - lo
                cand[i] = _project(t, lo, hi, float(cand[i] + np.random.normal(0.0, sigma * span)))
        loss = objective(cand, rows)
        if loss < best_loss or np.random.rand() < math.exp(-(loss - best_loss) / max(1e-12, sigma)):
            best_vec = cand
            best_loss = loss
        sigma *= fit_cfg.sigma_decay
    fitted = {}
    for i, n in enumerate(names):
        t, lo, hi = specs_map[n]
        fitted[n] = _project(t, lo, hi, float(best_vec[i]))
    new_params = dict(params0)
    new_params.update(fitted)
    _write_yaml_like(out_yaml, new_params)
    rep_rows = []
    for r in rows:
        pred = _predict_from_params(r, new_params)
        meas_rho = r.get('meas_rho_c_Ohm_cm2', None)
        meas_dvoc = r.get('meas_DeltaVoc_V', None)
        meas_j0 = r.get('meas_J0int_A_cm2', None)
        meas_sheet = r.get('meas_sheet_ohm_sq', None)
        row_out = dict(r)
        row_out['pred_rho_c_Ohm_cm2'] = pred['rho_c_Ohm_cm2']
        row_out['pred_J0int_A_cm2'] = pred['J0int_A_cm2']
        row_out['pred_sheet_ohm_sq'] = pred['sheet_ohm_sq']
        if meas_rho is not None:
            row_out['rho_c_ratio_pred/obs'] = pred['rho_c_Ohm_cm2'] / max(1e-30, meas_rho)
        if meas_j0 is not None:
            row_out['J0int_ratio_pred/obs'] = pred['J0int_A_cm2'] / max(1e-30, meas_j0)
        if meas_dvoc is not None and J0_bulk > 0:
            Vt = 0.025852 * (T_K / 300.0)
            dv_pred = Vt * math.log1p(pred['J0int_A_cm2'] / J0_bulk)
            row_out['DeltaVoc_pred/obs'] = dv_pred / max(1e-12, meas_dvoc)
        if meas_sheet is not None:
            row_out['sheet_ratio_pred/obs'] = pred['sheet_ohm_sq'] / max(1e-30, meas_sheet)
        rep_rows.append(row_out)
    pd.DataFrame(rep_rows).to_csv(out_report_csv, index=False)
    return {'out_yaml': out_yaml, 'report_csv': out_report_csv, 'fitted_params': fitted}
import json, math, random
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from .lab_bridge import LabBridge, _load_simple_config
from .objectives import dvoc_from_J0int
PARAM_SPECS = [('rho_c_min_Ohm_cm2', 'pos', 1e-06, 0.01, 'rho'), ('rho_c_max_Ohm_cm2', 'pos', 0.001, 0.1, 'rho'), ('rho_k_t', 'pos', 0.0001, 1.0, 'rho'), ('rho_k_T', 'pos', 0.0001, 1.0, 'rho'), ('rho_T0_C', 'real', 0.0, 300.0, 'rho'), ('rho_k_tau', 'pos', 0.0001, 1.0, 'rho'), ('rho_k_dose', 'pos', 0.0001, 1.0, 'rho'), ('J0int_min_A_cm2', 'pos', 1e-20, 1e-15, 'j0'), ('J0int_max_A_cm2', 'pos', 1e-16, 1e-10, 'j0'), ('J0_k_pass', 'pos', 0.0001, 2.0, 'j0'), ('J0_k_surf', 'pos', 0.0001, 0.5, 'j0'), ('plasma_W_star', 'real', 0.0, 30.0, 'j0'), ('plasma_s_star', 'real', 0.0, 120.0, 'j0'), ('plasma_sigma_W', 'pos', 0.1, 20.0, 'j0'), ('plasma_sigma_s', 'pos', 0.1, 60.0, 'j0'), ('rho_bulk_Ohm_cm', 'pos', 1e-05, 0.01, 'sheet')]
