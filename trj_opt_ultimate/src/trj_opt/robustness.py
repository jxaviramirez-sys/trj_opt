# Reconstructed by merge of 7 variants for trj_opt/robustness.py
# Strategy: union of imports, union of classes/functions with longest/most-params preference.

import numpy as np
import pandas as pd

def ci_from_samples(series, alpha=0.05):
    lo = np.quantile(series, alpha / 2)
    hi = np.quantile(series, 1 - alpha / 2)
    med = np.quantile(series, 0.5)
    return {'median': float(med), f'ci{int((1 - alpha) * 100)}_lo': float(lo), f'ci{int((1 - alpha) * 100)}_hi': float(hi)}

def local_sensitivity(df, run_id, t_col='t_ITO_nm', rho_col='rho_c_mOhm_cm2', dv_col='DeltaVoc_interface_mV'):
    row = df[df['run_id'] == run_id]
    if row.empty:
        return None
    t0 = row.iloc[0][t_col]
    mask = (df['interlayer'] == row.iloc[0]['interlayer']) & (df['V_O_cm^-3'] == row.iloc[0]['V_O_cm^-3']) & (df['anneal_C'] == row.iloc[0]['anneal_C']) & (df['perovskite_termination'] == row.iloc[0]['perovskite_termination']) & (df['surface_texture'] == row.iloc[0]['surface_texture'])
    fam = df[mask].dropna(subset=[rho_col, dv_col, t_col]).sort_values(t_col)
    if len(fam) < 3:
        return None
    lower = fam[fam[t_col] < t0].tail(1)
    upper = fam[fam[t_col] > t0].head(1)
    if lower.empty or upper.empty:
        return None
    t1, r1, d1 = (float(lower[t_col]), float(lower[rho_col]), float(lower[dv_col]))
    t2, r2, d2 = (float(upper[t_col]), float(upper[rho_col]), float(upper[dv_col]))
    dr_dt = (r2 - r1) / (t2 - t1)
    ddv_dt = (d2 - d1) / (t2 - t1)
    return {'run_id': run_id, 't0': t0, 'dr_dt_mOhmcm2_per_nm': dr_dt, 'ddVoc_dt_mV_per_nm': ddv_dt}

def mc_robustness(fn_compute, inputs, noise_spec, nsamples=200, random_state=0):
    """Generic Monte Carlo robustness.

    Args:
      fn_compute: callable(**kwargs) -> dict of metrics
      inputs: dict of baseline keyword args
      noise_spec: dict mapping key -> (mode, value) where mode in {"rel","abs"} for Gaussian sigma
      nsamples: number of MC draws
    Returns:
      DataFrame of samples with columns from fn_compute outputs
    """
    rng = np.random.default_rng(random_state)

    def jittered():
        kw = {}
        for k, v in inputs.items():
            if k in noise_spec:
                mode, sig = noise_spec[k]
                if mode == 'rel':
                    val = v * (1.0 + rng.normal(0.0, sig))
                else:
                    val = v + rng.normal(0.0, sig)
                kw[k] = val
            else:
                kw[k] = v
        return kw
    rows = []
    for i in range(nsamples):
        res = fn_compute(**jittered())
        res['_seed_idx'] = i
        rows.append(res)
    return pd.DataFrame(rows)
import numpy as np
import pandas as pd
