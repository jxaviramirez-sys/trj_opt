# Reconstructed by merge of 1 variants for trj_opt/calibration.py
# Strategy: union of imports, union of classes/functions with longest/most-params preference.

import json
import math
import os
import dataclasses
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

@dataclass
class CalibrationSettings:
    fudge_rho_c: float = 1.0
    fudge_J0_interface: float = 1.0
    anchor_rho_c_Ohm_cm2: Optional[float] = None
    anchor_lifetime_us: Optional[float] = None
    sheet_resistance_ohm_sq: Optional[float] = None
    ni_cm3: Optional[float] = None
    sigma_n_cm2: Optional[float] = None
    sigma_p_cm2: Optional[float] = None
    v_th_cm_s: Optional[float] = None
    NA_cm3: Optional[float] = None
    ND_cm3: Optional[float] = None
    measured_points_csv: Optional[str] = None

def _safe_median(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    if not x:
        return None
    return float(np.median(np.array(x, dtype=float)))

def apply_fudges_to_dataframe(df: pd.DataFrame, settings: CalibrationSettings) -> pd.DataFrame:
    AUTO_SCALING_BASELINES = {'sigma_n_cm2': 1e-15, 'sigma_p_cm2': 1e-15, 'v_th_cm_s': 10000000.0, 'ni_cm3': 10000000000.0}
    "Apply fudge factors to columns in-place and recompute derived metrics\n    that depend on J0_interface (i.e., DeltaVoc) using a light approximation.\n\n    DeltaVoc ~ (kT/q) * ln( (J0_bulk + J0_int * fudge) / (J0_bulk + J0_int) )\n    where the original DeltaVoc_interface_mV was computed from (J0_bulk, J0_int).\n    We don't know J0_bulk here; we infer it by inverting the original formula\n    assuming small-penalty linearization if necessary.\n    "
    df = df.copy()
    if 'rho_c_mOhm_cm2' in df.columns:
        df['rho_c_mOhm_cm2'] = df['rho_c_mOhm_cm2'].astype(float) * float(settings.fudge_rho_c)
    if 'J0_interface_A_cm2' in df.columns:
        J0_scale = float(settings.fudge_J0_interface)
        try:
            sig_n0 = AUTO_SCALING_BASELINES['sigma_n_cm2']
            sig_p0 = AUTO_SCALING_BASELINES['sigma_p_cm2']
            vth0 = AUTO_SCALING_BASELINES['v_th_cm_s']
            ni0 = AUTO_SCALING_BASELINES['ni_cm3']
            if settings.sigma_n_cm2 or settings.sigma_p_cm2:
                s_n = (settings.sigma_n_cm2 or sig_n0) / sig_n0
                s_p = (settings.sigma_p_cm2 or sig_p0) / sig_p0
                s_sig = (s_n * s_p) ** 0.5
                J0_scale *= float(s_sig)
            if settings.v_th_cm_s:
                J0_scale *= float(settings.v_th_cm_s / vth0)
            if settings.ni_cm3:
                J0_scale *= float((settings.ni_cm3 / ni0) ** 2)
        except Exception:
            pass
        df['J0_interface_A_cm2'] = df['J0_interface_A_cm2'].astype(float) * J0_scale
    if 'DeltaVoc_interface_mV' in df.columns and 'J0_interface_A_cm2' in df.columns:
        kT_over_q_mV = 25.852
        s = float(settings.fudge_J0_interface)
        if abs(s - 1.0) > 1e-12:
            J0_int = df['J0_interface_A_cm2'].astype(float) / s
            dVoc_mV_orig = df['DeltaVoc_interface_mV'].astype(float)
            with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                ratio = np.exp(dVoc_mV_orig / kT_over_q_mV) - 1.0
            ratio[~np.isfinite(ratio)] = np.nan
            J0_bulk_est = J0_int / ratio.replace(0.0, np.nan)
            J0_int_new = J0_int * s
            with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                dv_new = kT_over_q_mV * np.log(1.0 + J0_int_new / J0_bulk_est)
            dv_new = dv_new.replace([np.inf, -np.inf], np.nan)
            df['DeltaVoc_interface_mV'] = dv_new.astype(float)
    return df

def calibrate_csv(analyzed_csv: str, out_csv: str, settings_path: Optional[str]=None) -> str:
    """High-level helper: read analyzed CSV, compute anchor multipliers,
    apply fudges, and write calibrated CSV."""
    settings = load_settings(settings_path)
    df = pd.read_csv(analyzed_csv)
    settings2 = compute_anchor_multipliers(df, settings)
    df2 = apply_fudges_to_dataframe(df, settings2)
    df2.to_csv(out_csv, index=False)
    return out_csv

def compute_anchor_multipliers(df: pd.DataFrame, settings: CalibrationSettings) -> CalibrationSettings:
    """Derive fudge multipliers from measured anchors if provided.

    - If anchor_rho_c_Ohm_cm2 is provided, set fudge_rho_c so that the median
      model rho_c equals the measured value.
    - anchor_lifetime_us is stored but not auto-mapped to J0_bulk here because
      the mapping is device-specific. We keep it available for the pipeline
      to convert to J0_bulk elsewhere if desired.
    """
    out = dataclasses.replace(settings)
    if settings.anchor_rho_c_Ohm_cm2 is not None:
        med = _safe_median(df.get('rho_c_mOhm_cm2', []))
        if med and med > 0:
            target_mOhm_cm2 = 1000.0 * float(settings.anchor_rho_c_Ohm_cm2)
            out.fudge_rho_c = float(target_mOhm_cm2 / med)
    return out

def load_measured_points(csv_path: Optional[str]) -> Optional[pd.DataFrame]:
    if not csv_path:
        return None
    try:
        m = pd.read_csv(csv_path)
        rmap = {'rho_c': 'rho_c_mOhm_cm2', 'rhoc': 'rho_c_mOhm_cm2', 'DeltaVoc': 'DeltaVoc_interface_mV', 'DeltaVoc_mV': 'DeltaVoc_interface_mV', 'dvoc_mV': 'DeltaVoc_interface_mV', 'label': 'label'}
        cols = {c: rmap.get(c, c) for c in m.columns}
        m = m.rename(columns=cols)
        needed = {'rho_c_mOhm_cm2', 'DeltaVoc_interface_mV'}
        missing = needed - set(m.columns)
        if missing:
            raise ValueError(f'Measured points CSV missing columns: {missing}')
        return m
    except Exception as e:
        raise RuntimeError(f'Failed to load measured points from {csv_path}: {e}')

def load_settings(path: Optional[str]) -> CalibrationSettings:
    """Load settings from a JSON or YAML file (simple, minimal parser)."""
    if path is None:
        return CalibrationSettings()

    def _load_any(p: str) -> Dict[str, Any]:
        with open(p, 'r') as f:
            s = f.read()
        if p.endswith(('.yaml', '.yml')):
            data: Dict[str, Any] = {}
            for ln in s.splitlines():
                ln = ln.strip()
                if not ln or ln.startswith('#'):
                    continue
                if ':' in ln:
                    k, v = ln.split(':', 1)
                    k = k.strip()
                    v = v.strip()
                    if v.lower() in ('true', 'false'):
                        data[k] = v.lower() == 'true'
                    else:
                        try:
                            if '.' in v or 'e' in v.lower():
                                data[k] = float(v)
                            else:
                                data[k] = int(v)
                        except Exception:
                            data[k] = v
            return data
        else:
            return json.loads(s)
    raw = _load_any(path)
    mapping = {}
    aliases = {'fudge_rho_c': ['rho_c_multiplier', 'fudge_rho_c'], 'fudge_J0_interface': ['J0_multiplier', 'fudge_J0_interface', 'fudge_J0'], 'anchor_rho_c_Ohm_cm2': ['anchor_rho_c', 'measured_rho_c_Ohm_cm2'], 'anchor_lifetime_us': ['anchor_lifetime', 'measured_lifetime_us'], 'sheet_resistance_ohm_sq': ['measured_sheet_resistance', 'sheet_R_ohm_sq'], 'ni_cm3': ['n_i', 'ni_cm3'], 'sigma_n_cm2': ['sigma_n', 'sigma_n_cm2'], 'sigma_p_cm2': ['sigma_p', 'sigma_p_cm2'], 'v_th_cm_s': ['v_th', 'v_th_cm_s'], 'NA_cm3': ['NA', 'N_A_cm3', 'NA_cm3'], 'ND_cm3': ['ND', 'N_D_cm3', 'ND_cm3'], 'measured_points_csv': ['measured_csv', 'points_csv']}
    for field in dataclasses.fields(CalibrationSettings):
        val = None
        for k in aliases.get(field.name, [field.name]):
            if k in raw:
                val = raw[k]
                break
        mapping[field.name] = val if val is not None else getattr(CalibrationSettings(), field.name)
    return CalibrationSettings(**mapping)
import json
import math
import os
import dataclasses
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
