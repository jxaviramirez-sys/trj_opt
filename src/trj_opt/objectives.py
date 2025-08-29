# Reconstructed by merge of 1 variants for trj_opt/objectives.py
# Strategy: union of imports, union of classes/functions with longest/most-params preference.

import math
from typing import Dict, Tuple, Optional
import numpy as np

def dvoc_from_J0int(J0_bulk_A_cm2: float, J0_int_A_cm2: float, T: float=300.0) -> float:
    if J0_bulk_A_cm2 <= 0.0:
        return 0.0
    return kT_over_q(T) * float(np.log1p(max(0.0, J0_int_A_cm2) / float(J0_bulk_A_cm2)))

def efficiency_percent(Jsc_mA_cm2: float, Voc_bulk_V: float, J0_bulk_A_cm2: float, J0_int_A_cm2: float, Rs_area_Ohm_cm2: float, n: float=1.0, T: float=300.0, Rsh_area_Ohm_cm2: float=1000000000000.0, Pin_W_m2: float=1000.0) -> Dict[str, float]:
    dVoc_V = dvoc_from_J0int(J0_bulk_A_cm2, J0_int_A_cm2, T=T)
    Voc_V = max(0.0, Voc_bulk_V - dVoc_V)
    Pin_W_cm2 = float(Pin_W_m2) / 10000.0
    Jsc_A_cm2 = float(Jsc_mA_cm2) * 0.001
    FF = ff_from_rs(Jsc_A_cm2, Voc_V, Rs_area_Ohm_cm2, n=n, T=T, Rsh_area_Ohm_cm2=Rsh_area_Ohm_cm2)
    eta = 100.0 * (Voc_V * Jsc_A_cm2 * FF) / max(1e-12, Pin_W_cm2)
    return {'eta_percent': float(eta), 'FF': float(FF), 'Voc_V': float(Voc_V), 'DeltaVoc_V': float(dVoc_V)}

def estimate_J0_from_Voc_Jsc(Voc_V: float, Jsc_A_cm2: float, n: float=1.0, T: float=300.0) -> float:
    Vt = n * kT_over_q(T)
    arg = Voc_V / max(1e-12, Vt)
    return float(Jsc_A_cm2 / max(1e-30, np.expm1(arg)))

def ff_from_rs(Jsc_A_cm2: float, Voc_V: float, Rs_area_Ohm_cm2: float, n: float=1.0, T: float=300.0, Rsh_area_Ohm_cm2: float=1000000000000.0) -> float:
    J0_est = estimate_J0_from_Voc_Jsc(Voc_V, Jsc_A_cm2, n=n, T=T)
    V, J = iv_curve_single_diode(Jsc_A_cm2, J0_est, n, T, Rs_area_Ohm_cm2, Rsh_area_Ohm_cm2, npts=400)
    P = V * J
    Pmp = np.max(P)
    FF = float(Pmp / max(1e-12, Voc_V * Jsc_A_cm2))
    return float(np.clip(FF, 0.0, 0.95))

def iv_curve_single_diode(Jsc_A_cm2: float, J0_A_cm2: float, n: float, T: float, Rs_area_Ohm_cm2: float=0.0, Rsh_area_Ohm_cm2: float=1000000000000.0, npts: int=300):
    Vt = n * kT_over_q(T)
    Voc0 = Vt * np.log1p(max(0.0, Jsc_A_cm2) / max(1e-30, J0_A_cm2))
    V = np.linspace(0.0, max(1e-06, Voc0), npts)
    J = np.empty_like(V)
    for i, v in enumerate(V):
        j = Jsc_A_cm2 * max(0.0, 1.0 - v / max(1e-09, Voc0))
        for _ in range(60):
            exp_arg = (v + j * Rs_area_Ohm_cm2) / max(1e-12, Vt)
            ej = math.exp(min(60.0, exp_arg))
            f = Jsc_A_cm2 - J0_A_cm2 * (ej - 1.0) - (v + j * Rs_area_Ohm_cm2) / max(1e-30, Rsh_area_Ohm_cm2) - j
            df = -J0_A_cm2 * ej * (Rs_area_Ohm_cm2 / max(1e-12, Vt)) - Rs_area_Ohm_cm2 / max(1e-30, Rsh_area_Ohm_cm2) - 1.0
            if abs(df) < 1e-14:
                break
            step = f / df
            jn = j - step
            if not np.isfinite(jn):
                break
            if abs(step) < 1e-09:
                j = jn
                break
            j = jn
        J[i] = max(-1000.0, min(1000.0, j))
    sign_change = np.where(np.diff(np.signbit(J)))[0]
    if len(sign_change) > 0:
        idx = sign_change[-1]
        x1, x2 = (V[idx], V[idx + 1])
        y1, y2 = (J[idx], J[idx + 1])
        if y2 - y1 != 0:
            voc = x1 - y1 * (x2 - x1) / (y2 - y1)
        else:
            voc = x2
        V = np.linspace(0.0, max(1e-06, voc), npts)
        J = np.interp(V, [x1, x2], [y1, y2], left=J[0], right=0.0)
    return (V, J)

def kT_over_q(T: float=300.0) -> float:
    return k_B * T / q

def series_resistance_area(rho_c_Ohm_cm2: float, sheet_R_ohm_sq: Optional[float]=None, geom_factor: float=0.0, Rs_other_Ohm_cm2: float=0.0) -> float:
    rs = max(0.0, float(rho_c_Ohm_cm2))
    if sheet_R_ohm_sq is not None:
        rs += max(0.0, geom_factor) * max(0.0, float(sheet_R_ohm_sq))
    rs += max(0.0, float(Rs_other_Ohm_cm2))
    return rs
import math
from typing import Dict, Tuple, Optional
import numpy as np
k_B = 1.380649e-23
q = 1.602176634e-19
