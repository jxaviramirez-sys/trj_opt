# Reconstructed by merge of 1 variants for trj_opt/lab_bridge.py
# Strategy: union of imports, union of classes/functions with longest/most-params preference.

import json
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np

@dataclass
class LabBridge:
    params: Dict[str, Any]

    @classmethod
    def from_file(cls, path: str) -> 'LabBridge':
        return cls(params=_load_simple_config(path))

    def bounds(self) -> Dict[str, Tuple[float, float]]:
        B = {}
        for k, v in self.params.items():
            if k.endswith('_min') or k.endswith('_max'):
                base = k[:-4]
                lo = self.params.get(base + '_min', None)
                hi = self.params.get(base + '_max', None)
                if lo is not None and hi is not None:
                    B[base] = (float(lo), float(hi))
        return B

    def lab_to_model(self, x: Dict[str, float]) -> Dict[str, float]:
        p = self.params

        def g(name, default=0.0):
            return float(p.get(name, default))

        def gx(key):
            return float(x.get(key, float(p.get(key + '_init', 0.0))))
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
        f_t = 1.0 - np.exp(-k_t * max(0.0, t_nm))
        f_T = 1.0 - np.exp(-k_T * max(0.0, T_C - T0))
        f_tau = 1.0 - np.exp(-k_tau * max(0.0, t_min))
        f_dose = 1.0 - np.exp(-k_dose * max(0.0, dope_dose))
        contact_gain = f_t * f_T * f_tau * (0.5 + 0.5 * f_dose)
        rho_c = rho_max - (rho_max - rho_min) * contact_gain
        rho_c = float(max(rho_min, min(rho_max, rho_c)))
        J0_min = g('J0int_min_A_cm2', 1e-18)
        J0_max = g('J0int_max_A_cm2', 1e-12)
        k_pass = g('J0_k_pass', 0.25)
        k_surf = g('J0_k_surf', 0.015)
        P_W_star = g('plasma_W_star', 8.0)
        P_t_star = g('plasma_s_star', 15.0)
        P_sigma_W = max(0.001, g('plasma_sigma_W', 3.0))
        P_sigma_t = max(0.001, g('plasma_sigma_s', 6.0))
        penalty_plasma = np.exp(-0.5 * ((plasma_W - P_W_star) / P_sigma_W) ** 2 - 0.5 * ((plasma_s - P_t_star) / P_sigma_t) ** 2)
        pass_gain = 1.0 - np.exp(-k_pass * max(0.0, passiv_nm))
        surf_gain = 1.0 - np.exp(-k_surf * max(0.0, surf_s))
        reduc = 0.6 * pass_gain + 0.4 * surf_gain
        J0_int = J0_max - (J0_max - J0_min) * reduc * penalty_plasma
        J0_int = float(max(J0_min, min(J0_max, J0_int)))
        rho_bulk_Ohm_cm = g('rho_bulk_Ohm_cm', 0.0002)
        t_cm = max(1e-09, t_nm * 1e-07)
        sheet_R_ohm_sq = float(rho_bulk_Ohm_cm / t_cm)
        return {'rho_c_Ohm_cm2': rho_c, 'J0_interface_A_cm2': J0_int, 'sheet_resistance_ohm_sq': sheet_R_ohm_sq}

def _load_simple_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        s = f.read()
    if path.endswith(('.yaml', '.yml')):
        data: Dict[str, Any] = {}
        for ln in s.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith('#') or ':' not in ln:
                continue
            k, v = ln.split(':', 1)
            k = k.strip()
            v = v.strip()
            if v.lower() in ('true', 'false'):
                data[k] = v.lower() == 'true'
            else:
                try:
                    if any((ch in v for ch in ['.', 'e', 'E'])):
                        data[k] = float(v)
                    else:
                        data[k] = int(v)
                except Exception:
                    data[k] = v
        return data
    else:
        return json.loads(s)
import json
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
