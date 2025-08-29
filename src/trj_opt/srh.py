# Reconstructed by merge of 7 variants for trj_opt/srh.py
# Strategy: union of imports, union of classes/functions with longest/most-params preference.

import numpy as np
import json as _json, os as _os

def _load_json(path):
    with open(path, 'r') as f:
        return _json.load(f)

def compute_S_eff_from_Dit(ener_eV, Dit_cm2eV, T_K=300.0, sigma_n_cm2=1e-15, sigma_p_cm2=1e-15, v_th_cm_s=10000000.0, E_i_eV=0.0):
    kT_eV = kB * T_K / q
    w = sech2((ener_eV - E_i_eV) / (2.0 * kT_eV))
    N_t_eff = np.trapz(Dit_cm2eV * w, x=ener_eV)
    sigma_eff = 0.5 * (sigma_n_cm2 + sigma_p_cm2)
    S_eff = v_th_cm_s * sigma_eff * N_t_eff
    return (S_eff, N_t_eff)

def compute_from_Dit(E_eV, Dit_cm2eV, T_K=300.0, sigma_n_cm2=1e-15, sigma_p_cm2=1e-15, v_th_cm_s=10000000.0, E_i_eV=0.0, N_A_cm3=1e+17, N_D_cm3=1e+17, J0_bulk_A_cm2=1e-15, delta_qF_eV=0.0):
    """Main helper: returns (S_eff, J0_interface, ΔVoc_interface[V]).

    This approach is still an approximation; we document assumptions explicitly.
    """
    ni_cm3 = ni_Si_approx(T_K)
    n_s, p_s = estimate_surface_carriers(N_A_cm3, N_D_cm3, T_K, ni_cm3, delta_qF_eV)
    S_eff = srh_surface_U_per_Dit(E_eV, Dit_cm2eV, T_K, n_s, p_s, sigma_n_cm2, sigma_p_cm2, v_th_cm_s, E_i_eV, ni_cm3)
    J0_int = j0_interface(S_eff, N_A_cm3, N_D_cm3, ni_cm3, T_K)
    dVoc_V = delta_voc(J0_bulk_A_cm2, J0_int, T_K)
    return (S_eff, J0_int, dVoc_V)

def delta_voc(J0_bulk_A_cm2, J0_interface_A_cm2, T_K=300.0, Jsc_A_cm2=0.04):
    """Voc penalty in Volts when adding J0_interface in parallel to J0_bulk.

    Voc = (kT/q) ln( (Jsc/J0_total) + 1 ); ΔVoc = Voc_bulk - Voc_total.
    For Jsc >> J0, ΔVoc ≈ (kT/q) ln( (J0_bulk + J0_interface)/J0_bulk ).
    """
    kbT_over_q = kB * T_K / q
    J0_tot = J0_bulk_A_cm2 + J0_interface_A_cm2
    return kbT_over_q * np.log(max(J0_tot, 1e-300) / max(J0_bulk_A_cm2, 1e-300))

def delta_voc_from_J0s(J0_rad_A_cm2=1e-15, J0_int_A_cm2=1e-13, T_K=300.0):
    kbT_over_q = kB * T_K / q
    return kbT_over_q * np.log((J0_rad_A_cm2 + J0_int_A_cm2) / J0_rad_A_cm2)

def estimate_surface_carriers(N_A_cm3=1e+17, N_D_cm3=1e+17, T_K=300.0, ni_cm3=None, delta_qF_eV=0.0):
    """Rough estimate of surface carrier densities under quasi-equilibrium.

    Uses n_s ≈ n0 * exp(+ΔqF/kT), p_s ≈ p0 * exp(+ΔqF/kT) for simplicity.
    For an n/p junction at open-circuit, ΔqF is the local quasi-Fermi splitting (eV).
    """
    if ni_cm3 is None:
        ni_cm3 = ni_Si_approx(T_K)
    n0 = N_D_cm3 + ni_cm3 ** 2 / max(N_A_cm3, 1e-30)
    p0 = N_A_cm3 + ni_cm3 ** 2 / max(N_D_cm3, 1e-30)
    kT_eV = kB * T_K / q
    factor = np.exp(np.clip(delta_qF_eV / kT_eV, -50, 50))
    n_s = n0 * factor
    p_s = p0 * factor
    return (n_s, p_s)

def intrinsic_carrier_concentration_Si(TK=300.0):
    return 10000000000.0

def j0_interface(S_eff_cm_s, N_A_cm3=None, N_D_cm3=None, ni_cm3=None, T_K=300.0):
    """J0 at the interface for a p-n junction with symmetric surfaces.

    J0_int ≈ q * ni^2 * S_eff * (1/N_A + 1/N_D)
    """
    if ni_cm3 is None:
        ni_cm3 = ni_Si_approx(T_K)
    if N_A_cm3 is None:
        N_A_cm3 = 1e+17
    if N_D_cm3 is None:
        N_D_cm3 = 1e+17
    return q * ni_cm3 ** 2 * S_eff_cm_s * (1.0 / max(N_A_cm3, 1e-30) + 1.0 / max(N_D_cm3, 1e-30))

def j0_interface_from_S(S_eff_cm_s, N_A_cm3=1e+17, N_D_cm3=1e+17, ni_cm3=None, T_K=300.0):
    if ni_cm3 is None:
        ni_cm3 = intrinsic_carrier_concentration_Si(T_K)
    J0 = q * ni_cm3 ** 2 * S_eff_cm_s * (1.0 / N_A_cm3 + 1.0 / N_D_cm3)
    return J0

def load_material_preset(name):
    """Load material presets from data/materials.json; returns dict or None if not found."""
    try:
        here = _os.path.dirname(__file__)
        mj = _load_json(_os.path.join(here, 'data', 'materials.json'))
        return mj.get(name)
    except Exception:
        return None

def n1_p1(Et_minus_Ei_eV, T_K, ni_cm3):
    kT_eV = kB * T_K / q
    n1 = ni_cm3 * np.exp(+Et_minus_Ei_eV / kT_eV)
    p1 = ni_cm3 * np.exp(-Et_minus_Ei_eV / kT_eV)
    return (n1, p1)

def ni_Si_approx(T_K=300.0):
    """Very rough intrinsic carrier concentration for Si (cm^-3)."""
    return 10000000000.0 * (T_K / 300.0) ** 1.5

def resolve_material(name=None, overrides=None):
    """Return a dict of material parameters (ni at 300K, Eg, Nc, Nv, sigma_n/p, v_th at 300K)."""
    mat = {}
    if name:
        base = load_material_preset(name)
        if base:
            mat.update(base)
    if overrides:
        mat.update({k: v for k, v in overrides.items() if v is not None})
    return mat or None

def sech2(x):
    c = np.cosh(x)
    return 1.0 / (c * c)

def srh_surface_U_per_Dit(E_eV, Dit_cm2eV, T_K, n_s, p_s, sigma_n_cm2=1e-15, sigma_p_cm2=1e-15, v_th_cm_s=10000000.0, E_i_eV=0.0, ni_cm3=None):
    """Compute SRH recombination rate per unit Dit (cm/s) weighting over energy.

    Returns S_eff (cm/s) after integrating Dit * s(E) dE where
        s(E) = v_th * (sigma_n*sigma_p*(n_s+p_s)) / (sigma_n*(n_s + n1) + sigma_p*(p_s + p1))
    This is a common approximation for interface SRH limited by capture kinetics.
    """
    if ni_cm3 is None:
        ni_cm3 = ni_Si_approx(T_K)
    kT_eV = kB * T_K / q
    Et_minus_Ei = E_eV - E_i_eV
    n1, p1 = n1_p1(Et_minus_Ei, T_K, ni_cm3)
    num = sigma_n_cm2 * sigma_p_cm2 * (n_s + p_s)
    den = sigma_n_cm2 * (n_s + n1) + sigma_p_cm2 * (p_s + p1)
    s_E = v_th_cm_s * num / np.maximum(den, 1e-300)
    S_eff = np.trapz(Dit_cm2eV * s_E, x=E_eV)
    return S_eff
import numpy as np
import json as _json, os as _os
q = 1.602176634e-19
kB = 1.380649e-23
