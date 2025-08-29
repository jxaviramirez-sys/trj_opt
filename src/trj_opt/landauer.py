# Reconstructed by merge of 7 variants for trj_opt/landauer.py
# Strategy: union of imports, union of classes/functions with longest/most-params preference.

import numpy as np

def _energy_to_joule(e_eV):
    return e_eV * q

def _ensure_monotonic(x):
    return np.all(np.diff(x) > 0)

def conductance_per_area_from_T(E_eV, T_of_E, T_K, area_m2=None, mu_eV=0.0, spin_deg=2.0, checks=True):
    """Compute small-signal conductance per area (S/m^2) from transmission T(E).

    This implements the linear-response Landauer formula:
        G = (g_s * q^2 / h) * \\int dE T(E) (-df/dE)
    If 'area_m2' is provided, returns G/A; otherwise returns G and sets area normalization to 1.

    Args:
        E_eV: energy grid (ascending, eV)
        T_of_E: transmission (dimensionless, total over transverse modes)
        T_K: temperature (K)
        area_m2: cross-sectional area corresponding to T(E) (m^2). If provided, returns G_per_area.
        mu_eV: Fermi level reference (eV) relative to E grid zero
        spin_deg: spin degeneracy (typically 2)
        checks: if True, run sanity checks and raise ValueError on failures.

    Returns:
        G_S: conductance (S) if area_m2 None, else conductance per area (S/m^2)
    """
    E_eV = np.asarray(E_eV, float)
    T_of_E = np.asarray(T_of_E, float)
    if checks:
        if not _ensure_monotonic(E_eV):
            raise ValueError('E grid must be strictly increasing')
        if np.any(T_of_E < -1e-12):
            raise ValueError('Transmission has negative values')
        if np.any(~np.isfinite(T_of_E)):
            raise ValueError('Transmission has NaNs/inf')
    integrand = T_of_E * -dfermi_dE(E_eV, mu_eV, T_K)
    integral_eV = np.trapz(integrand, x=E_eV)
    G = spin_deg * q * q / h * integral_eV
    if area_m2 is not None and area_m2 > 0:
        return G / area_m2
    return G

def dfermi_dE(E_eV, mu_eV, T_K):
    """Derivative df/dE in units of 1/eV."""
    kbT_eV = kB * T_K / q
    y = (E_eV - mu_eV) / (2.0 * kbT_eV)
    sech2 = 1.0 / np.cosh(np.clip(y, -700, 700)) ** 2
    return -sech2 / (4.0 * kbT_eV)

def fermi(E_eV, mu_eV, T_K):
    kbT_eV = kB * T_K / q
    x = (E_eV - mu_eV) / kbT_eV
    x = np.clip(x, -700, 700)
    return 1.0 / (1.0 + np.exp(x))

def landauer_current_from_T(E_eV, T_of_E, V, T_K, spin_deg=2, area_cm2=1.0):
    mu_L = V / 2.0
    mu_R = -V / 2.0
    fL = fermi(E_eV, mu_L, T_K)
    fR = fermi(E_eV, mu_R, T_K)
    prefactor = spin_deg * q / h
    integral = np.trapz(T_of_E * (fL - fR), x=E_eV)
    I_A = prefactor * integral * q
    J_A_cm2 = I_A / area_cm2
    return J_A_cm2

def small_signal_rho_c(E_eV, T_of_E, T_K, area_cm2=1.0, dV=0.001):
    Jp = landauer_current_from_T(E_eV, T_of_E, dV, T_K, area_cm2=area_cm2)
    Jm = landauer_current_from_T(E_eV, T_of_E, -dV, T_K, area_cm2=area_cm2)
    dJdV = (Jp - Jm) / (2 * dV)
    rho_c = 1.0 / dJdV
    return rho_c

def specific_contact_resistivity(E_eV, T_of_E, T_K, area_m2, mu_eV=0.0, spin_deg=2.0):
    """Return specific contact resistivity rho_c (Ohm*cm^2).

    rho_c = 1 / (G/A) with G from Landauer (linear response).
    """
    G_per_area = conductance_per_area_from_T(E_eV, T_of_E, T_K, area_m2=area_m2, mu_eV=mu_eV, spin_deg=spin_deg)
    if G_per_area <= 0:
        raise ValueError('Non-positive conductance per area')
    rho_SI = 1.0 / G_per_area
    rho_c_Ohm_cm2 = rho_SI * 10000.0
    return rho_c_Ohm_cm2
import numpy as np
q = 1.602176634e-19
h = 6.62607015e-34
kB = 1.380649e-23
hbar = h / (2.0 * np.pi)
