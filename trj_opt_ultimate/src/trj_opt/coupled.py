# Reconstructed by merge of 3 variants for trj_opt/coupled.py
# Strategy: union of imports, union of classes/functions with longest/most-params preference.

import numpy as np
from dataclasses import dataclass
from typing import Sequence, Tuple
from .landauer import q, kB, h, dfermi_dE

class InterfaceSystem:
    """Collection of levels; computes multi-terminal transmissions in WBL (dimensionless).
    If you want per-area, provide Γ that already reflect per-area coupling.
    """

    def __init__(self, levels: Sequence[Level]):
        self.levels = list(levels)

    @staticmethod
    def _lorentz(Ga, Gb, Gtot, eps, E):
        denom = (E - eps) ** 2 + (0.5 * Gtot) ** 2
        return Ga * Gb / denom

    def T_ab(self, E_eV, a: str, b: str):
        idx = {'L': 0, 'R': 1, 'C': 2, 'V': 3}
        ai, bi = (idx[a], idx[b])
        E = np.atleast_1d(E_eV).astype(float)
        T = np.zeros_like(E)
        for lvl in self.levels:
            GL, GR, GC, GV = lvl.Gammas()
            Gtot = GL + GR + GC + GV
            Ga = (GL, GR, GC, GV)[ai]
            Gb = (GL, GR, GC, GV)[bi]
            T += self._lorentz(Ga, Gb, Gtot, lvl.epsilon_eV, E)
        return T if E.ndim else T.item()

@dataclass
class Level:
    """Single localized level in WBL with four terminals (energies in eV, Gammas in eV)."""
    epsilon_eV: float
    GammaL_eV: float
    GammaR_eV: float
    GammaC_eV: float
    GammaV_eV: float

    def Gammas(self) -> Tuple[float, float, float, float]:
        return (self.GammaL_eV, self.GammaR_eV, self.GammaC_eV, self.GammaV_eV)

    def Gamma_tot(self) -> float:
        GL, GR, GC, GV = self.Gammas()
        return GL + GR + GC + GV

def J0_int_from_jCV(jCV_S_per_m2: float, T_K: float) -> float:
    return kB * T_K / q * jCV_S_per_m2

def delta_Voc(J0_int_A_m2: float, J0_bulk_A_m2: float, T_K: float) -> float:
    return kB * T_K / q * np.log(1.0 + J0_int_A_m2 / J0_bulk_A_m2)

def dfermi_window(E_eV, mu_eV, T_K):
    kbT_eV = kB * T_K / q
    y = (E_eV - mu_eV) / (2.0 * kbT_eV)
    sech2 = 1.0 / np.cosh(np.clip(y, -700, 700)) ** 2
    return 1.0 / (4.0 * kbT_eV) * sech2

def gLR_from_T(TLR_of_E, E_eV, mu_eV, T_K, spin_deg=2.0):
    integrand = TLR_of_E * dfermi_window(E_eV, mu_eV, T_K)
    integral = np.trapz(integrand, x=E_eV)
    return spin_deg * q * q / h * integral

def gLR_from_system(sys: InterfaceSystem, E_eV, T_K, mu_eV=0.0, spin_deg=2.0):
    """g_{LR} per area (S/m^2) from system's T_{LR}(E)."""
    E = np.asarray(E_eV, float)
    TLR = sys.T_ab(E, 'L', 'R')
    integrand = TLR * -dfermi_dE(E, mu_eV, T_K)
    integral = np.trapz(integrand, x=E)
    G = spin_deg * q * q / h * integral
    return G

def jCV_from_T(TCV_of_E, E_eV, mu_eV, T_K, spin_deg=2.0):
    integrand = TCV_of_E * dfermi_window(E_eV, mu_eV, T_K)
    integral = np.trapz(integrand, x=E_eV)
    return spin_deg * q * q / h * integral

def jCV_from_system(sys: InterfaceSystem, E_eV, T_K, mu_eV=0.0, spin_deg=2.0):
    """j_{CV} per area (S/m^2) from system's T_{CV}(E)."""
    E = np.asarray(E_eV, float)
    TCV = sys.T_ab(E, 'C', 'V')
    integrand = TCV * -dfermi_dE(E, mu_eV, T_K)
    integral = np.trapz(integrand, x=E)
    j = spin_deg * q * q / h * integral
    return j

def lorentz_transmission(E_eV, eps_eV, Gam_a_eV, Gam_b_eV, Gam_tot_eV):
    denom = (E_eV - eps_eV) ** 2 + (0.5 * Gam_tot_eV) ** 2
    return Gam_a_eV * Gam_b_eV / denom

def multi_terminal_T(E_eV, eps_eV, GamL_eV, GamR_eV, GamC_eV, GamV_eV):
    Gam_tot = GamL_eV + GamR_eV + GamC_eV + GamV_eV

    def T_ab(a, b):
        Ga = {'L': GamL_eV, 'R': GamR_eV, 'C': GamC_eV, 'V': GamV_eV}[a]
        Gb = {'L': GamL_eV, 'R': GamR_eV, 'C': GamC_eV, 'V': GamV_eV}[b]
        return lorentz_transmission(E_eV, eps_eV, Ga, Gb, Gam_tot)
    return (T_ab, Gam_tot)

def rho_c_from_gLR(gLR_S_per_m2: float) -> float:
    if gLR_S_per_m2 <= 0:
        raise ValueError('Non-positive gLR')
    return 1.0 / gLR_S_per_m2

def tradeoff_bound_point(Gamma_min_eV: float, Gamma_tot_eV: float, T_K: float) -> float:
    """Return lower bound on product rho_c * J0_int in Volts."""
    num = 4.0 * Gamma_min_eV ** 2
    den = (Gamma_tot_eV - 2.0 * Gamma_min_eV) ** 2
    return kB * T_K / q * (num / den)
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Tuple
from .landauer import q, kB, h, dfermi_dE
