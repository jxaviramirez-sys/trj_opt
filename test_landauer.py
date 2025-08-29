
import numpy as np
from trj_opt.landauer import specific_contact_resistivity

def test_rho_monotonic_with_T():
    E = np.linspace(-0.5, 0.5, 1001)
    Tweak = np.ones_like(E)*1e-3
    area_m2 = 1e-12
    rho1 = specific_contact_resistivity(E, Tweak, 300.0, area_m2)
    Tweak2 = Tweak*2
    rho2 = specific_contact_resistivity(E, Tweak2, 300.0, area_m2)
    assert rho2 < rho1
