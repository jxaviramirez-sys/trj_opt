
import numpy as np
from trj_opt.srh import compute_from_Dit

def test_interface_dvoc_limits():
    E = np.linspace(-0.6, 0.6, 601)
    Dit = np.zeros_like(E)
    S, J0i, dVoc = compute_from_Dit(E, Dit, T_K=300.0)
    assert S < 1e-6
    assert J0i < 1e-30
    assert abs(dVoc) < 1e-6
