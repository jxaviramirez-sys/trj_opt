# Reconstructed by merge of 7 variants for trj_opt/analyze.py
# Strategy: union of imports, union of classes/functions with longest/most-params preference.

import os, pandas as pd, numpy as np
from . import landauer, srh
from .io import read_te_csv, read_dit_csv

def analyze_batch(doe_csv, te_dir, dos_dir, out_path, tempK=300.0, area_cm2=None, NA=1e+17, ND=1e+17, J0_bulk=1e-15, EF_eV=0.0, delta_qF_eV=0.0, material_name=None, material_overrides=None):
    """
    Batch analysis joining a DOE CSV with run_{id}_T.csv and run_{id}_Dit.csv files.
    Writes a CSV with columns including rho_c_mOhm_cm2, S_interface_cm_s, DeltaVoc_interface_mV, J0_interface_A_cm2.
    v0.2.0: Enforces area metadata for Landauer; supports material presets/overrides.
    """
    df = pd.read_csv(doe_csv)
    for idx, row in df.iterrows():
        run_id = int(row['run_id'])
        te_file = os.path.join(te_dir, f'run_{run_id}_T.csv')
        dos_file = os.path.join(dos_dir, f'run_{run_id}_Dit.csv')
        if os.path.exists(te_file):
            transport_warning = None
            E, T, metaT = read_te_csv(te_file)
            area_m2 = None
            area_source = None
            if isinstance(metaT, dict):
                if 'area_m2' in metaT:
                    area_m2 = float(metaT['area_m2'])
                    area_source = 'header:area_m2'
                elif 'A_cell_A2' in metaT:
                    area_m2 = float(metaT['A_cell_A2']) * 1e-20
                    area_source = 'header:A_cell_A2'
            if area_m2 is None and area_cm2 is not None:
                area_m2 = float(area_cm2) * 0.0001
                area_source = 'flag:area_cm2'
            if area_m2 is None or not area_m2 > 0:
                raise ValueError('Missing cross-section: provide # area_m2, # A_cell_A2, or --area-cm2')
            EF_eff = float(metaT.get('Ef_eV', EF_eV)) if isinstance(metaT, dict) else float(EF_eV)
            if 'Ef_eV' not in (metaT or {}):
                transport_warning = (transport_warning or '') + 'EF not provided; default 0 eV. '
            try:
                rho_c = landauer.specific_contact_resistivity(E, T, tempK, area_m2=area_m2, mu_eV=EF_eff)
            except Exception as e:
                rho_c = np.nan
                df.at[idx, 'rho_c_error'] = str(e)
            df.at[idx, 'rho_c_mOhm_cm2'] = 1000.0 * rho_c if np.isfinite(rho_c) else np.nan
            if transport_warning:
                df.at[idx, 'transport_warning'] = transport_warning
            df.at[idx, 'area_source'] = area_source
        if os.path.exists(dos_file):
            E, Dit, metaD = read_dit_csv(dos_file)
            try:
                mat = srh.resolve_material(material_name, material_overrides)
                S_eff, J0_int, dVoc_V = srh.compute_from_Dit(E, Dit, T_K=tempK, N_A_cm3=NA, N_D_cm3=ND, J0_bulk_A_cm2=J0_bulk, delta_qF_eV=delta_qF_eV, material=mat)
                df.at[idx, 'S_interface_cm_s'] = S_eff
                df.at[idx, 'DeltaVoc_interface_mV'] = 1000.0 * dVoc_V
                df.at[idx, 'J0_interface_A_cm2'] = J0_int
            except Exception as e:
                df.at[idx, 'interface_error'] = str(e)
    df.to_csv(out_path, index=False)
    return out_path
import os, pandas as pd, numpy as np
from . import landauer, srh
from .io import read_te_csv, read_dit_csv
