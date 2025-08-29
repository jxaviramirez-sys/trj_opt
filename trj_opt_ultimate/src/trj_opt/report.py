# Reconstructed by merge of 7 variants for trj_opt/report.py
# Strategy: union of imports, union of classes/functions with longest/most-params preference.

import os, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from .pareto import pareto_front, plot_pareto
from .robustness import local_sensitivity, ci_from_samples
from .robustness import local_sensitivity

def _add_bound_overlay_plot(df, out_dir, T_K, Gamma_min_meV, Gamma_tot_meV):
    """Creates 'pareto_rhoc_vs_J0int_with_bound.png' overlaying the no-go curve y = bound/x.
    Expects df columns: 'rho_c_mOhm_cm2', 'J0_interface_A_cm2'.
    """
    import matplotlib.pyplot as plt
    rho_mOhm_cm2 = df.get('rho_c_mOhm_cm2', pd.Series(dtype=float)).astype(float)
    J0_A_cm2 = df.get('J0_interface_A_cm2', pd.Series(dtype=float)).astype(float)
    x = rho_mOhm_cm2 * 0.001
    y = J0_A_cm2
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[valid]
    y = y[valid]
    if x.empty or y.empty:
        return None
    eV = 1.602176634e-19
    Gamma_min_eV = Gamma_min_meV * 0.001
    Gamma_tot_eV = Gamma_tot_meV * 0.001
    from .coupled import tradeoff_bound_point
    B = tradeoff_bound_point(T_K, Gamma_min_eV, Gamma_tot_eV)
    xmin, xmax = (np.nanmin(x), np.nanmax(x))
    xs = np.logspace(np.log10(xmin * 0.7), np.log10(xmax * 1.3), 400)
    ys = B / xs
    plt.figure(figsize=(6, 5))
    plt.loglog(x, y, 'o', alpha=0.8, label='Runs')
    plt.loglog(xs, ys, '-', label='No-go bound')
    plt.xlabel('$\\rho_c$ (Ω·cm$^2$)')
    plt.ylabel('$J_{0,\\mathrm{int}}$ (A/cm$^2$)')
    plt.title('Coupled trade-off: bound overlay')
    plt.legend()
    out_png = os.path.join(out_dir, 'pareto_rhoc_vs_J0int_with_bound.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()
    return out_png

def _combined_score(row, w_rho=0.5, w_dvoc=0.5):
    return w_rho * np.log10(max(row.get('rho_c_mOhm_cm2', np.nan), 1e-12)) + w_dvoc * (row.get('DeltaVoc_interface_mV', np.nan) / 100.0)

def make_report(analyzed_csv, out_dir='out', tempK=300.0, EF_eV=0.0, area_source='unknown', material_name=None, top_k=3):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(analyzed_csv)
    sidecar = os.path.join(out_dir, 'bound_overlay.json')
    bound_png = None
    if os.path.exists(sidecar):
        try:
            import json
            with open(sidecar, 'r') as f:
                cfg = json.load(f)
            bound_png = _add_bound_overlay_plot(df, out_dir, cfg['T_K'], cfg['Gamma_min_meV'], cfg['Gamma_tot_meV'])
        except Exception as e:
            bound_png = None
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = df.get('rho_c_mOhm_cm2', pd.Series(dtype=float))
    y = df.get('DeltaVoc_interface_mV', pd.Series(dtype=float))
    ax.scatter(x, y)
    ax.set_xlabel('ρc (mΩ·cm²)')
    ax.set_ylabel('ΔVoc_interface (mV)')
    fig.savefig(os.path.join(out_dir, 'pareto_rho_vs_dVoc.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    d = df.copy()
    if 'rho_c_mOhm_cm2' in d and 'DeltaVoc_interface_mV' in d:
        r = d['rho_c_mOhm_cm2'].astype(float)
        v = d['DeltaVoc_interface_mV'].astype(float)
        r_norm = (r - r.min()) / (r.max() - r.min() + 1e-12)
        v_norm = (v - v.min()) / (v.max() - v.min() + 1e-12)
        score = r_norm + v_norm
        d = d.assign(_score=score).sort_values('_score').head(top_k)
    table_md = d[['run_id', 'rho_c_mOhm_cm2', 'DeltaVoc_interface_mV', 'S_interface_cm_s', 'J0_interface_A_cm2', 't_ITO_nm', 'interlayer', 'surface_texture']].to_markdown(index=False) if len(d) > 0 else 'N/A'
    md_lines = ['# TRJ Optimization Report', '', f'Analyzed file: `{os.path.basename(analyzed_csv)}`', '', '## Parameters & Units', '', f'Temperature: {tempK} K  ', f'EF reference: {EF_eV} eV  ', f'Area source: {area_source}  ', f'Material: {material_name}  ' if material_name else '', '', '## Pareto: ρc vs ΔVoc', '', '![Pareto](pareto_rho_vs_dVoc.png)', '', '## Top Candidates (by combined score)', '', table_md, '', '## Robustness (local thickness sensitivity)', '', 'None', '', '## Coupled Trade-off Bound (overlay)', '', f'![]({os.path.basename(bound_png)})' if bound_png else '_No overlay generated (missing data or config)_.', '', 'None']
    with open(os.path.join(out_dir, 'report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    return os.path.join(out_dir, 'report.md')
import os, pandas as pd, numpy as np
import matplotlib.pyplot as plt
