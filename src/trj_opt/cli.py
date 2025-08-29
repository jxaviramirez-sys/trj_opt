import argparse, os, pandas as pd, numpy as np
from . import landauer, srh
from .io import read_te_csv, read_dit_csv
from .analyze import analyze_batch
from .report import make_report

def main():
    p = argparse.ArgumentParser(prog='trj_opt', description='TRJ optimizer CLI: contact resistivity and interface recombination')
    sp = p.add_subparsers(dest='cmd', required=True)
    c1 = sp.add_parser('compute-rhoc', help='Compute specific contact resistivity from T(E)')
    c1.add_argument('--T', required=True, help='Transmission CSV (E_eV,T)')
    c1.add_argument('--T-K', type=float, default=300.0)
    c1.add_argument('--area-cm2', type=float, default=None)
    c1.add_argument('--EF-eV', type=float, default=None)
    c1.add_argument('--material', type=str, default=None)
    c1.add_argument('--te_file', required=True)
    c1.add_argument('--tempK', type=float, default=300.0)
    c1.add_argument('--area_cm2', type=float, default=None, help='Cross-sectional area (cm^2) if not supplied in metadata')
    c1.add_argument('--Ef_eV', type=float, default=0.0)
    c2 = sp.add_parser('compute-interface', help='Compute S_eff, J0_interface, ΔVoc from Dit(E)')
    c2.add_argument('--dit_file', required=True)
    c2.add_argument('--tempK', type=float, default=300.0)
    c2.add_argument('--NA', type=float, default=1e+17)
    c2.add_argument('--ND', type=float, default=1e+17)
    c2.add_argument('--J0_bulk', type=float, default=1e-15)
    c2.add_argument('--delta_qF_eV', type=float, default=0.0)
    c4 = sp.add_parser('coupled', help='Coupled transport–recombination from a single-level WBL model + bound')
    c4.add_argument('--tempK', type=float, default=300.0)
    c4.add_argument('--mu_eV', type=float, default=0.0)
    c4.add_argument('--epsilon_eV', type=float, default=0.0)
    for t in ['L', 'R', 'C', 'V']:
        c4.add_argument(f'--Gamma{t}_meV', type=float, required=True)
    c4.add_argument('--Gamma_min_meV', type=float, required=True, help='Minimum Γ for C and V (meV)')
    c4.add_argument('--J0_bulk', type=float, default=1e-12, help='Bulk dark saturation density [A/m^2]')
    c4.add_argument('--grid_meV', type=float, default=100.0, help='± energy window for integration [meV]')
    c4.add_argument('--points', type=int, default=20001)
    c3 = sp.add_parser('analyze-batch', help='Analyze a DOE CSV with per-run T(E) and Dit(E)')
    c3.add_argument('--doe_csv', required=True)
    c3.add_argument('--te_dir', required=True)
    c3.add_argument('--dos_dir', required=True)
    c3.add_argument('--out_csv', required=True)
    c3.add_argument('--tempK', type=float, default=300.0)
    c3.add_argument('--area_cm2', type=float, default=None)
    c3.add_argument('--NA', type=float, default=1e+17)
    c3.add_argument('--ND', type=float, default=1e+17)
    c3.add_argument('--J0_bulk', type=float, default=1e-15)
    c3.add_argument('--Ef_eV', type=float, default=0.0)
    c3.add_argument('--delta_qF_eV', type=float, default=0.0)
    c4 = sp.add_parser('make-report', help='Build a Markdown report with Pareto plot')
    c4.add_argument('--analyzed_csv', required=True)
    c4.add_argument('--out_dir', required=True)
    c5 = sp.add_parser('demo', help='Run on bundled toy example and emit report')
    c5.add_argument('--out_dir', required=True)
    args = p.parse_args()
    if args.cmd == 'compute-rhoc':
        E, T, meta = read_te_csv(args.te_file)
        area_cm2 = args.area_cm2
        if area_cm2 is None:
            if 'area_m2' in meta:
                area_cm2 = meta['area_m2'] * 10000.0
            elif 'A_cell_A2' in meta:
                area_cm2 = meta['A_cell_A2'] * 1e-16
        if area_cm2 is None:
            raise SystemExit('No area provided (via --area_cm2 or metadata A_cell_A2/area_m2)')
        rho_c = landauer.small_signal_rho_c(E, T, args.tempK, area_cm2=area_cm2, mu_eV=args.Ef_eV)
        print(f'rho_c = {rho_c:.6e} Ohm*cm^2')
    elif args.cmd == 'compute-interface':
        E, Dit, meta = read_dit_csv(args.dit_file)
        S, J0i, dVoc = srh.compute_from_Dit(E, Dit, T_K=args.tempK, N_A_cm3=args.NA, N_D_cm3=args.ND, J0_bulk_A_cm2=args.J0_bulk, delta_qF_eV=args.delta_qF_eV)
        print(f'S_eff = {S:.3e} cm/s\nJ0_interface = {J0i:.3e} A/cm^2\nDeltaVoc_interface = {dVoc * 1000.0:.2f} mV')
    elif args.cmd == 'analyze-batch':
        analyze_batch(args.doe_csv, args.te_dir, args.dos_dir, args.out_csv, tempK=args.tempK, area_cm2=args.area_cm2, NA=args.NA, ND=args.ND, J0_bulk=args.J0_bulk, EF_eV=args.Ef_eV, delta_qF_eV=args.delta_qF_eV)
        print(args.out_csv)
    elif args.cmd == 'make-report':
        md, png = make_report(args.analyzed_csv, args.out_dir)
        print(md)
        print(png)
    elif args.cmd == 'demo':
        root = os.path.dirname(os.path.dirname(__file__))
        ex = os.path.join(root, 'examples')
        doe = os.path.join(ex, 'doe_plan.csv')
        te_dir = ex
        dos_dir = ex
        os.makedirs(args.out_dir, exist_ok=True)
        out_csv = os.path.join(args.out_dir, 'doe_plan_analyzed.csv')
        analyze_batch(doe, te_dir, dos_dir, out_csv)
        make_report(out_csv, args.out_dir)
        print(f'Demo outputs -> {args.out_dir}')
    args = p.parse_args()
    if args.cmd == 'compute-rhoc':
        from .io import read_te_csv
        from .landauer import specific_contact_resistivity
        E, T, meta = read_te_csv(args.T)
        area_m2 = meta.get('area_m2') or meta.get('A_cell_A2', 0.0) * 1e-20
        if args.area_cm2 is not None:
            area_m2 = args.area_cm2 * 0.0001
        if not area_m2:
            raise SystemExit('Missing cross-section: provide # area_m2, # A_cell_A2, or --area-cm2')
        EF = meta.get('Ef_eV', 0.0) if args.EF_eV is None else args.EF_eV
        rho = specific_contact_resistivity(E, T, args.T_K, area_m2, mu_eV=EF)
        print(f'rho_c = {rho:.6e} Ohm*cm^2')
    elif args.cmd == 'compute-interface':
        from .io import read_dit_csv
        from .srh import compute_from_Dit, resolve_material
        E, Dit, meta = read_dit_csv(args.Dit)
        mat = resolve_material(args.material, {'sigma_n_cm2': args.sigma_n_cm2, 'sigma_p_cm2': args.sigma_p_cm2, 'v_th_cm_s_at_300K': args.v_th_cm_s})
        S, J0, dVoc = compute_from_Dit(E, Dit, T_K=args.T_K, N_A_cm3=args.NA, N_D_cm3=args.ND, J0_bulk_A_cm2=args.J0_bulk, delta_qF_eV=args.delta_qF_eV, material=mat)
        print(f'S_interface = {S:.3f} cm/s')
        print(f'J0_interface = {J0:.6e} A/cm^2')
        print(f'DeltaVoc_interface = {1000.0 * dVoc:.3f} mV')
    elif args.cmd == 'analyze-batch':
        from .analyze import analyze_batch
        mat_over = {'sigma_n_cm2': args.sigma_n_cm2, 'sigma_p_cm2': args.sigma_p_cm2, 'v_th_cm_s_at_300K': args.v_th_cm_s}
        analyze_batch(args.doe, args.te_dir, args.dos_dir, args.out, tempK=args.T_K, area_cm2=args.area_cm2, NA=args.NA, ND=args.ND, J0_bulk=args.J0_bulk, EF_eV=args.EF_eV, delta_qF_eV=args.delta_qF_eV, material_name=args.material, material_overrides=mat_over)
        print(f'Wrote {args.out}')
    elif args.cmd == 'demo':
        import os
        from .analyze import analyze_batch
        from .report import make_report
        ex_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
        out_csv = os.path.join(args.out_dir, 'analyzed.csv')
        os.makedirs(args.out_dir, exist_ok=True)
        analyze_batch(os.path.join(ex_dir, 'doe_plan.csv'), ex_dir, ex_dir, out_csv, tempK=300.0, area_cm2=args.area_cm2, material_name=args.material)
        make_report(out_csv, out_dir=args.out_dir, tempK=300.0, EF_eV=0.0, area_source='flag', material_name=args.material)
        print('Demo complete. See:', args.out_dir)
    else:
        p.print_help()
    if args.cmd == 'coupled':
        from .coupled import Level, InterfaceSystem, gLR_from_system, jCV_from_system, rho_c_from_gLR, J0_int_from_jCV, delta_Voc, tradeoff_bound_point
        eV = 1.0
        meV_to_eV = 0.001
        GL = args.GammaL_meV * meV_to_eV
        GR = args.GammaR_meV * meV_to_eV
        GC = args.GammaC_meV * meV_to_eV
        GV = args.GammaV_meV * meV_to_eV
        lvl = Level(epsilon_eV=args.epsilon_eV, GammaL_eV=GL, GammaR_eV=GR, GammaC_eV=GC, GammaV_eV=GV)
        sysm = InterfaceSystem([lvl])
        W = args.grid_meV * meV_to_eV
        import numpy as np, json
        E = np.linspace(args.mu_eV - W, args.mu_eV + W, args.points)
        gLR = gLR_from_system(sysm, E, args.tempK, mu_eV=args.mu_eV)
        jCV = jCV_from_system(sysm, E, args.tempK, mu_eV=args.mu_eV)
        rho_c_SI = rho_c_from_gLR(gLR)
        J0_int = J0_int_from_jCV(jCV, args.tempK)
        dVoc = delta_Voc(J0_int, args.J0_bulk, args.tempK)
        Gmin = args.Gamma_min_meV * meV_to_eV
        Gtot = GL + GR + GC + GV
        boundV = tradeoff_bound_point(Gmin, Gtot, args.tempK)
        out = {'gLR_S_per_m2': float(gLR), 'rho_c_Ohm_m2': float(rho_c_SI), 'rho_c_Ohm_cm2': float(rho_c_SI * 10000.0), 'jCV_S_per_m2': float(jCV), 'J0_int_A_per_m2': float(J0_int), 'Delta_Voc_V': float(dVoc), 'bound_rho_c_times_J0int_min_V': float(boundV), 'product_rho_c_times_J0int_V': float(rho_c_SI * J0_int)}
        print(json.dumps(out, indent=2))
        return
import argparse, os, pandas as pd, numpy as np
from . import landauer, srh
from .io import read_te_csv, read_dit_csv
from .analyze import analyze_batch
from .report import make_report
