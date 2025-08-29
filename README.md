# trj_opt — Perovskite–Si Interface Optimizer

**Purpose:** A theory-led toolkit to optimize the perovskite/Si recombination junction (TRJ) for tandem solar cells.
It ingests NEGF **transmission** (`T(E)`) and **interface DOS** to compute the **specific contact resistivity** (ρc),
**interface recombination metrics** (S_eff, J0_interface), and **ΔVoc_interface**, then ranks candidates, checks robustness,
and generates a report with Pareto plots. Use it to convert quantum-transport & DFT outputs into a **fabrication-ready recipe**.

## Quickstart (local)

```bash
python -m pip install -r requirements.txt
python -m trj_opt.cli demo  # writes outputs to ./trj_opt_out
```

Or run individual steps:

```bash
python -m trj_opt.cli analyze-batch   --doe_csv examples/doe_plan.csv   --te_dir examples   --dos_dir examples   --out_dir trj_opt_out

python -m trj_opt.cli plot-pareto --analyzed_csv trj_opt_out/doe_plan_analyzed.csv --out_dir trj_opt_out
python -m trj_opt.cli report --analyzed_csv trj_opt_out/doe_plan_analyzed.csv --out_dir trj_opt_out
```

### File formats
- **T(E)** CSV: two columns `E_eV,T` with energy in eV and transmission (dimensionless).
- **Interface DOS** CSV: two columns `E_eV,Dit_cm2eV` (states per cm² per eV).
- **DOE CSV**: includes per-run parameters (`run_id`, `t_ITO_nm`, etc.). The analyzer fills metrics columns.

### Subcommands
- `compute-rhoc`  — Landauer from T(E) → ρc
- `compute-srh`   — DOS → S_eff, J0_interface, ΔVoc_interface
- `analyze-batch` — Fill DOE with ρc/ΔVoc/J0 from per-run files
- `plot-pareto`   — ρc vs ΔVoc plot (PNG)
- `report`        — Markdown report with top candidates & robustness
- `demo`          — Runs a mini pipeline on included example files

## License
AGPL-3.0-only. For commercial use, please contact jxaviramirez@gmail.com.



## Updates
- Landauer linear-response ρc per area with metadata-aware normalization (A_cell_A2 / area_m2).
- SRH-based J0_interface and ΔVoc computation from Dit(E) with clear assumptions.
- Working CLI subcommands: compute-rhoc, compute-interface, analyze-batch, make-report, demo.
- Robustness utilities: local thickness sensitivity; Monte Carlo helper.
- Markdown report with Pareto plot and top candidates.
- Basic pytest unit tests.


## New: Coupled transport–recombination mode (single-level WBL)

This release adds a **coupled** subcommand implementing the shared-Hamiltonian model with a rigorous trade‑off bound:

```
python -m trj_opt.cli coupled --tempK 300 --mu_eV 0 --epsilon_eV 0   --GammaL_meV 20 --GammaR_meV 20 --GammaC_meV 5 --GammaV_meV 5   --Gamma_min_meV 5 --J0_bulk 1e-12
```

Outputs JSON with ρ_c, J0_int, ΔV_oc and the **lower bound** on ρ_c·J0_int:
\[
\rho_c J_{0,\mathrm{int}} \ge (kT/q)\, \frac{4\Gamma_\min^2}{(\Gamma-2\Gamma_\min)^2}.
\]
