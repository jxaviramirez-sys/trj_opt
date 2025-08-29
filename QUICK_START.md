
# Quick Start — TRJ Optimization Toolkit (Patched)

## Inputs

### 1) Transport T(E)
Provide a CSV with either:
- **Two columns**: `E_eV,T_total` where `T_total` is the total transmission (dimensionless) integrated over transverse modes/k.
- **Multi-k**: columns `E_eV,kx,ky,weight,T`. The loader will BZ-average using `sum(weight*T)/sum(weight)`.

Also include **one** of the following cross-section metadata:
- Header line like `# A_cell_A2: 125.0` (unit-cell area in Å^2), or
- Header `# area_m2: 1.25e-14`, or
- Sidecar JSON file with the same keys: `run_X_T.csv.meta.json`.

> The ρc calculation uses the linear-response Landauer formula with the Fermi level at `# Ef_eV` (default 0).

### 2) Interface DOS Dit(E)
CSV with `E_eV,Dit_cm^-2eV^-1`. Optional header metadata allowed.

## Commands

```bash
# ρc from transport
python -m trj_opt.cli compute-rhoc --te_file path/to/run_1_T.csv --area_cm2 1e-4 --tempK 300

# Interface recombination metrics
python -m trj_opt.cli compute-interface --dit_file path/to/run_1_Dit.csv --NA 1e17 --ND 1e17 --J0_bulk 1e-15

# Batch + report
python -m trj_opt.cli analyze-batch --doe_csv examples/doe_plan.csv --te_dir examples --dos_dir examples --out_csv out/analyzed.csv
python -m trj_opt.cli make-report --analyzed_csv out/analyzed.csv --out_dir out/
```

## Assumptions & Physics Notes

- **ρc** uses `G/A = (2q^2/h) ∫ T(E)(-∂f/∂E) dE` with area normalization from metadata.
- **S_eff, J0_interface, ΔVoc_interface** computed via SRH integration over Dit(E). Requires doping and (optional) quasi-Fermi splitting `ΔqF` for the interface.
- **ΔVoc_interface** is computed against a user-provided `J0_bulk` (default 1e-15 A/cm²).

## Robustness

The code includes:
- Local thickness sensitivity (`robustness.local_sensitivity`)
- A generic Monte Carlo helper (`robustness.mc_robustness`) for uncertainty propagation.
