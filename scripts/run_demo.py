#!/usr/bin/env python3
import subprocess, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
EX_T = ROOT / "examples" / "run_demo_T.csv"
EX_D = ROOT / "examples" / "run_demo_Dit.csv"
OUT = ROOT / "trj_opt_out"
OUT.mkdir(exist_ok=True)

def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    if not EX_T.exists() or not EX_D.exists():
        print("Missing example CSVs in examples/.", file=sys.stderr)
        sys.exit(1)

    # Try CLI help
    run([sys.executable, "-m", "trj_opt.cli", "--help"])

    # Contact resistivity
    run([sys.executable, "-m", "trj_opt.cli", "compute-rhoc",
         "--te_csv", str(EX_T), "--out_json", str(OUT / "rho_c.json")])

    # Interface recombination
    run([sys.executable, "-m", "trj_opt.cli", "compute-srh",
         "--dit_csv", str(EX_D), "--out_json", str(OUT / "srh.json")])

    # Pareto + report
    run([sys.executable, "-m", "trj_opt.cli", "plot-pareto",
         "--analyzed_csv", str(OUT / "doe_plan_analyzed.csv"), "--out_dir", str(OUT)])
    run([sys.executable, "-m", "trj_opt.cli", "report",
         "--analyzed_csv", str(OUT / "doe_plan_analyzed.csv"), "--out_dir", str(OUT)])

if __name__ == "__main__":
    main()
