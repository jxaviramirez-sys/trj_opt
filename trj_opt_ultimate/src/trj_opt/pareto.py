# Reconstructed by merge of 7 variants for trj_opt/pareto.py
# Strategy: union of imports, union of classes/functions with longest/most-params preference.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pareto_front(df, xcol, ycol):
    data = df[[xcol, ycol, 'run_id']].dropna().copy()
    data = data.sort_values([xcol, ycol], ascending=[True, True])
    pareto = []
    best_y = np.inf
    for _, row in data.iterrows():
        if row[ycol] <= best_y:
            pareto.append(row)
            best_y = row[ycol]
    return pd.DataFrame(pareto)

def plot_pareto(df, xcol, ycol, out_png, title='Pareto Frontier'):
    plt.figure(figsize=(6, 5))
    plt.scatter(df[xcol], df[ycol])
    pf = pareto_front(df, xcol, ycol)
    if not pf.empty:
        plt.plot(pf[xcol], pf[ycol], marker='o')
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close()

def plot_pareto_with_overlay(df, xcol, ycol, out_png, title='Pareto Frontier', measured_df=None, measured_label_col='label'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    plt.scatter(df[xcol], df[ycol], label='Model grid', alpha=0.7)
    pf = pareto_front(df, xcol, ycol)
    if not pf.empty:
        plt.plot(pf[xcol], pf[ycol], marker='o', label='Pareto (model)')
    if measured_df is not None and len(measured_df) > 0:
        if measured_label_col in measured_df.columns:
            for _, r in measured_df.iterrows():
                plt.scatter([r[xcol]], [r[ycol]], marker='x', s=60)
                try:
                    lbl = str(r[measured_label_col])
                except Exception:
                    lbl = None
                if lbl:
                    plt.annotate(lbl, (r[xcol], r[ycol]), xytext=(5, 5), textcoords='offset points')
        else:
            plt.scatter(measured_df[xcol], measured_df[ycol], marker='x', s=60, label='Measured')
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
