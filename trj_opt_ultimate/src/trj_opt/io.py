# Reconstructed by merge of 7 variants for trj_opt/io.py
# Strategy: union of imports, union of classes/functions with longest/most-params preference.

import pandas as pd
import json
import re

def _parse_metadata_from_header(path):
    meta = {}
    header_lines = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header_lines.append(line[1:].strip())
            else:
                break
    for ln in header_lines:
        m = re.match('\\s*meta\\s*:\\s*(\\{.*\\})\\s*$', ln)
        if m:
            try:
                meta.update(json.loads(m.group(1)))
                continue
            except Exception:
                pass
        if ':' in ln:
            k, v = ln.split(':', 1)
            meta[k.strip()] = v.strip()
    for k in list(meta.keys()):
        val = meta[k]
        if isinstance(val, str):
            try:
                if val.lower().endswith('eV'.lower()):
                    meta[k] = float(val.split()[0])
                elif val.lower().endswith('A^2') or val.lower().endswith('Å^2') or 'A2' in val:
                    meta[k] = float(re.findall('[-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?', val)[0])
                else:
                    meta[k] = float(val)
            except Exception:
                pass
    return meta

def read_dit_csv(path):
    """Read interface DOS CSV.

    Expected columns: E_eV, Dit_cm^-2eV^-1 (or Dit)
    Optional header metadata accepted like read_te_csv.
    Returns: E_eV, Dit, meta (dict)
    """
    meta = _parse_metadata_from_header(path)
    try:
        with open(path + '.meta.json', 'r') as f:
            import json as _json
            side = _json.load(f)
            meta.update(side)
    except Exception:
        pass
    df = pd.read_csv(path, comment='#')
    cols = [c.strip() for c in df.columns]
    df.columns = cols
    E = df.iloc[:, 0].values
    Dit = df.iloc[:, 1].values
    return (E, Dit, meta)

def read_te_csv(path):
    """Read a transmission CSV.

    Supported formats:
    1) Two columns: E_eV, T_total
    2) Multi-k: columns include at least ['E_eV','kx','ky','weight','T'] and will be BZ-averaged using 'weight'
    Optional metadata can be supplied in header lines starting with '#', or via a sidecar JSON at path+'.meta.json'.
    Returns: E_eV (1D array), T_total (1D array), meta (dict)
    """
    meta = _parse_metadata_from_header(path)
    try:
        with open(path + '.meta.json', 'r') as f:
            import json as _json
            side = _json.load(f)
            meta.update(side)
    except Exception:
        pass
    df = pd.read_csv(path, comment='#')
    cols = [c.strip() for c in df.columns]
    df.columns = cols
    if len(cols) >= 2 and cols[0].lower().startswith('e'):
        if set(['kx', 'ky', 'weight', 'T']).issubset(set(cols)):
            grouped = df.groupby(df.columns[0])
            E = []
            Ttot = []
            for e, sub in grouped:
                wT = (sub['weight'] * sub['T']).sum()
                wsum = sub['weight'].sum()
                Ttot.append(wT / max(wsum, 1e-30))
                E.append(e)
            import numpy as np
            E = np.array(E, float)
            Ttot = np.array(Ttot, float)
            meta.setdefault('T_is_BZ_avg', True)
            return (E, Ttot, meta)
        else:
            E = df.iloc[:, 0].values
            T = df.iloc[:, 1].values
            return (E, T, meta)
    else:
        raise ValueError(f'Unsupported T(E) file format for {path}')
import pandas as pd
import json
import re
