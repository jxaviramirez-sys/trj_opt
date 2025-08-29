# Changelog (Reconstruction)

- Merged multiple partial versions into a single package using AST-guided union.
- Preserved unique modules present only in newer variants.
- Normalized imports and module docstrings where possible.
- Added `pyproject.toml` with `trj-opt` CLI entry point.
- Added `__version__ = "0.1.0-reconstructed"` in `trj_opt/__init__.py`.
- Smoke import passed.
- Notes:
[WARN] import trj_opt.coupled failed: name 'Level' is not defined
[WARN] import trj_opt.gui_app failed: No module named 'streamlit'

