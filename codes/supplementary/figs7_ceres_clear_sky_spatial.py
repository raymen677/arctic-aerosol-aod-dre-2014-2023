"""Figure S7: clear sky CERES radiation spatial evaluation.

Inputs are resolved relative to this script.
Outputs: matching PNG (1000 dpi) and PDF (190 mm wide) in figures/.
Shared exports and offline map helpers: codes/main/fig1_region.py.
"""

import sys
sys.dont_write_bytecode = True

# Resolve shared figure modules from either code subdirectory.
from pathlib import Path as _EPPath
_ep_root = _EPPath(__file__).resolve().parents[2]
for _ep_code_dir in (_ep_root / "codes" / "main", _ep_root / "codes" / "supplementary"):
    if str(_ep_code_dir) not in sys.path:
        sys.path.insert(0, str(_ep_code_dir))
from figs5_ceres_radiation_scatter import render_ceres_spatial

if __name__ == "__main__":
    render_ceres_spatial(True)
