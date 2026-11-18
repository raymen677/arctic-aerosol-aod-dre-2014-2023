"""Figure S4: MERRA-2 total and component AOD spatial biases.

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
from figs3_merra2_aod_scatter import plot_spatial, ds_gc, ds_m2

if __name__ == "__main__":
    try:
        plot_spatial("figs4_merra2_aod_spatial_bias.png")
    finally:
        ds_gc.close()
        ds_m2.close()
