"""Figure S4: MERRA-2 total and component AOD spatial biases.

Inputs are resolved relative to this script.
Outputs: matching PNG (1000 dpi) and PDF (190 mm wide) in figures/.
Shared exports and offline map helpers: fig1_region.py.
"""

import sys
sys.dont_write_bytecode = True
from figs3_merra2_aod_scatter import plot_spatial, ds_gc, ds_m2

if __name__ == "__main__":
    try:
        plot_spatial("figs4_merra2_aod_spatial_bias.png")
    finally:
        ds_gc.close()
        ds_m2.close()
