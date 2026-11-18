"""Figure S6: all sky CERES radiation spatial evaluation.

Inputs are resolved relative to this script.
Outputs: matching PNG (1000 dpi) and PDF (190 mm wide) in figures/.
Shared exports and offline map helpers: fig1_region.py.
"""

import sys
sys.dont_write_bytecode = True
from figs5_ceres_radiation_scatter import render_ceres_spatial

if __name__ == "__main__":
    render_ceres_spatial(False)
