"""Figure S21: black carbon surface shortwave dre.

Inputs are resolved relative to this script.
Outputs: matching PNG (1000 dpi) and PDF (190 mm wide) in figures/.
Shared exports and offline map helpers: fig1_region.py.
"""

import sys
sys.dont_write_bytecode = True
from fig4_surface_shortwave_dre import render_radiative_figure

if __name__ == "__main__":
    render_radiative_figure('BC', 0, 'figs21_black_carbon_surface_shortwave_dre')
