"""Figure S31: sulfate surface longwave dre.

Inputs are resolved relative to this script.
Outputs: matching PNG (1000 dpi) and PDF (190 mm wide) in figures/.
Shared exports and offline map helpers: fig1_region.py.
"""

import sys
sys.dont_write_bytecode = True
from fig4_surface_shortwave_dre import render_radiative_figure

if __name__ == "__main__":
    render_radiative_figure('SU', 2, 'figs31_sulfate_surface_longwave_dre')
