"""Figure S11: sulfate AOD spatial and temporal patterns.

Inputs are resolved relative to this script.
Outputs: matching PNG (1000 dpi) and PDF (190 mm wide) in figures/.
Shared exports and offline map helpers: fig1_region.py.
"""

import sys
sys.dont_write_bytecode = True
from fig2_arctic_aod import render_aod

if __name__ == "__main__":
    render_aod('SU')
