"""Figure S34: Natural sea-salt surface and TOA shortwave and longwave DRE.

Inputs are resolved relative to this script.
Outputs: matching PNG (1000 dpi) and PDF (190 mm wide) in figures/.
Shared exports and offline map helpers: fig1_region.py.
"""

import sys
sys.dont_write_bytecode = True
from figs33_natural_dust_dre import render_natural_dre

if __name__ == "__main__":
    render_natural_dre("SS", "figs34_natural_sea_salt_dre")
