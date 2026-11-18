# Pan-Arctic aerosol AOD and direct radiative effects (2014–2023)

Data and Python scripts for reproducing Figures 1–5 and Supplementary Figures S1–S34.

## Files and downloads

- **39 figure scripts:** 5 in `codes/main/` and 34 in `codes/supplementary/`; filenames identify their figure number and content.
- **25 NetCDF inputs:** download from [Release v1.0.0](https://github.com/raymen677/arctic-aerosol-aod-dre-2014-2023/releases/tag/v1.0.0). [data_manifest.json](data_manifest.json) lists their required paths and SHA-256 checksums.
- **Full resolution figures:** download [figures_png_pdf_v1.0.0.zip](https://github.com/raymen677/arctic-aerosol-aod-dre-2014-2023/releases/download/v1.0.0/figures_png_pdf_v1.0.0.zip), containing 39 PNG/PDF pairs. See [figures_manifest.json](figures_manifest.json) for checksums.

The repository contains the code; data and complete figure exports are stored as Release attachments. Git LFS is not required.

## Setup

Tested with Python 3.13.11. Install the dependencies:

```bash
python -m pip install numpy pandas matplotlib cartopy xarray netCDF4 scipy Pillow shapely pypdf
```

Skip this step if data/ is already present. Otherwise, run this Python snippet from the repository root to download the inputs:

```python
from pathlib import Path
import json
import urllib.request

for item in json.loads(Path("data_manifest.json").read_text())["files"]:
    target = Path(item["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading:", target)
    urllib.request.urlretrieve(item["download_url"], target)
```

## Run

```bash
python -B codes/main/fig1_region.py
python -B codes/main/fig2_arctic_aod.py
python -B codes/supplementary/figs8_nitrate_aod.py
```

Keep the `codes/` directory structure because some scripts share functions. Paths are resolved from the repository root: inputs are read from `data/`, and outputs are written to `figures/main/` and `figures/supplementary/`.

PNG exports use **1000 dpi** and are at least **7480 pixels wide**. PDF exports are **190 mm wide**, with embedded fonts.

This archive contains processed inputs and plotting code. It does not include a complete GEOS-Chem simulation setup. The figure label NI retains the original AM + NI definition.
