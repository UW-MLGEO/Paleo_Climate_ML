# Paleo Climate ML | ESS 469/569

**Team:** Manali, Allti, Justin, Filip, David.

## Research Question

Can we predict **Cloud Radiative Effect (CRE)** at the top of the atmosphere (TOA) using machine learning models trained on **surface temperature** data?

## Key Objectives

1. Develop ML models to predict TOA Cloud Radiative Effects from surface temperature
2. Compare model performance across architectures
3. Identify efficient dimensionality reduction techniques
4. Enable global trend analysis with minimal computational overhead

## Repository Structure

```
Paleo_Climate_ML/
├── code/
│   ├── utils/              # Reusable Python modules (NEW!)
│   │   ├── data_io.py
│   │   ├── data_processing.py
│   │   └── visualization.py
│   │
│   ├── notebooks/          # Consolidated analysis notebooks (NEW!)
│   │   ├── 01_data_preparation.ipynb
│   │   ├── 02_data_exploration.ipynb
│   │   └── 03_training_data_setup.ipynb
│   │
│   └── [Original notebooks preserved in subdirectories]
│
├── data/                   # Climate model output
│   ├── CanESM5_*.nc       # Historical & SSP data
│   └── splits/            # Train/val/test splits
│
└── outputs/               # Model outputs and results
```

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install xarray[complete] numpy matplotlib cartopy dask rioxarray

# Navigate to notebooks
cd code/notebooks
```

### 2. Run Analysis Workflow

Execute notebooks in order:

```bash
jupyter notebook 01_data_preparation.ipynb    # Prepare and merge data
jupyter notebook 02_data_exploration.ipynb    # Explore patterns
jupyter notebook 03_training_data_setup.ipynb # Create train/val/test splits
```

### 3. Use Utility Functions

```python
# In any notebook
import sys
sys.path.append('..')

from utils import load_dataset, plot_spatial_map, compute_cloud_radiative_effect

# Load data
data = load_dataset("../../data/CanESM5_historical_tas.nc")

# Visualize
plot_spatial_map(data["tas"], title="Surface Temperature")
```

## Recent Refactoring (Feb 2026)

The codebase has been refactored to improve maintainability:

- ✅ **16 notebooks consolidated into 3** focused workflows
- ✅ **Reusable `utils` module** for common operations
- ✅ **Consistent naming convention** (lowercase_with_underscores)
- ✅ **Clear numbered workflow** (01 → 02 → 03)
- ✅ **Original notebooks preserved** for reference

See [REFACTORING_GUIDE.md](REFACTORING_GUIDE.md) for details.

**Migration:** Optional - old notebooks still work!

## Data Files

### Input Data
- `CanESM5_historical_tas.nc` — Surface temperature (1850-2014)
- `CanESM5_ssp_tas.nc` — Surface temperature projections (2015-2100)
- `CanESM5_hist_rsdt.nc` — TOA incoming shortwave radiation
- `CanESM5_hist_rsut.nc` — TOA outgoing shortwave radiation (all-sky)
- `CanESM5_hist_rlut.nc` — TOA outgoing longwave radiation (all-sky)
- `CanESM_1850-2100_rsutcs.nc` — TOA shortwave (clear-sky)
- `CanESM5_1850-2100_rlutcs.nc` — TOA longwave (clear-sky)

### Processed Data (created by notebooks)
- `CanESM5_1850-2100_tas.nc` — Merged surface temperature
- `CanESM5_1850-2100_rsutcre.nc` — Shortwave Cloud Radiative Effect
- `CanESM5_1850-2100_rlutcre.nc` — Longwave Cloud Radiative Effect

## Key Concepts

### Cloud Radiative Effect (CRE)

**Shortwave CRE** = All-sky reflected solar - Clear-sky reflected solar
- Positive → Clouds reflect more solar radiation (cooling)

**Longwave CRE** = Clear-sky thermal - All-sky thermal
- Positive → Clouds trap more thermal radiation (warming)

### Training Strategy

- **Input (X):** Surface temperature (tas)
- **Target (y):** Cloud Radiative Effects (SWCRE, LWCRE)
- **Time periods:** 
  - Train: 1850-2014 (historical)
  - Validation: 1850-2014 (different ensemble members)
  - Test: 2015-2100 (future projections)
- **Ensemble members:** 25 total (17 train / 4 val / 4 test)

## Documentation

- [code/README.md](code/README.md) - Detailed code organization
- [REFACTORING_GUIDE.md](REFACTORING_GUIDE.md) - Migration guide
- Function docstrings in `utils/` modules

## Contributing

When adding new functionality:

1. **Reusable code** → Add to `code/utils/`
2. **Analysis workflows** → Create numbered notebook in `code/notebooks/`
3. **Follow naming conventions** → lowercase_with_underscores
4. **Document functions** → Use numpy-style docstrings
