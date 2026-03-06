# Repository Refactoring Guide

## Overview

This document describes the refactoring performed on the Paleo Climate ML repository to improve code organization, reduce redundancy, and enhance maintainability.

## Changes Summary

### What Changed

1. **Created `utils/` module** with reusable Python functions
2. **Consolidated 16 notebooks into 3 concise ones under `notebooks/`**
3. **Established consistent naming convention** (lowercase with underscores)
4. **Created single `notebooks/` folder** with numbered workflow

### What Stayed the Same


## New Structure

```
code/
├── utils/                          # NEW: Reusable modules
│   ├── __init__.py
│   ├── data_io.py
│   └── visualization.py
│
├── notebooks/                       # NEW: Consolidated notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_data_exploration.ipynb
│   ├── 03_training_data_setup.ipynb
│   └── 04_model_training.ipynb
│
├── Initial Data Exploration/        # KEPT: Original notebooks
│   └── ... (6 notebooks)
│
└── Training Data Setup & exploration/  # KEPT: Original notebooks
    └── ... (10 notebooks)
```

## Detailed Changes

### 1. Utils Module Creation

#### `utils/__init__.py`

#### `utils/data_io.py`
Functions extracted from multiple notebooks:
- `load_dataset()` - netCDF loading
- `select_variable()` - Variable selection (from Training_Data_Setup.ipynb)
- `save_dataset()` - save to netCDF file


#### `utils/visualization.py`
Functions extracted from multiple notebooks:
- `setup_map_axes()` - Cartopy setup pattern
- `plot_spatial_map()` - 2D map plotting (all exploration notebooks)
- `plot_time_series()` - Time series with smoothing (Initial_exploration.ipynb)
- `plot_zonal_mean()` - Hovmöller diagrams (rsdt_rsut_rlut.ipynb)

### 2. Notebook Consolidation

#### `01_data_preparation.ipynb` (NEW)
**Consolidates:**
- `rsut_concat.ipynb` - Concatenation logic
- `rlut_concat.ipynb` - Concatenation logic
- `Shortwave_CRE.ipynb` - CRE computation
- `Longwave_CRE.ipynb` - CRE computation
- `rsut_cre_Data_Setup.ipynb` - CRE setup
- `rlut_cre_Data_Setup.ipynb` - CRE setup
- `Input_Data_Setup.ipynb` - Input preparation

**Why consolidated:**
- All perform data preparation steps
- Sequential workflow: concat → compute CRE → save
- Reduces duplication between shortwave/longwave processing

#### `02_data_exploration.ipynb` (NEW)
**Consolidates:**
- `Initial_exploration.ipynb` - Temperature analysis
- `rsdt_rsut_rlut.ipynb` - Radiation budget
- `future_tas.ipynb` - Future projections
- `radiative_shortwave_downwelling.ipynb` - Radiation components
- `Reflected_shortwave_radiation_history.ipynb` - Historical radiation
- `Obervation_data_exploration.ipynb` - Observational data

#### `03_training_data_setup.ipynb` (NEW)
**Consolidates:**
- `Training_Data_Setup.ipynb` - Main splitting logic
- `Training_Data_Splitting.ipynb` - Additional splits
- `Input_Data_Setup.ipynb` - Data alignment

## Post-Refactor Conventions (Required for New Files)

- New notebooks must be created in `code/notebooks/`.
- Continue numbering sequentially (`04`, `05`, `06`, ...).
- Naming format: `NN_descriptive_name.ipynb` (`lowercase_with_underscores`).
- Reusable code belongs in `code/utils/`, not duplicated across notebooks.
