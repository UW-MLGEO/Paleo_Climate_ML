# Paleo Climate ML | ESS 469/569

**Team:** Manali, Allti, Justin, Filip, David.

## Research Question

Can we predict **net radiation at the top of the atmosphere (TOA)** using machine learning models trained on **sea surface temperature (SST)** data?


## Key Objectives

1. Develop ML models to predict TOA radiation from SST
2. Compare model performance across architectures
3. Identify efficient dimensionality reduction techniques
4. Enable global trend analysis with minimal computational overhead

## Data

- `CanESM5_hist_rsdt.nc` — Shortwave radiation downwelling
- `CanESM5_hist_rsut.nc` — Shortwave radiation upwelling  
- `CanESM5_hist_rlut.nc` — Longwave radiation upwelling
- `CanESM5_historical_tas.nc` — Surface temperature

## Notebooks

- `Initial_exploration.ipynb` — Data exploration
- `rsdt_rsut_rlut.ipynb` — TOA radiation analysis
- `Reflected_shortwave_radiation_history.ipynb` — Shortwave trends
- `Radiative_shortwave_downwelling.ipynb` — Downwelling radiation analysis
