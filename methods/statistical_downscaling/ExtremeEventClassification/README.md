
# Extreme Event Classification

Framework for extreme event type classification from
> Pupić Vurilj, M. et al. *Storm surge hydrographs from historical observations of sea level along the Dutch North Sea coast.* Nat Hazards (2025). [https://doi.org/10.1007/s11069-025-07351-8](https://doi.org/10.1007/s11069-025-07351-8)

The framework consists of four notebooks:

#### 1. GESLA extract variables
Notebook to extract all necessary variables from the [GESLA-3 dataset](https://gesla787883612.wordpress.com/).

#### 2. Extremes
Notebook to extract extreme events using Peak Over Threshold (POT).

#### 3. Event types
Notebook to normalise storm surge hydrographs and perform clustering.

#### 4. Characterisation
Notebook to characterise event types.

## Installation

The notebook 01_GESLA_extract_variables.ipynb requires the [Utide package](https://github.com/wesleybowman/UTide). UTide depends on a specific version of SciPy to function properly.

```
  pip uninstall scipy
  pip install scipy==1.11.4 
  pip install utide
```

## Authors

- [@pupicvuriljm](https://github.com/pupicvuriljm)

