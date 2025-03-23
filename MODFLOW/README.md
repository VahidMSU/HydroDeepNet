# MODGenX: MODFLOW Model Generator

MODGenX is a Python package for automatically generating MODFLOW groundwater models from SWAT+ outputs. It provides a streamlined workflow for creating, running, and evaluating groundwater models with minimal manual intervention.

## Introduction

MODGenX connects surface water modeling (SWAT+) with groundwater modeling (MODFLOW) by using SWAT+ outputs as inputs for generating complete MODFLOW models. This integration allows for more comprehensive watershed modeling that accounts for both surface and subsurface water processes.

## Configuration Parameters

The MODGenX system uses several configuration parameters to control model behavior:

### Unit Conversion Factors

- `fit_to_meter`: Converts feet to meters (default: 0.3048)
- `recharge_conv_factor`: Converts recharge from inch/year to meter/day (default: 0.0254/365.25)

These factors are used to ensure consistent units throughout the modeling process.

The recharge data is expected to be provided in inches/year and will be automatically converted to meters/day during processing.

### Model Discretization Parameters

- `n_sublay_1`: Number of sublayers in the first main layer (default: 2)
- `n_sublay_2`: Number of sublayers in the second main layer (default: 3)
- `k_bedrock`: Bedrock hydraulic conductivity in m/day (default: 1e-4)
- `bedrock_thickness`: Bedrock thickness in meters (default: 40)

### Convergence Parameters

- `headtol`: Head change tolerance for convergence in meters (default: 0.01)
- `fluxtol`: Flux tolerance for convergence in m³/day (default: 0.001)
- `maxiterout`: Maximum number of outer iterations (default: 100)

## Installation

### Prerequisites

- Python 3.7+
- FloPy
- GDAL
- NumPy
- Pandas
- Matplotlib
- Geopandas
- Rasterio
- SciPy

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/modgenx.git
   cd modgenx
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure MODFLOW-NWT executable is available in the `/data/SWATGenXApp/codes/bin/` directory.

## Usage

### Basic Usage

```python
from MODGenX_API import create_modflow_model

model = create_modflow_model(
    username="user123",
    NAME="04112500",     # HUC ID or basin name
    VPUID="0405",        # Virtual Polygon Unit ID
    RESOLUTION=250,      # Resolution in meters
    ML=False             # Whether to use machine learning predictions
)
```

### Command Line Interface

```bash
python MODGenX_API.py --username user123 --name 04112500 --resolution 250
```

## Workflow

MODGenX follows a streamlined workflow to convert SWAT+ outputs to MODFLOW models:

1. Input Data
2. Create DEM Raster
3. Define Model Boundaries
4. Load Input Parameters
5. Generate River Network
6. Process Lakes
7. Create Wells
8. Build MODFLOW Model
9. Run MODFLOW
10. Visualize Results
11. Calculate Performance Metrics
12. Save Metrics to CSV

## File Structure
INPUT PROCESSING
---------------
Input Data
    │
    ▼
Create DEM Raster
    │
    ▼
Define Model Boundaries
    │
    ▼
Load Input Parameters
    │
    ├────────────┬────────────┐
    │            │            │
    ▼            ▼            ▼
FEATURE PROCESSING
----------------
Generate Rivers   Process Lakes   Create Wells
    │                │               │
    │                │               │
    └────────────────┼───────────────┘
                     │
                     ▼
MODEL CREATION & EXECUTION
------------------------
Build MODFLOW Model
          │
          ▼
    Run MODFLOW
          │
          ▼
RESULTS ANALYSIS
--------------
Visualize Results
          │
          ▼
Calculate Performance Metrics
          │
          ▼
 Save Metrics to CSV