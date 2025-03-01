# Climate Change Analysis with LOCA2 Data

This README provides guidance on how to use the climate change analysis tools to analyze future climate projections using LOCA2 data.

## Overview

The climate change analysis tools allow you to:

1. Extract and process historical and future scenario climate data from LOCA2 datasets
2. Calculate climate change metrics between historical and future periods
3. Generate visualizations showing projected changes in temperature and precipitation
4. Create comprehensive reports documenting climate change impacts

## Important Note About LOCA2 Data Structure

The LOCA2 dataset divides future scenario data into three specific time periods:
- 2015-2044 (near-term)
- 2045-2074 (mid-century)
- 2075-2100 (long-term)

When selecting years for analysis, try to stay within a single time period for best results.

## Quick Start

Run a simple analysis using the command-line interface:

```bash
python AI_agent/AI_agent/run_climate_analysis.py \
  --hist-start 2000 --hist-end 2014 \
  --fut-start 2020 --fut-end 2030 \
  --model ACCESS-CM2 \
  --scenario ssp245 \
  --output-dir my_climate_results
```

## Checking LOCA2 Dataset Structure

If you encounter issues accessing the data, you can check the structure of the LOCA2 dataset:

```bash
python AI_agent/AI_agent/check_loca2_paths.py
```

This will show the available models, scenarios, and time periods in the dataset.

## Example Usage in Python

```python
from AI_agent.AI_agent.climate_change_analysis import ClimateChangeAnalysis

# Configuration
config = {
    "aggregation": "monthly",
    "bounding_box": [-85.444332, 43.658148, -85.239256, 44.164683],
    "output_dir": "climate_results",
    "use_synthetic_data_fallback": True
}

# Create analysis object
analysis = ClimateChangeAnalysis(config)

# Historical configuration
historical_config = {
    'start_year': 2000,
    'end_year': 2014,
    'model': 'ACCESS-CM2',
    'ensemble': 'r1i1p1f1'
}

# Future scenario configuration
scenario_configs = [
    {
        'name': 'ssp245',
        'start_year': 2020,
        'end_year': 2030,
        'model': 'ACCESS-CM2',
        'ensemble': 'r1i1p1f1'
    }
]

# Run analysis
analysis.extract_data(historical_config, scenario_configs)
metrics = analysis.calculate_climate_change_metrics()
analysis.plot_timeseries_comparison()
analysis.plot_spatial_change_maps(variable='tmean')
report_path = analysis.generate_climate_change_report()
```

## Key Commands and Parameters

### Historical Period Configuration

- `hist-start`: Start year for historical period (default: 2000)
- `hist-end`: End year for historical period (default: 2014)

### Future Period Configuration

- `fut-start`: Start year for future period (default: 2020)
- `fut-end`: End year for future period (default: 2030)
- `scenario`: Future scenario (default: ssp245, options include ssp126, ssp245, ssp370, ssp585)

### Model Configuration

- `model`: Climate model to use (default: ACCESS-CM2)
- `ensemble`: Ensemble member (default: r1i1p1f1)

### Output Configuration

- `output-dir`: Directory to store analysis results (default: climate_analysis_results)
- `agg`: Temporal aggregation type (default: monthly, options: daily, monthly, seasonal, annual)

### Geographic Extent

- `bbox`: Bounding box as min_lon,min_lat,max_lon,max_lat (default Central Michigan area)

## Handling Data Availability Issues

If LOCA2 data is not available for the specified scenario, model, or time period, you have these options:

1. Use the `--synthetic` flag to generate synthetic future data for demonstration purposes
2. Choose a different model or ensemble that has data for your scenario
3. Adjust the future years to fall within one of the available time periods

## Troubleshooting

If you encounter errors accessing LOCA2 data:

1. Check that the years you selected fall within a single time period (2015-2044, 2045-2074, or 2075-2100)
2. Verify the model and ensemble combination has data for your scenario using the check_loca2_paths.py script
3. Try a different ensemble member (e.g., r1i1p1f1, r2i1p1f1)
4. Ensure the configuration paths in AgentConfig point to the correct locations
