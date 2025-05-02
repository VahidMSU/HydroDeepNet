# Environmental Data Report Generator

## Overview

The Environmental Data Report Generator is a comprehensive tool designed to analyze and generate reports from various environmental and hydrological data sources. It processes data for a specified geographic region (bounding box) and time period to produce detailed markdown and HTML reports with visualizations.

## Features

- **Multi-source Reports**: Generate reports from various data sources:
  - PRISM climate data (temperature, precipitation)
  - MODIS satellite data (vegetation indices, land cover)
  - Cropland Data Layer (CDL) for agricultural analysis
  - Groundwater data
  - NSRDB solar radiation data
  - gSSURGO soil data
  - SNODAS snow data
  - USGS governmental units
  - Climate change projections (LOCA2)

- **Comprehensive Analysis**: Each report includes:
  - Statistical analysis
  - Spatial visualizations
  - Temporal trends
  - Seasonal patterns
  - Recommendations based on data findings

- **Flexible Output**: Reports are generated in both Markdown and HTML formats with:
  - Interactive visualizations
  - Data tables
  - CSV exports of processed data
  - An HTML index page linking all reports

## Installation Requirements

The report generator depends on several Python libraries:
- numpy
- pandas
- matplotlib
- xarray (for some data sources)
- gdal (for spatial data processing)
- markdown (for HTML conversion)

Install dependencies with:
```bash
pip install numpy pandas matplotlib xarray gdal markdown
```

## Usage

### Basic Usage

```bash
python GeoReporter/report_generator.py --type all --output reports --min-lon -85.444332 --min-lat 43.158148 --max-lon -84.239256 --max-lat 44.164683
```

This command generates reports for all data sources for the specified bounding box.

### Common Parameters

- `--type`: Report type to generate (`prism`, `modis`, `cdl`, `groundwater`, `gov_units`, `gssurgo`, `climate_change`, `nsrdb`, `snodas`, or `all`)
- `--output`: Directory to save report files (default: `reports`)
- `--start-year`: Start year for analysis (default: 2010)
- `--end-year`: End year for analysis (default: 2020)
- `--min-lon`, `--min-lat`, `--max-lon`, `--max-lat`: Bounding box coordinates
- `--resolution`: Resolution for data analysis in meters (default: 250)
- `--aggregation`: Temporal aggregation (`daily`, `monthly`, `seasonal`, `annual`, default: `monthly`)
- `--sequential`: Run report generation sequentially (disable parallel processing)
- `--use-synthetic`: Use synthetic data if actual data is not available

### Report-Specific Parameters

#### PRISM
- `--prism-dataset`: PRISM dataset type (daily, monthly, etc.)
- `--prism-variables`: Comma-separated list of PRISM variables to include
- `--prism-fill-gaps`: Fill gaps in PRISM data

#### NSRDB
- `--nsrdb-extract-for-swat`: Extract NSRDB data in SWAT format
- `--nsrdb-variables`: Comma-separated list of NSRDB variables to include

#### MODIS
- `--modis-product`: MODIS product to use (MOD13Q1, etc.)
- `--modis-indices`: Comma-separated list of MODIS indices to include
- `--modis-backend`: Backend to use for processing (`gdal` or `xarray`)

#### CDL
- `--cdl-recode-crops`: Recode crops into major categories
- `--cdl-top-n-crops`: Number of top crops to analyze separately

#### Groundwater
- `--groundwater-max-depth`: Maximum depth to include in analysis
- `--groundwater-min-samples`: Minimum number of samples required for analysis

#### Governmental Units
- `--gov-units-db-path`: Path to governmental units database
- `--gov-units-layers`: Comma-separated list of layers to include

#### gSSURGO
- `--gssurgo-properties`: Comma-separated list of soil properties to analyze
- `--gssurgo-depth-range`: Depth range to analyze (e.g., "0-30,30-100")

#### Climate Change
- `--hist-start-year`: Start year for historical climate period (default: 2000)
- `--hist-end-year`: End year for historical climate period (default: 2014)
- `--fut-start-year`: Start year for future climate period (default: 2045)
- `--fut-end-year`: End year for future climate period (default: 2060)
- `--cc-model`: Climate model for climate change analysis (default: ACCESS-CM2)
- `--cc-ensemble`: Ensemble member (default: r1i1p1f1)
- `--cc-scenario`: Climate scenario for future projections (default: ssp245)

#### SNODAS
- `--snodas-variables`: Comma-separated list of SNODAS variables to include
- `--snodas-snow-season`: Months to consider for snow season (e.g., "11,12,1,2,3,4")

## Report Structure

The report generator creates a structured output directory containing:

1. **Main index.html**: A central hub linking to all generated reports
2. **Data source directories**: Separate folders for each data source:
   - `/prism/`
   - `/modis/`
   - `/cdl/`
   - `/groundwater/`
   - `/gov_units/`
   - `/gssurgo/`
   - `/climate_change/` 
   - `/nsrdb/`
   - `/snodas/`

3. **Content within each directory**:
   - Markdown report (e.g., `prism_report.md`)
   - HTML version of the report (e.g., `prism_report.html`)
   - Visualizations (PNG files)
   - Data tables (CSV files)
   - Additional analysis files

4. **Configuration file**: A `config.json` file in the main directory containing the settings used for report generation.

### Example Reports

#### PRISM Report
- `prism_report.md/html`: Comprehensive climate analysis
- `prism_stats.csv`: Time series of climate variables
- `prism_seasonal.png`: Seasonal climate patterns
- `prism_spatial.png`: Spatial distribution of climate variables
- `prism_timeseries.png`: Temporal trends over analysis period

#### MODIS Report
- `modis_comprehensive_report.md/html`: Analysis of vegetation indices
- Multiple subdirectories for different MODIS products
- Visualizations of vegetation patterns

#### CDL Report
- `cdl_report.md/html`: Analysis of agricultural land use
- `cdl_data.csv`: Tabular data on crop distribution
- Various visualizations for crop diversity, rotation patterns, and trends

#### Climate Change Report
- `climate_change_report.md/html`: Analysis of projected climate changes
- Multiple visualizations comparing historical and future climate
- Detailed CSV files with climate projection data

## Examples

### Generate a PRISM climate report only:
```bash
python GeoReporter/report_generator.py --type prism --output reports/prism_only --start-year 2015 --end-year 2020
```

### Generate a comprehensive report with climate change projections:
```bash
python GeoReporter/report_generator.py --type all --output reports/full_analysis --start-year 2010 --end-year 2020 --hist-start-year 2000 --hist-end-year 2014 --fut-start-year 2045 --fut-end-year 2060 --cc-scenario ssp585
```

### Generate a CDL report with custom crop analysis:
```bash
python GeoReporter/report_generator.py --type cdl --output reports/crop_analysis --cdl-recode-crops --cdl-top-n-crops 10
```

## Interpreting Reports

Each report contains several sections:
1. **Overview**: Basic information about the data and region
2. **Summary Statistics**: Statistical analysis of key variables
3. **Temporal Analysis**: Changes over time with trend analysis
4. **Spatial Analysis**: Geographic distribution of variables
5. **Seasonal Patterns**: How variables change throughout the year
6. **Implications**: Potential effects on agriculture, water resources, etc.
7. **Recommendations**: Suggestions based on the data analysis

The HTML index page provides easy navigation between different report types.

## Troubleshooting

- If report generation fails, check the error logs in the output directory.
- For large areas, consider increasing resolution to improve processing speed.
- If charts appear blank, verify the data date range is valid for the chosen dataset.
- For memory issues with large datasets, try running reports individually rather than with `--type all`.

## Performance Considerations

- Report generation can be memory-intensive, especially for high-resolution analyses.
- By default, the tool will try to use parallel processing, but you can force sequential processing with the `--sequential` flag.
- To improve performance:
  - Use a more focused bounding box
  - Limit the time range with `--start-year` and `--end-year`
  - Adjust the resolution parameter for faster processing
