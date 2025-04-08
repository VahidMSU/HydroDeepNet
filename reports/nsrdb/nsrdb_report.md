# NSRDB Climate Data Analysis Report

## Overview

**Period:** 2010 to 2020

**Data Source:** National Solar Radiation Database (NSRDB)

**Region:** Lat [43.1581, 44.1647], Lon [-85.4443, -84.2393]

**Available Variables:**

- Global Horizontal Irradiance (W/m²)
- Wind Speed (m/s)
- Relative Humidity (%)

## Summary Statistics

### Global Horizontal Irradiance

**Mean:** 161.76 W/m²

**Minimum:** 7.08 W/m²

**Maximum:** 373.52 W/m²

**Standard Deviation:** 98.41 W/m²

**Temporal Variability:** 0.608

**Spatial Variability:** 0.075

### Wind Speed

**Mean:** 2.64 m/s

**Minimum:** 0.63 m/s

**Maximum:** 8.59 m/s

**Standard Deviation:** 1.20 m/s

**Temporal Variability:** 0.454

**Spatial Variability:** 0.218

### Relative Humidity

**Mean:** 81.05 %

**Minimum:** 51.10 %

**Maximum:** 100.00 %

**Standard Deviation:** 8.48 %

**Temporal Variability:** 0.105

**Spatial Variability:** 0.017

## Seasonal Patterns

### Global Horizontal Irradiance by Month

| Year | Month | Mean | Min | Max | Std Dev |
|------|-------|------|-----|-----|---------|
| 2010 | January | 55.51 | 16.30 | 114.40 | 28.68 |
| 2010 | February | 77.24 | 33.04 | 175.07 | 36.13 |
| 2010 | March | 177.83 | 47.61 | 254.91 | 59.14 |
| 2010 | April | 203.54 | 33.61 | 327.12 | 87.48 |
| 2010 | May | 249.37 | 50.41 | 342.04 | 91.00 |
| 2010 | June | 258.83 | 130.63 | 365.17 | 80.22 |
| 2010 | July | 271.14 | 122.52 | 366.43 | 66.05 |
| 2010 | August | 235.29 | 139.49 | 310.59 | 47.27 |
| 2010 | September | 157.00 | 54.98 | 251.41 | 58.91 |
| 2010 | October | 133.20 | 37.35 | 198.69 | 44.84 |
| 2010 | November | 72.31 | 19.15 | 142.14 | 35.22 |
| 2010 | December | 53.73 | 24.40 | 96.42 | 20.92 |

### Wind Speed by Month

| Year | Month | Mean | Min | Max | Std Dev |
|------|-------|------|-----|-----|---------|
| 2010 | January | 3.07 | 0.93 | 5.61 | 1.22 |
| 2010 | February | 2.83 | 0.88 | 5.95 | 1.32 |
| 2010 | March | 2.70 | 1.15 | 5.87 | 1.07 |
| 2010 | April | 3.12 | 1.21 | 5.54 | 1.12 |
| 2010 | May | 2.54 | 1.07 | 4.97 | 1.02 |
| 2010 | June | 2.22 | 1.31 | 3.98 | 0.73 |
| 2010 | July | 1.84 | 0.80 | 3.15 | 0.70 |
| 2010 | August | 1.94 | 0.91 | 3.54 | 0.60 |
| 2010 | September | 2.51 | 0.98 | 5.72 | 1.22 |
| 2010 | October | 2.44 | 1.08 | 6.59 | 1.16 |
| 2010 | November | 2.76 | 0.89 | 6.01 | 1.22 |
| 2010 | December | 2.95 | 1.11 | 6.41 | 1.23 |

### Relative Humidity by Month

| Year | Month | Mean | Min | Max | Std Dev |
|------|-------|------|-----|-----|---------|
| 2010 | January | 84.18 | 66.91 | 98.75 | 7.75 |
| 2010 | February | 80.93 | 69.64 | 87.41 | 4.54 |
| 2010 | March | 75.94 | 53.85 | 97.42 | 10.62 |
| 2010 | April | 75.34 | 60.99 | 92.70 | 8.26 |
| 2010 | May | 78.63 | 60.95 | 93.94 | 8.45 |
| 2010 | June | 83.74 | 69.92 | 97.35 | 6.52 |
| 2010 | July | 82.19 | 63.82 | 96.24 | 6.59 |
| 2010 | August | 79.45 | 62.34 | 93.35 | 7.83 |
| 2010 | September | 80.21 | 68.05 | 94.69 | 6.72 |
| 2010 | October | 78.00 | 61.69 | 93.34 | 7.50 |
| 2010 | November | 81.49 | 60.69 | 97.09 | 9.04 |
| 2010 | December | 82.05 | 68.75 | 99.80 | 7.74 |

### Seasonal Analysis

**Solar Radiation:**

The highest solar radiation typically occurs in **July**, while the lowest is in **December**. This pattern follows the expected seasonal variation based on day length and sun angle.

**Wind Speed:**

Wind speeds are typically highest in **February** and lowest in **August**. This seasonal pattern should be considered for wind energy applications and agricultural planning.

## Advanced Solar Radiation Analysis

### Solar Energy Potential

The study area receives an average of **3.88 kWh/m²/day** of solar energy, equivalent to **15599.1 kWh/m²/year**. The variability in solar resources (coefficient of variation) is 0.61.

**Key solar metrics:**

- Daily minimum: 0.17 kWh/m²/day
- Daily maximum: 8.96 kWh/m²/day
- 90th percentile: 7.39 kWh/m²/day
- 10th percentile: 1.02 kWh/m²/day

![Solar Energy Potential](solar_analysis/solar_energy_potential.png)

### Photovoltaic System Performance

For a standard 1 kW photovoltaic system:

- Annual production: **1062.8 kWh**
- Capacity factor: 12.1%
- Performance ratio: 0.75
- Estimated annual value: $127.53
- Simple payback period: 23.5 years

A simulation of a 10 kW system shows the following performance characteristics:

![PV System Simulation](solar_analysis/pv_simulation.png)

### High Solar Radiation Periods

Extended periods of high solar radiation ('heat waves') were analyzed using a threshold of the 90th percentile (308.1 W/m²):

- Number of heat wave events: **46**
- Average duration: 3.9 days
- Maximum duration: 9 days
- Percentage of days in heat wave: 4.4%

**Monthly distribution of heat wave events:**

| Month | Number of Events |
|-------|----------------|
| Jan | 0 |
| Feb | 0 |
| Mar | 0 |
| Apr | 2 |
| May | 11 |
| Jun | 19 |
| Jul | 14 |
| Aug | 0 |
| Sep | 0 |
| Oct | 0 |
| Nov | 0 |
| Dec | 0 |

![Heat Wave Analysis](solar_analysis/heat_wave_analysis.png)

### Solar Radiation Extremes

Analysis of extreme solar radiation values revealed:

- Extreme high days per year: 18.3
- Extreme low days per year: 18.3
- Longest high radiation run: 9 days
- Longest low radiation run: 4 days

**Top 5 highest radiation days:**

| Date | Radiation (W/m²) |
|------|----------------|
| 2020-06-14 | 373.52 |
| 2016-06-12 | 373.44 |
| 2020-06-15 | 371.81 |
| 2016-06-29 | 370.69 |
| 2018-07-06 | 369.21 |

### Climate Variable Correlations

The analysis of correlations between solar radiation and other climate variables shows:

- Solar radiation vs. humidity: **-0.45**
- Solar radiation vs. wind speed: **-0.36**
- Solar aridity index: 2.07

![Climate Correlations](solar_analysis/climate_correlations.png)

## Time Series Analysis

The time series shows the change in climate variables over the entire period.

![Time Series](nsrdb_timeseries.png)

## Spatial Distribution

The spatial maps show the distribution of climate variables across the study area.

![Spatial Distribution](nsrdb_spatial.png)

## Applications

### Solar Energy Applications

The solar radiation data can be used for:

- Solar energy potential assessment
- Photovoltaic system design and optimization
- Solar resource forecasting
- Agricultural solar applications (solar drying, greenhouse operations)

**Solar design recommendations:**

- **Moderate solar potential:** Solar installations are feasible but will have lower yields compared to sunnier regions.
- **High variability noted:** Consider energy storage solutions to manage the variable solar resource.
### Wind Energy Applications

The wind speed data can be used for:

- Wind farm siting and energy production estimation
- Small-scale wind energy applications
- Agricultural applications (windbreaks, ventilation)

### Agricultural Applications

The NSRDB data can inform agricultural decisions related to:

- Evapotranspiration modeling using solar radiation data
- Crop growth modeling incorporating radiation and humidity
- Planning for wind-sensitive operations
- Irrigation scheduling

### Hydrological Modeling Applications

The NSRDB data can be used in SWAT+ and other hydrological models for:

- Improved evapotranspiration calculations
- Energy balance modeling
- Snow accumulation and melt modeling

## Data Source and Methodology

This analysis is based on the National Solar Radiation Database (NSRDB), which provides solar radiation, meteorological data, and other environmental parameters. The NSRDB uses a combination of satellite imagery, ground measurements, and physical models to generate gridded estimates of solar radiation and related variables.

**Variable details:**

| Variable | Description | Units | Scaling |
|----------|-------------|-------|--------|
| ghi | Global Horizontal Irradiance | W/m² | 1.0 |
| wind_speed | Wind Speed | m/s | 10.0 |
| relative_humidity | Relative Humidity | % | 100.0 |

**Processing steps:**

1. Extraction of raw data from HDF5 database
2. Application of correct scaling factors
3. Spatial subsetting to the region of interest
4. Temporal aggregation from 30-minute to daily values
5. Statistical analysis and visualization
6. Advanced solar resource assessment and PV performance modeling
7. Analysis of extreme events and climate correlations

### Limitations

- The NSRDB data has a spatial resolution that may not capture fine-scale variations
- The analysis is limited by the temporal range of the available data
- Local terrain effects and microclimates may not be fully represented
- Cloud cover and atmospheric conditions can affect data accuracy
- PV performance estimates use standard assumptions and may differ from actual system performance

## Data Export

The complete dataset has been exported to CSV format. Access the data at: [nsrdb_stats.csv](nsrdb_stats.csv)

---

*Report generated on 2025-04-08 at 18:02*
