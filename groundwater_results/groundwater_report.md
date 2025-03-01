# Groundwater Properties Analysis Report

## Overview

**Region:** Lat [43.1581, 44.1647], Lon [-85.4443, -84.2393]

**Available Variables:**

- Upper Aquifer Thickness (ft)
- Lower Aquifer Thickness (ft)
- Upper Aquifer Horizontal Hydraulic Conductivity (ft/day)
- Lower Aquifer Horizontal Hydraulic Conductivity (ft/day)
- Upper Aquifer Vertical Hydraulic Conductivity (ft/day)
- Lower Aquifer Vertical Hydraulic Conductivity (ft/day)
- Upper Aquifer Transmissivity (ft²/day)
- Lower Aquifer Transmissivity (ft²/day)
- Static Water Level (ft below surface)

## Summary Statistics

### Upper Aquifer Thickness (AQ_THK_1)

**Mean:** 56.3088 ft

**Median:** 52.6617 ft

**Range:** 15.0843 - 395.2444 ft

**Standard Deviation:** 18.3293 ft

**Coefficient of Variation:** 32.55%

**Coverage:** 100.00% (182284 of 182284 cells)

**Mean Error:** 17.2449 ft

### Lower Aquifer Thickness (AQ_THK_2)

**Mean:** 76.4172 ft

**Median:** 72.0493 ft

**Range:** 16.9812 - 451.3060 ft

**Standard Deviation:** 27.6194 ft

**Coefficient of Variation:** 36.14%

**Coverage:** 100.00% (182284 of 182284 cells)

**Mean Error:** 22.5043 ft

### Upper Aquifer Horizontal Hydraulic Conductivity (H_COND_1)

**Mean:** 243.6227 ft/day

**Median:** 241.7318 ft/day

**Range:** -7.2863 - 745.4312 ft/day

**Standard Deviation:** 110.1300 ft/day

**Coefficient of Variation:** 45.21%

**Coverage:** 100.00% (182284 of 182284 cells)

**Mean Error:** 45.7050 ft/day

### Lower Aquifer Horizontal Hydraulic Conductivity (H_COND_2)

**Mean:** 240.7356 ft/day

**Median:** 235.5305 ft/day

**Range:** -5.8463 - 692.9962 ft/day

**Standard Deviation:** 109.7985 ft/day

**Coefficient of Variation:** 45.61%

**Coverage:** 100.00% (182284 of 182284 cells)

**Mean Error:** 44.3235 ft/day

### Upper Aquifer Vertical Hydraulic Conductivity (V_COND_1)

**Mean:** 236.5361 ft/day

**Median:** 235.8721 ft/day

**Range:** -4.2345 - 686.6814 ft/day

**Standard Deviation:** 107.0677 ft/day

**Coefficient of Variation:** 45.26%

**Coverage:** 100.00% (182284 of 182284 cells)

**Mean Error:** 44.4986 ft/day

### Lower Aquifer Vertical Hydraulic Conductivity (V_COND_2)

**Mean:** 225.5545 ft/day

**Median:** 220.5628 ft/day

**Range:** -3.3603 - 592.0555 ft/day

**Standard Deviation:** 103.3738 ft/day

**Coefficient of Variation:** 45.83%

**Coverage:** 100.00% (182284 of 182284 cells)

**Mean Error:** 43.5317 ft/day

### Upper Aquifer Transmissivity (TRANSMSV_1)

**Mean:** 14870.1702 ft²/day

**Median:** 14147.6867 ft²/day

**Range:** 63.9456 - 108351.9822 ft²/day

**Standard Deviation:** 9325.6928 ft²/day

**Coefficient of Variation:** 62.71%

**Coverage:** 100.00% (182284 of 182284 cells)

**Mean Error:** 5565.4146 ft²/day

### Lower Aquifer Transmissivity (TRANSMSV_2)

**Mean:** 20160.4575 ft²/day

**Median:** 18479.6349 ft²/day

**Range:** 104.0878 - 144109.0378 ft²/day

**Standard Deviation:** 13458.5711 ft²/day

**Coefficient of Variation:** 66.76%

**Coverage:** 100.00% (182284 of 182284 cells)

**Mean Error:** 7307.0404 ft²/day

### Static Water Level (SWL)

**Mean:** 112.8735 ft below surface

**Median:** 78.4845 ft below surface

**Range:** -388.1354 - 25436.9356 ft below surface

**Standard Deviation:** 251.2909 ft below surface

**Coefficient of Variation:** 222.63%

**Coverage:** 100.00% (182284 of 182284 cells)

**Mean Error:** 74.7967 ft below surface

## Spatial Distribution

The maps below show the spatial distribution of groundwater properties across the study area.

![Groundwater Maps](groundwater_maps.png)

## Estimation Uncertainty

These maps show the standard error of the kriging estimates, indicating the level of confidence in the data.

![Error Maps](groundwater_error_maps.png)

## Statistical Distributions

Histograms showing the frequency distribution of each parameter.

![Histograms](groundwater_histograms.png)

## Property Correlations

This heatmap shows the Pearson correlation coefficients between different groundwater properties.

![Correlation Matrix](groundwater_correlation.png)

## Property Relationships

### Hydraulic Conductivity vs. Transmissivity (Upper Aquifer)

Transmissivity is theoretically the product of hydraulic conductivity and aquifer thickness.

![K vs T Upper](compare_H_COND_1_TRANSMSV_1.png)

### Hydraulic Conductivity vs. Transmissivity (Lower Aquifer)

The relationship between K and T in the lower aquifer may differ from the upper aquifer.

![K vs T Lower](compare_H_COND_2_TRANSMSV_2.png)

### Aquifer Thickness vs. Hydraulic Conductivity (Upper Aquifer)

![Thickness vs K Upper](compare_AQ_THK_1_H_COND_1.png)

### Aquifer Thickness vs. Hydraulic Conductivity (Lower Aquifer)

![Thickness vs K Lower](compare_AQ_THK_2_H_COND_2.png)

## Hydrogeologic Implications

The water table is **deep** in this area (mean: 112.87 ft). Deep groundwater conditions suggest:

- Greater pumping costs for water extraction
- Good protection from surface contamination
- Limited interaction with surface water systems
- Potential for confined aquifer conditions

The high spatial variability in water table depth (CV: 222.63%) indicates complex hydrogeologic conditions across the area.

The upper aquifer has **very high** hydraulic conductivity (243.62 ft/day) and **high** transmissivity (14870.17 ft²/day). With an average thickness of 56.31 ft, this suggests:

- High potential well yield (suitable for municipal or industrial use)
- Rapid groundwater movement and potential for fast contaminant transport
- Material likely consists of clean gravel or karstic limestone

## Recommendations

Based on the groundwater properties analysis, the following recommendations are provided:

### Well Development

- The upper aquifer has good hydraulic properties for well development
- Target depth: 117.9 - 169.2 m
- The lower aquifer has good hydraulic properties and may be a viable target

### Groundwater Management

- Establish monitoring wells to track long-term water level trends
- Develop sustainable extraction limits based on aquifer properties

### Further Investigations

- Conduct aquifer tests to verify estimated hydraulic properties
- Analyze groundwater quality to assess suitability for intended uses
- Develop numerical groundwater flow models for scenario analysis

## Data Source and Methodology

This analysis is based on the Empirical Bayesian Kriging (EBK) interpolation of groundwater properties. EBK is a geostatistical interpolation method that accounts for the uncertainty in semivariogram estimation through a process of subsetting and simulation.

**Processing steps:**

1. Extraction of interpolated data from HDF5 database
2. Spatial subsetting to the region of interest
3. Statistical analysis and visualization
4. Assessment of property relationships and implications

### Limitations

- The analysis is based on interpolated data rather than direct measurements
- Kriging uncertainty may be high in areas with sparse well data
- Local heterogeneity may not be captured at the analysis resolution
- Temporal variations in water levels are not addressed

## Data Export

The complete dataset has been exported to CSV format. Access the data at: [groundwater_stats.csv](groundwater_stats.csv)

---

*Report generated on 2025-02-28 at 21:02*
