# MODIS Normalized Difference Vegetation Index Analysis Report

## Overview

**Product:** MOD13Q1_NDVI - Normalized Difference Vegetation Index

**Period:** 2010 to 2016

**Region:** Lat [43.1581, 44.1647], Lon [-85.4443, -84.2393]

**Units:** NDVI

**Data points:** 84

## Summary Statistics

**Mean:** 0.4870 NDVI

**Minimum:** 0.0417 NDVI

**Maximum:** 0.8159 NDVI

**Standard Deviation:** 0.2541 NDVI

## Yearly Averages

| Year | Mean | Min | Max | Std Dev |
|------|------|-----|-----|---------|
| 2010 | 0.4833 | 0.0430 | 0.8159 | 0.2773 |
| 2011 | 0.5079 | 0.0477 | 0.8087 | 0.2374 |
| 2012 | 0.4705 | 0.0417 | 0.8111 | 0.2787 |
| 2013 | 0.4856 | 0.0997 | 0.8132 | 0.2336 |

## Seasonal Patterns

The seasonal analysis shows how Normalized Difference Vegetation Index values vary throughout the year.

![Seasonal Analysis](MOD13Q1_NDVI_seasonal.png)

The highest values typically occur during **Spring**, while the lowest values are in **Winter**. The seasonal range is approximately 0.2360 NDVI, representing a 48.0% variation from the annual mean.

## Time Series Analysis

The time series shows the change in Normalized Difference Vegetation Index over the entire period.

![Time Series](MOD13Q1_NDVI_timeseries.png)

### Trend Analysis

The data shows a decreasing trend of 0.0030 NDVI per year (p-value: 0.7467, not statistically significant).

Over the entire 7 year period, this represents a change of 0.0183 NDVI (3.7%).

## Spatial Distribution

The spatial map shows the distribution of Normalized Difference Vegetation Index across the study area.

![Spatial Distribution](MOD13Q1_NDVI_spatial.png)

### Pixel-level Statistics

**Valid data pixels:** 182284 out of 182284 (100.0%)

**Spatial variability:** 0.0648 NDVI (coefficient of variation: 111.3%)

## Potential Applications

This vegetation index data can be used for:

- Monitoring crop health and agricultural productivity
- Assessing drought impacts on vegetation
- Tracking seasonal vegetation phenology
- Land cover change detection

## Data Source and Methodology

This analysis is based on the MODIS (Moderate Resolution Imaging Spectroradiometer) satellite data. The original MODIS data products are provided by NASA's Earth Observing System Data and Information System (EOSDIS) and have been processed to extract time series for the specified geographic region.

**Processing steps:**

1. Extraction of raw data from HDF5 database
2. Spatial subsetting to the region of interest
3. Application of quality filters and scaling factors
4. Statistical analysis and visualization

## Data Export

The complete dataset has been exported to CSV format. Access the data at: [MOD13Q1_NDVI_stats.csv](MOD13Q1_NDVI_stats.csv)

---

*Report generated on 2025-02-28 at 22:26*
