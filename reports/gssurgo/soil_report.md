# gSSURGO Soil Analysis Report

## Overview

**Region:** Lat [43.1581, 44.1647], Lon [-85.4443, -84.2393]

**Available Soil Parameters:**

- Albedo (fraction)
- Available Water Capacity (cm/cm)
- Bulk Density (g/cm³)
- Calcium Carbonate (%)
- Organic Carbon (%)
- Clay Content (%)
- Depth (cm)
- Total Soil Depth (cm)
- Electrical Conductivity (dS/m)
- Soil pH (pH)
- Rock Fragment Content (%)
- Sand Content (%)
- Silt Content (%)
- Hydraulic Conductivity (mm/hr)

## Soil Characteristics Summary

**Dominant Soil Texture:** Sandy Loam

**Texture Composition:**

![Soil Texture Composition](soil_texture.png)

**Soil Fertility Level:** Moderate

**Description:** Moderately fertile soils requiring standard amendments for good productivity.

### Soil Fertility Components

| Parameter | Value | Rating | Implication |
|-----------|-------|--------|-------------|
| pH | 6.09 | Optimal | Favorable for most crops; good nutrient availability |
| Organic Carbon | 2.73% | Moderate | Moderate fertility; adequate soil structure |
| Available Water Capacity | 0.119 cm/cm | Moderate | Moderate water retention; average irrigation needs |
| Soil Texture | Sandy Loam | - | Affects drainage and nutrient retention |
| Drainage | - | Good | Well-drained; low risk of waterlogging |
| Nutrient Retention | - | Low | Poor nutrient retention; potential leaching issues |

## Soil Parameter Statistics

The following table presents key statistics for the analyzed soil parameters:

| Parameter | Mean | Median | Min | Max | Std Dev | CV |
|-----------|------|--------|-----|-----|---------|----|
| Albedo | 0.220 | 0.230 | 0.000 | 0.370 | 0.080 | 0.364 |
| Available Water Capacity | 0.119 | 0.112 | 0.000 | 0.550 | 0.063 | 0.530 |
| Bulk Density | 1.499 | 1.537 | 0.150 | 1.756 | 0.243 | 0.162 |
| Calcium Carbonate | 3.560 | 3.333 | 0.000 | 67.500 | 3.413 | 0.959 |
| Organic Carbon | 2.729 | 0.561 | 0.000 | 53.052 | 8.221 | 3.012 |
| Clay Content | 13.566 | 10.400 | 0.000 | 68.000 | 10.174 | 0.750 |
| Depth | 872.554 | 842.500 | 20.000 | 2030.000 | 154.719 | 0.177 |
| Total Soil Depth | 1788.768 | 2000.000 | 20.000 | 2030.000 | 263.629 | 0.147 |
| Electrical Conductivity | 0.056 | 0.000 | 0.000 | 1.100 | 0.088 | 1.562 |
| Soil pH | 6.089 | 6.540 | 0.000 | 8.200 | 1.724 | 0.283 |
| Rock Fragment Content | 5.283 | 4.000 | 0.000 | 31.000 | 3.907 | 0.740 |
| Sand Content | 65.374 | 70.750 | 0.000 | 97.333 | 24.349 | 0.372 |
| Silt Content | 18.884 | 18.233 | 0.000 | 68.900 | 14.459 | 0.766 |
| Hydraulic Conductivity | 162.241 | 156.528 | 0.000 | 1015.200 | 162.911 | 1.004 |

*CV: Coefficient of Variation (Std Dev / Mean)*

## Soil Parameter Distributions

The distributions of key soil parameters are shown below:

![Soil Parameter Distributions](soil_distributions_composite.png)

*Individual parameter distributions can be found in the 'distributions' folder.*

## Spatial Distribution Maps

The spatial distribution of key soil parameters across the study area is shown below:

![Soil Parameter Maps](soil_maps_composite.png)

*Individual parameter maps can be found in the 'parameter_maps' folder.*

## Soil Limitations

### Soil Depth Limitations

- **Mean soil depth:** 1788.8 cm
- **Areas with shallow soil (<50cm):** 0.0%
- **Limitation level:** Slight
- **Recommendation:** Choose shallow-rooted crops or implement raised beds in affected areas.

### Rock Content Limitations

- **Mean rock content:** 5.3%
- **Areas with high rock content (>15%):** 2.0%
- **Limitation level:** Slight
- **Recommendation:** Consider rock removal or selecting crops tolerant of rocky soils.

### Drainage Limitations

- **Areas with poor drainage:** 0.8%
- **Limitation level:** Slight
- **Recommendation:** Install drainage systems or select water-tolerant crops in poorly drained areas.

### pH Limitations

- **Areas with acidic soils (pH<5.5):** 16.5%
- **Areas with alkaline soils (pH>7.5):** 6.5%
- **Limitation level:** Slight
- **Recommendation:** Apply lime in acidic areas or sulfur in alkaline areas to adjust pH.

### Salinity Limitations

- **Mean electrical conductivity:** 0.06 dS/m
- **Areas with high salinity (>4 dS/m):** 0.0%
- **Limitation level:** Slight
- **Recommendation:** Leach salts with irrigation, use salt-tolerant crops in affected areas.

## Soil Management Recommendations

Based on the analysis of soil properties, the following management practices are recommended:

1. Maintain organic matter through crop residue retention and minimal tillage.
2. Apply fertilizers in smaller, more frequent doses to prevent nutrient leaching.

## Parameter Correlations

The correlation matrix below shows relationships between soil parameters. Strong positive correlations appear in red, while strong negative correlations appear in blue.

![Soil Parameter Correlation Matrix](soil_correlation.png)

### Key Parameter Relationships

| Parameter 1 | Parameter 2 | Correlation | Interpretation |
|------------|------------|-------------|---------------|
| Bulk Density | Organic Carbon | -0.92 | Strong negative relationship |
| Clay Content | Silt Content | 0.90 | Strong positive relationship |
| Sand Content | Silt Content | -0.86 | Strong negative relationship |
| Clay Content | Sand Content | -0.84 | Strong negative relationship |
| Available Water Capacity | Organic Carbon | 0.82 | Strong positive relationship |

## Data Source and Methodology

This report is based on the Gridded Soil Survey Geographic (gSSURGO) database, which provides soil information at a 250m resolution. The gSSURGO database combines data from soil surveys conducted by the USDA Natural Resources Conservation Service (NRCS).

**Processing steps:**

1. Extraction of soil data for the specified region
2. Statistical analysis of soil parameters
3. Soil texture classification based on sand, silt, and clay content
4. Assessment of soil fertility and limitations
5. Generation of visualizations and spatial maps

### Limitations

- The analysis is based on the resolution of the gSSURGO data (250m), which may not capture fine-scale soil variations
- Some parameters may have missing data in certain areas
- Local soil conditions may vary and on-site testing is recommended for specific applications

## Data Export

The complete dataset has been exported to CSV format for further analysis: [soil_data.csv](soil_data.csv)

---

*Report generated on 2025-03-07 at 07:39*
