# gSSURGO Soil Analysis Report

## Overview

**Region:** Lat [42.4219, 42.7766], Lon [-84.6031, -84.1406]

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
| pH | 5.65 | Optimal | Favorable for most crops; good nutrient availability |
| Organic Carbon | 4.19% | High | High fertility; good soil structure; high water retention |
| Available Water Capacity | 0.140 cm/cm | Moderate | Moderate water retention; average irrigation needs |
| Soil Texture | Sandy Loam | - | Affects drainage and nutrient retention |
| Drainage | - | Good | Well-drained; low risk of waterlogging |
| Nutrient Retention | - | Low | Poor nutrient retention; potential leaching issues |

## Soil Parameter Statistics

The following table presents key statistics for the analyzed soil parameters:

| Parameter | Mean | Median | Min | Max | Std Dev | CV |
|-----------|------|--------|-----|-----|---------|----|
| Albedo | 0.226 | 0.300 | 0.000 | 0.370 | 0.123 | 0.546 |
| Available Water Capacity | 0.140 | 0.132 | 0.000 | 0.450 | 0.094 | 0.672 |
| Bulk Density | 1.425 | 1.527 | 0.150 | 1.756 | 0.404 | 0.283 |
| Calcium Carbonate | 4.948 | 4.500 | 0.000 | 40.000 | 4.921 | 0.995 |
| Organic Carbon | 4.186 | 0.430 | 0.000 | 50.872 | 11.558 | 2.761 |
| Clay Content | 13.807 | 17.167 | 0.000 | 36.000 | 8.942 | 0.648 |
| Depth | 853.783 | 897.500 | 20.000 | 1775.000 | 263.657 | 0.309 |
| Total Soil Depth | 1646.347 | 1730.000 | 20.000 | 2030.000 | 549.156 | 0.334 |
| Electrical Conductivity | 0.138 | 0.000 | 0.000 | 1.133 | 0.268 | 1.941 |
| Soil pH | 5.646 | 6.767 | 0.000 | 7.800 | 2.714 | 0.481 |
| Rock Fragment Content | 5.994 | 3.000 | 0.000 | 31.000 | 6.172 | 1.030 |
| Sand Content | 43.671 | 44.625 | 0.000 | 95.400 | 26.277 | 0.602 |
| Silt Content | 23.771 | 26.067 | 0.000 | 60.420 | 15.854 | 0.667 |
| Hydraulic Conductivity | 57.956 | 32.400 | 0.000 | 338.400 | 74.507 | 1.286 |

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

- **Mean soil depth:** 1646.3 cm
- **Areas with shallow soil (<50cm):** 0.5%
- **Limitation level:** Slight
- **Recommendation:** Choose shallow-rooted crops or implement raised beds in affected areas.

### Rock Content Limitations

- **Mean rock content:** 6.0%
- **Areas with high rock content (>15%):** 12.6%
- **Limitation level:** Moderate
- **Recommendation:** Consider rock removal or selecting crops tolerant of rocky soils.

### Drainage Limitations

- **Areas with poor drainage:** 0.0%
- **Limitation level:** Slight
- **Recommendation:** Install drainage systems or select water-tolerant crops in poorly drained areas.

### pH Limitations

- **Areas with acidic soils (pH<5.5):** 21.0%
- **Areas with alkaline soils (pH>7.5):** 15.4%
- **Limitation level:** Moderate
- **Recommendation:** Apply lime in acidic areas or sulfur in alkaline areas to adjust pH.

### Salinity Limitations

- **Mean electrical conductivity:** 0.14 dS/m
- **Areas with high salinity (>4 dS/m):** 0.0%
- **Limitation level:** Slight
- **Recommendation:** Leach salts with irrigation, use salt-tolerant crops in affected areas.

## Soil Management Recommendations

Based on the analysis of soil properties, the following management practices are recommended:

1. Apply fertilizers in smaller, more frequent doses to prevent nutrient leaching.

## Parameter Correlations

The correlation matrix below shows relationships between soil parameters. Strong positive correlations appear in red, while strong negative correlations appear in blue.

![Soil Parameter Correlation Matrix](soil_correlation.png)

### Key Parameter Relationships

| Parameter 1 | Parameter 2 | Correlation | Interpretation |
|------------|------------|-------------|---------------|
| Depth | Total Soil Depth | 0.89 | Strong positive relationship |
| Organic Carbon | Electrical Conductivity | 0.88 | Strong positive relationship |
| Bulk Density | Organic Carbon | -0.88 | Strong negative relationship |
| Albedo | Soil pH | 0.85 | Strong positive relationship |
| Available Water Capacity | Organic Carbon | 0.83 | Strong positive relationship |

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

*Report generated on 2025-04-03 at 09:35*
