# HydroDeepNet

An AI-powered online platform integrating hydrological modeling, deep learning, and multi-agent AI systems to provide comprehensive hydrological insights at local, state, and national scales.

![HydroDeepNet Architecture](./HydroDeepNet.drawio "HydroDeepNet System Architecture")

## Overview

HydroDeepNet represents a revolutionary approach to hydrological modeling and analysis, leveraging advanced AI techniques and existing hydrological models to deliver unprecedented insights. The platform consists of four major components:

### 1. Automated Hydrological Model Generation

HydroDeepNet streamlines the generation, calibration, and validation of:
- Surface water models (SWAT+)
- Groundwater models (MODFLOW)

**Key advantages:**
- High-resolution modeling capability across the conterminous United States (CONUS)
- Significantly reduced model creation time
- Leverages NHDPlus HR (1:24k resolution), providing 20 times greater detail than other platforms like NAM
- More precise watershed delineation and hydrological modeling
- Improved water balance estimation and contaminant fate predictions

**Validation:** Successfully created 660 models out of 700 attempted across the US using USGS Federally Prioritized Streamgage (FPS) data, with 40 failures attributed to hydrographical complexities. Additionally, 60 models were calibrated and validated for USGS watersheds across Michigan with high predictive accuracy.

### 2. 4D Spatiotemporal Deep Learning Vision System

A hybrid CNN-Transformer deep learning model with 130 million parameters for predicting:
- Evapotranspiration
- Groundwater recharge in 4D (time and 3D space)

**Benefits:**
- Faster hydrological predictions compared to traditional modeling
- Comparable accuracy to physics-based models
- Enhanced model predictions
- Improved data accuracy in missing-value scenarios
- Support for drought monitoring, flood forecasting, and groundwater management

### 3. Automated Environmental and Agricultural Reporting System

Compiles and structures large datasets to generate automated reports on:
- Climate conditions
- Crop composition
- Solar energy balance
- Climate change projections
- Soil properties
- Groundwater hydraulic conditions

Reports are generated in a structured format with markdown-based evaluations, allowing rapid assessments of agricultural and environmental conditions. Currently operational for Michigan with plans for expansion to CONUS.

### 4. Multi-AI Agent System for Hydrological Insights

A sophisticated multi-agent AI system that:
- Analyzes reports and model outputs
- Provides evidence-based agricultural and environmental insights
- Utilizes a Retrieval-Augmented Generation (RAG) approach

The system is partially implemented and will be fully operational upon the official web application launch.

## Technical Architecture

The HydroDeepNet system architecture integrates multiple data sources, modeling frameworks, and AI components as illustrated in the flowchart. Key components include:

- **Data Sources:** NLCD, PRISM, USGS DEM, NHDPlus HR, etc.
- **Modeling Frameworks:** SWATGenX, QSWAT+, SWAT+, MODFLOW-NWT
- **AI Components:** Vision System Deep Learning Framework (PyTorch), Multi-AI agents RAG system
- **Data Storage:** HydroGeoDataset (HDF5)
- **Processing Systems:** Parallel Processing System for model calibration and validation

## Current Status

- The automated hydrological model generation system is fully functional
- The deep learning vision system has been developed with results submitted for publication
- The automated reporting system is operational for Michigan
- The multi-agent AI system is under development

## Future Work

- Expand coverage to the entire CONUS
- Enhance the multi-agent AI system capabilities
- Integrate additional environmental and agricultural data sources
- Develop user-friendly interfaces for broader accessibility

---

© HydroDeepNet Team