# HydroDeepNet: AI-Powered Hydrological Insights Platform

HydroDeepNet is an AI-powered online platform that integrates hydrological modeling, deep learning, and multi-agent AI systems to provide comprehensive hydrological insights at local, state, and national scales.

## Key Developments

The platform currently consists of four significant developments:

### 1. Automated Hydrological Model Generation

HydroDeepNet streamlines the generation, calibration, and validation of surface water (SWAT+) and groundwater (MODFLOW) models at high resolution with exceptional flexibility. It enables rapid model creation for any location across the conterminous United States (CONUS), significantly reducing the traditionally time-consuming process that requires extensive data collection and technical expertise.

HydroDeepNet leverages NHDPlus HR (1:24k resolution) to provide precise watershed delineation and enhanced hydrological modeling, improving water balance estimation and contaminant fate predictions.

The system has been tested with 700 models across the US using USGS Federally Prioritized Streamgage (FPS) data, successfully creating 660 models while 40 failed due to hydrographical complexities. Additionally, 60 calibrated and validated models were established for USGS watersheds across Michigan, demonstrating high predictive accuracy.

### 2. 4D Spatiotemporal Deep Learning Vision System

HydroDeepNet includes a hybrid CNN-Transformer deep learning model with 130 million parameters to predict evapotranspiration and groundwater recharge in 4D (time and 3D space). The results have been submitted to the Engineering Applications of Artificial Intelligence journal.

Traditional modeling, such as SWAT+ and MODFLOW, remains resource-intensive. While HydroDeepNet simplifies model creation, deep learning models provide a faster alternative for hydrological predictions with comparable accuracy.

Deep learning enhances model predictions, improves data accuracy in missing-value scenarios, and supports decision-making for drought monitoring, flood forecasting, and groundwater management.

### 3. Automated Environmental and Agricultural Reporting System

HydroDeepNet compiles and structures large datasets, generating automated reports on environmental indicators, including climate conditions, crop composition, solar energy balance, climate change projections, soil properties, and groundwater hydraulic conditions.

Reports are generated in a structured format with markdown-based evaluations, allowing rapid assessments of agricultural and environmental conditions. The system is currently operational for Michigan and will be expanded to CONUS.

### 4. Multi-AI Agent System for Hydrological Insights

A multi-agent AI system reads reports and model outputs, providing users with evidence-based agricultural and environmental insights.

While the current AI agent implementation is limited, it will be fully operational upon the official launch of the web application.

## Software and Libraries

HydroDeepNet incorporates elements from various open-source software and libraries, including:

*   QGIS, QSWAT+, and SWATPlusEditor are used to create SWAT+ models. Example: [QGSWAT+ GitHub Repository](https://github.com/swat-model/QSWATPlus)
*   GDAL library for the analyses and processing of geospatial data.
*   Flopy library for creating MODFLOW models.
*   PyTorch library for creating and training hydrological deep learning models.
*   Ollama library for the Multi-AI agents of the platform.
*   React for serving frontend
*   Docker and Kubernetes for deploying the platform

![HydroDeepNet Architecture](./HydroDeepNet.drawio.svg)