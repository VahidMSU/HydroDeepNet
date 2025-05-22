# HydroDeepNet Web Application

A comprehensive web-based platform for hydrological modeling, deep learning, and environmental data analysis.

## Overview

HydroDeepNet is an integrated platform that combines:
- Automated SWAT+ model generation (SWATGenX)
- Deep learning capabilities for hydrological predictions
- Comprehensive hydrological and geological datasets
- Interactive visualizations and reporting

![HydroDeepNet Architecture](HydroDeepNet.png)

## Key Features

### 1. Core Components

- **SWATGenX**: Automated SWAT+ model creation for USGS streamgage stations
- **Vision System**: Deep learning framework for hydrological predictions
- **HydroGeoDataset**: Comprehensive environmental data access
- **Interactive Visualizations**: Real-time data visualization tools

### 2. User Features

- Secure authentication system
- User dashboard for file management
- Report generation and visualization
- FTPS server access for data transfer
- Interactive chat assistant for support

### 3. Data Integration

- PRISM climate data
- MODIS satellite data
- LOCA2 climate projections
- Wellogic geological data
- NHDPlus hydrological data

## Technology Stack

- **Frontend**: React.js with Material-UI
- **Backend**: Python Flask
- **Storage**: HDF5 database
- **Authentication**: JWT-based auth system
- **File Transfer**: FTPS server
- **AI Assistant**: Ollama-powered chat system

## Getting Started

### Prerequisites

```sh
node >= 14.0.0
npm >= 6.14.0
Python >= 3.8