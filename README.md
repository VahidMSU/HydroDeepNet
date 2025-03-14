```mermaid
graph TD;
    subgraph National_Database[National Database]
        PRISM -->|Climate Data| HydroGeoDataset
        LOCA2 -->|Climate Data| HydroGeoDataset
        NLCD -->|Land Cover| HydroGeoDataset
        NSRSDB -->|Solar Radiation| HydroGeoDataset
        gSSURGO -->|Soil Data| HydroGeoDataset
        3D_Elevation_Program -->|Elevation Data| HydroGeoDataset
    end


    subgraph Data_Processing
        NWIS --> SWATGenX;
        "NHDPlus HR" --> SWATGenX;
        SWATGenX --> QSWAT+;
        SWATGenX --> "SWAT+Editor";
        "Water well information (Wellogic)" --> "Empirical Bayesian Kriging";
        "Empirical Bayesian Kriging" --> "Hydraulic properties";
        "Hydraulic properties" --> MODGenX;
        MODGenX --> Flopy;
        Flopy --> "MODFLOW-NWT";
    end

    subgraph SWAT_Processing
        "SWAT+" -->|Streams| "SWAT+gwflow";
        "SWAT+gwflow" --> HydroGeoDataset;
    end

    subgraph Parallel_Processing
        HydroGeoDataset -->|Validation (ensemble)| HydroGeoDataset_HDF5;
        HydroGeoDataset -->|Calibration (PSO)| HydroGeoDataset_HDF5;
        HydroGeoDataset -->|Sensitivity Analysis (Morris)| HydroGeoDataset_HDF5;
    end

    subgraph AI_System
        HydroGeoDataset_HDF5 --> "Vision System Deep Learning Framework (PyTorch)";
        HydroGeoDataset_HDF5 --> HydroGeoDataset;
        "Vision System Deep Learning Framework (PyTorch)" --> "Multi-AI Agents RAG System";
        HydroGeoDataset --> "Multi-AI Agents RAG System";
        "Multi-AI Agents RAG System" --> "Reports & Visualization";
    end

    subgraph Legends
        classDef developed fill:#B0C4DE,stroke:#000,stroke-width:1px;
        classDef existing fill:#87CEEB,stroke:#000,stroke-width:1px;
        classDef models fill:#ADFF2F,stroke:#000,stroke-width:1px;
        classDef database fill:#FFD700,stroke:#000,stroke-width:1px;
        classDef storage fill:#FF4500,stroke:#000,stroke-width:1px;

        HydroGeoDataset_HDF5:::storage;
        HydroGeoDataset:::storage;
        "SWAT+gwflow":::models;
        "SWAT+":::models;
        "MODFLOW-NWT":::models;
        QSWAT+:::existing;
        "SWAT+Editor":::existing;
        SWATGenX:::developed;
        MODGenX:::developed;
        "Multi-AI Agents RAG System":::developed;
    end
```
