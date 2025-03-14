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
        NWIS --> SWATGenX
        NHDPlusHR[NHDPlus HR] --> SWATGenX
        SWATGenX --> QSWATPlus[QSWAT+]
        SWATGenX --> SWATPlusEditor[SWAT+ Editor]
        WellInfo[Water well information (Wellogic)] --> EBK[Empirical Bayesian Kriging]
        EBK --> HydraulicProps[Hydraulic properties]
        HydraulicProps --> MODGenX
        MODGenX --> Flopy
        Flopy --> MODFLOWNWT[MODFLOW-NWT]
    end

    subgraph SWAT_Processing
        SWATPlus[SWAT+] -->|Streams| SWATPlusGwflow[SWAT+gwflow]
        SWATPlusGwflow --> HydroGeoDataset
    end

    subgraph Parallel_Processing
        HydroGeoDataset -->|Validation (ensemble)| HydroGeoDataset_HDF5
        HydroGeoDataset -->|Calibration (PSO)| HydroGeoDataset_HDF5
        HydroGeoDataset -->|Sensitivity Analysis (Morris)| HydroGeoDataset_HDF5
    end

    subgraph AI_System
        HydroGeoDataset_HDF5 --> VisionSystem[Vision System Deep Learning Framework (PyTorch)]
        HydroGeoDataset_HDF5 --> HydroGeoDataset
        VisionSystem --> MultiAI[Multi-AI Agents RAG System]
        HydroGeoDataset --> MultiAI
        MultiAI --> Reports[Reports & Visualization]
    end

    subgraph Legends
        classDef developed fill:#B0C4DE,stroke:#000,stroke-width:1px
        classDef existing fill:#87CEEB,stroke:#000,stroke-width:1px
        classDef models fill:#ADFF2F,stroke:#000,stroke-width:1px
        classDef database fill:#FFD700,stroke:#000,stroke-width:1px
        classDef storage fill:#FF4500,stroke:#000,stroke-width:1px

        HydroGeoDataset_HDF5:::storage
        HydroGeoDataset:::storage
        SWATPlusGwflow:::models
        SWATPlus:::models
        MODFLOWNWT:::models
        QSWATPlus:::existing
        SWATPlusEditor:::existing
        SWATGenX:::developed
        MODGenX:::developed
        MultiAI:::developed
    end
```