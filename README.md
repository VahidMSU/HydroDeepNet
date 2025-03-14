```mermaid
%%{init: {'theme': 'neutral', 'flowchart': {'curve': 'basis'}}}%%
graph TB
    title["HydroAI System: Hydrological Modeling, AI, and Multi-Agent Retrieval"]

    %% Data Sources
    PRISM["PRISM (Climate)"] -->|Climate Data| HydroGeoDataset[(HydroGeoDataset)]
    LOCA2["LOCA2 (Climate)"] -->|Climate Data| HydroGeoDataset
    NLCD["NLCD (Land Cover)"] -->|Land Cover| HydroGeoDataset
    NSRSDB["NSRSDB (Solar Radiation)"] -->|Solar Radiation| HydroGeoDataset
    gSSURGO["gSSURGO (Soil)"] -->|Soil Data| HydroGeoDataset
    USGSDEM["USGS DEM"] -->|Elevation Data| HydroGeoDataset
    SNODAS["SNODAS"] -->|Snow Data| HydroGeoDataset
    CDL["CDL (Crop Data)"] -->|Agricultural Data| HydroGeoDataset

    %% Hydrological & Groundwater Models
    SWATGenX{{SWATGenX}} --> QSWATPlus["QSWAT+"]
    SWATGenX --> SWATPlusEditor["SWAT+ Editor"]
    SWATPlus["SWAT+"] -->|Streams| SWATPlusGwflow["SWAT+gwflow"]
    SWATPlusGwflow --> HydroGeoDataset

    WellInfo["Water Well Info (Wellogic)"] --> EBK["Empirical Bayesian Kriging"]
    EBK --> HydraulicProps["Hydraulic Properties"]
    HydraulicProps --> MODGenX{{MODGenX}}
    MODGenX --> Flopy["Flopy"]
    Flopy --> MODFLOWNWT["MODFLOW-NWT"]
    MODFLOWNWT --> HydroGeoDataset

    %% AI System
    VisionSystem["Vision System (Deep Learning)"]
    HydroGeoDataset -->|Processed Data| VisionSystem
    VisionSystem -->|Predictions & Insights| AI_Output["AI Model Outputs"]

    %% Parallel Processing
    HydroGeoDataset -- "Validation (Ensemble)" --> HydroGeoDataset_HDF5[("HydroGeoDataset HDF5")]
    HydroGeoDataset -- "Calibration (PSO)" --> HydroGeoDataset_HDF5
    HydroGeoDataset -- "Sensitivity Analysis (Morris)" --> HydroGeoDataset_HDF5

    %% Multi-AI RAG System
    UserQuery["User Query Manager"] -->|Selects Report Type| ReportAggregator["📝 Report Aggregator"]
    MultiAI["Multi-AI Agents RAG System"] -->|Retrieves Data| ReportAggregator
    ReportAggregator --> Reports["📄 Final Reports"]

    MultiAI -->|Retrieves Structured Data| HydroGeoDataset
    MultiAI -->|Filters Data by User Request| ReportAggregator

    %% Potential Future Connections
    MultiAI -.-|Future Integration| AI_Output
    MultiAI -.-|Future Integration| SWATPlusGwflow

    %% Legend
    Legend(["**Legend**"])
    dev["Developed Component"]:::developed
    ex["Existing Tool"]:::existing
    mod["Model"]:::models
    db["Database"]:::database
    sto["Storage"]:::storage
    llm["LLM (AI Model)"]:::llm

    %% Styling
    classDef developed fill:#B0C4DE,stroke:#000,stroke-width:2px,rx:5,ry:5
    classDef existing fill:#87CEEB,stroke:#000,stroke-width:2px,rx:5,ry:5
    classDef models fill:#ADFF2F,stroke:#000,stroke-width:2px,rx:5,ry:5
    classDef database fill:#FFD700,stroke:#000,stroke-width:2px,rx:10,ry:10
    classDef storage fill:#FF4500,stroke:#000,stroke-width:2px,rx:10,ry:10
    classDef llm fill:#4682B4,stroke:#000,stroke-width:2px,rx:10,ry:10
    classDef title font-size:18px,fill:none,stroke:none

    %% Assign Classes to Nodes
    HydroGeoDataset_HDF5:::storage
    HydroGeoDataset:::storage
    SWATPlusGwflow:::models
    SWATPlus:::models
    MODFLOWNWT:::models
    QSWATPlus:::existing
    SWATPlusEditor:::existing
    SWATGenX:::developed
    MODGenX:::developed
    MultiAI:::llm
    Reports:::database
```