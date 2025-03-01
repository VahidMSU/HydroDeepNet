from dataclasses import dataclass, field
import os

# Define different model tiers based on task complexity
MODELS = {
    "very_large" : "deepseek-r1:70b", # For very complex tasks
    "large": "deepseek-r1:7b",    # For complex reasoning and analysis
    "medium": "deepseek-r1:1.5b", # For moderate tasks
    "small": "mistral:latest",    # For simpler tasks
    "tiny": "llama2:latest"       # For basic tasks
}

@dataclass
class AgentConfig:
    county: str = None
    state: str = "Michigan"
    years: list = field(default_factory=lambda: [2010])
    analysis_type: str = "climate"
    focus: str = "pattern"
    
    # Model configuration by task complexity
    models: dict = field(default_factory=lambda: MODELS)
    
    # Task-specific model assignments
    query_analysis_model: str = MODELS["small"]     # Simple parsing, pattern recognition
    data_processing_model: str = MODELS["tiny"]     # Data manipulation, minimal reasoning
    analysis_model: str = MODELS["large"]           # Complex reasoning, expert knowledge
    synthesis_model: str = MODELS["medium"]         # Summarization, moderate complexity
    
    # Default model when specific task model isn't specified
    default_model: str = MODELS["medium"]
    
    # Paths
    USGS_governmental_path = "/data/SWATGenXApp/GenXAppData/USGS/GovernmentUnits_National_GDB/GovernmentUnits_National_GDB.gdb"
    HydroGeoDataset_ML_250_path = "/data/SWATGenXApp/GenXAppData/HydroGeoDataset/HydroGeoDataset_ML_250.h5"
    CDL_CODES_path = "/data/SWATGenXApp/GenXAppData/CDL/CDL_CODES.csv"
    PRISM_PATH = '/data/SWATGenXApp/GenXAppData/HydroGeoDataset/PRISM_ML_250m.h5'
    loca2_path = '/data/SWATGenXApp/GenXAppData/HydroGeoDataset/LOCA2_MLP.h5'
    