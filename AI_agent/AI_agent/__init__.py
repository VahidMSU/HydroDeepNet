# Make core modules available for import
try:
    from .loca2_multi_period_multi_scenario import full_climate_change_data, analyze_climate_changes
    from .loca2_dataset import DataImporter, list_of_cc_models
    from .climate_change_analysis import ClimateChangeAnalysis
except ImportError as e:
    print(f"Error importing AI agent modules: {e}")
