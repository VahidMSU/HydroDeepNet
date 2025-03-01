"""
Script to check available paths and data in the LOCA2 dataset.

This script helps diagnose issues with accessing LOCA2 climate data by
examining the structure of the HDF5 file and checking for available models,
scenarios, ensembles, and time periods.
"""

import os
import sys
import h5py
import numpy as np
from datetime import datetime

# Add project path to sys.path if needed
sys.path.append('/data/SWATGenXApp/codes')

LOCA2_PATH = '/data/SWATGenXApp/GenXAppData/HydroGeoDataset/LOCA2_MLP.h5'


def check_h5_structure(filepath=LOCA2_PATH, max_depth=3, max_items=10, start_path='/'):
    """
    Recursively explore and print the structure of an HDF5 file.
    
    Args:
        filepath: Path to the HDF5 file
        max_depth: Maximum recursion depth
        max_items: Maximum number of items to show per group
        start_path: Starting path in the HDF5 hierarchy
    """
    def _explore_group(name, obj, depth, path):
        if depth > max_depth:
            return
            
        indent = '  ' * depth
        if isinstance(obj, h5py.Group):
            print(f"{indent}GROUP: {name} at {path}/{name}")
            items = list(obj.keys())
            if len(items) > max_items:
                print(f"{indent}  Showing {max_items} of {len(items)} items...")
                items = items[:max_items] + ['...']
            
            for item_name in items:
                if item_name == '...':
                    print(f"{indent}  ...")
                    continue
                item_obj = obj.get(item_name)
                _explore_group(item_name, item_obj, depth + 1, f"{path}/{name}")
                
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}DATASET: {name}, Shape: {obj.shape}, Type: {obj.dtype}")

    try:
        with h5py.File(filepath, 'r') as f:
            print(f"Exploring HDF5 file: {filepath}")
            print(f"Starting from path: {start_path}")
            print("-" * 80)
            
            if start_path == '/':
                # Explore from root
                for name, obj in f.items():
                    _explore_group(name, obj, 0, '')
            else:
                # Explore from specific path
                if start_path in f:
                    group = f[start_path]
                    _explore_group(os.path.basename(start_path), group, 0, os.path.dirname(start_path))
                else:
                    print(f"Path {start_path} not found in the HDF5 file.")
    
    except Exception as e:
        print(f"Error exploring HDF5 file: {e}")


def check_available_periods(filepath=LOCA2_PATH):
    """
    Check available time periods for specific scenarios in the LOCA2 dataset.
    """
    try:
        if not os.path.exists(filepath):
            print(f"LOCA2 file not found at: {filepath}")
            return
            
        print(f"\nChecking time periods in LOCA2 dataset: {filepath}")
        
        with h5py.File(filepath, 'r') as f:
            # Check for SSP scenarios
            if 'e_n_cent' in f:
                e_n_cent = f['e_n_cent']
                
                # List each model
                for model in e_n_cent.keys():
                    model_path = e_n_cent[model]
                    ssp_scenarios = [s for s in model_path.keys() if s.startswith('ssp')]
                    
                    if ssp_scenarios:
                        print(f"\nModel: {model}")
                        print(f"  SSP scenarios: {ssp_scenarios}")
                        
                        # Check time periods for first SSP scenario
                        first_ssp = ssp_scenarios[0]
                        if not model_path[first_ssp]:
                            continue
                            
                        # Check first ensemble
                        ensembles = list(model_path[first_ssp].keys())
                        if not ensembles:
                            continue
                            
                        first_ensemble = ensembles[0]
                        ensemble_path = model_path[first_ssp][first_ensemble]
                        
                        # Check time steps
                        if 'daily' in ensemble_path:
                            time_periods = list(ensemble_path['daily'].keys())
                            print(f"  Time periods for {first_ssp}/daily: {time_periods}")

    except Exception as e:
        print(f"Error checking time periods: {e}")


def test_ssp245_access(model="ACCESS-CM2"):
    """
    Specifically test access to ssp245 scenario data.
    """
    try:
        print(f"\nTesting access to ssp245 scenario for model {model}")
        
        with h5py.File(LOCA2_PATH, 'r') as f:
            # Check if model exists
            if f'e_n_cent/{model}' not in f:
                print(f"Model {model} not found")
                return
                
            # Check if ssp245 scenario exists
            if f'e_n_cent/{model}/ssp245' not in f:
                print(f"Scenario ssp245 not found for model {model}")
                return
            
            # Get available ensembles
            ensembles = list(f[f'e_n_cent/{model}/ssp245'].keys())
            print(f"Available ensembles for {model}/ssp245: {ensembles}")
            
            # Check first ensemble
            first_ensemble = ensembles[0]
            
            # Check available time steps
            time_steps = list(f[f'e_n_cent/{model}/ssp245/{first_ensemble}'].keys())
            print(f"Available time steps: {time_steps}")
            
            # Check daily time step
            if 'daily' in time_steps:
                # Get available time periods
                time_periods = list(f[f'e_n_cent/{model}/ssp245/{first_ensemble}/daily'].keys())
                print(f"Available time periods for daily: {time_periods}")
                
                # Check each time period
                for period in time_periods:
                    # Get available variables
                    variables = list(f[f'e_n_cent/{model}/ssp245/{first_ensemble}/daily/{period}'].keys())
                    print(f"Variables for time period {period}: {variables}")
                    
                    # Check first variable
                    if variables:
                        var = variables[0]
                        data_shape = f[f'e_n_cent/{model}/ssp245/{first_ensemble}/daily/{period}/{var}'].shape
                        print(f"Shape of {var} data for period {period}: {data_shape}")
            else:
                print("Daily time step not found")
    
    except Exception as e:
        print(f"Error testing ssp245 access: {e}")


def main():
    """Main function to run the script."""
    print("LOCA2 Dataset Structure Analysis")
    print("=" * 40)
    
    if not os.path.exists(LOCA2_PATH):
        print(f"ERROR: LOCA2 file not found at {LOCA2_PATH}")
        sys.exit(1)
    
    # Check top-level structure
    print("\nChecking top-level structure:")
    with h5py.File(LOCA2_PATH, 'r') as f:
        print(f"Top-level keys: {list(f.keys())}")
        
        if 'e_n_cent' in f:
            print(f"\nModels in dataset: {len(list(f['e_n_cent'].keys()))}")
            print(f"First few models: {list(f['e_n_cent'].keys())[:5]}")
    
    # Check available time periods
    check_available_periods()
    
    # Test ssp245 access
    test_ssp245_access()
    
    print("\nRecommendations for LOCA2 data access:")
    print("1. For future scenarios, use specific time periods (2015_2044, 2045_2074, 2075_2100)")
    print("2. Verify the model/ensemble combination has data for your scenario")
    print("3. Use the check_h5_structure function to explore specific paths")
    
    print("\nExample call to explore a specific model:")
    print("check_h5_structure(start_path='e_n_cent/ACCESS-CM2')")


if __name__ == "__main__":
    main()
