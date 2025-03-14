"""
Climate Change Analysis module for LOCA2 data.

This module provides functions for analyzing climate change projections
from LOCA2 datasets, comparing historical periods with future scenarios,
and generating statistics, visualizations, and reports.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import calendar
try:
    from AI_agent.loca2_multi_period_multi_scenario import (
        full_climate_change_data,
        analyze_climate_changes,
    )
    from AI_agent.loca2_dataset import DataImporter
except ImportError:
    try:
        from loca2_multi_period_multi_scenario import (
            full_climate_change_data,
            analyze_climate_changes,
        )
        from loca2_dataset import DataImporter
    except ImportError:
        # Try relative imports as last resort
        from .loca2_multi_period_multi_scenario import (
            full_climate_change_data,
            analyze_climate_changes,
        )
        from .loca2_dataset import DataImporter
# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Set matplotlib style for consistent visuals
plt.style.use('seaborn-v0_8-whitegrid')

class ClimateChangeAnalysis:
    """
    Class for analyzing climate change based on LOCA2 data.
    
    This class provides methods to compare historical climate data with future
    scenarios and analyze changes in temperature and precipitation patterns.
    """
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the climate change analysis with configuration.
        
        Args:
            config: Configuration dictionary with analysis parameters
        """
        self.config = config or {}
        self.historical_data = {}
        self.scenario_data = {}
        self.historical_means = {}
        self.scenario_means = {}
        self.deltas = {}
        self.data_importer = DataImporter(config)
        self.output_dir = self.config.get('output_dir', 'climate_change_results')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up default temporal resolution
        self.aggregation = self.config.get('aggregation', 'monthly')
        
    def extract_data(self, historical_config: Dict[str, Any], scenario_configs: List[Dict[str, Any]]) -> bool:
        """
        Extract historical and scenario climate data for analysis.
        
        Args:
            historical_config: Configuration for historical data extraction
            scenario_configs: List of configurations for scenario data extraction
        
        Returns:
            Boolean indicating success
        """
        # Extract historical data
        hist_start_year = historical_config.get('start_year', 2000) 
        hist_end_year = historical_config.get('end_year', 2014)
        hist_model = historical_config.get('model', 'ACCESS-CM2')
        hist_ensemble = historical_config.get('ensemble', 'r1i1p1f1')
        
        logger.info(f"Extracting historical data: {hist_model}, {hist_start_year}-{hist_end_year}")
        
        hist_pr, hist_tmax, hist_tmin = self.data_importer.LOCA2(
            start_year=hist_start_year,
            end_year=hist_end_year,
            cc_model=hist_model,
            scenario='historical',
            ensemble=hist_ensemble,
            cc_time_step='daily'
        )
        
        # Store historical data
        self.historical_data = {
            'pr': hist_pr,
            'tmax': hist_tmax,
            'tmin': hist_tmin,
            'config': {
                'start_year': hist_start_year,
                'end_year': hist_end_year,
                'model': hist_model,
                'ensemble': hist_ensemble
            }
        }
        
        logger.info(f"Historical data extracted with shapes: PR {hist_pr.shape}, TMAX {hist_tmax.shape}, TMIN {hist_tmin.shape}")
        
        # Extract scenario data for each configuration
        for scenario_config in scenario_configs:
            scenario_name = scenario_config.get('name', 'ssp245')
            scen_start_year = scenario_config.get('start_year', 2050)
            scen_end_year = scenario_config.get('end_year', 2070)
            scen_model = scenario_config.get('model', hist_model)
            scen_ensemble = scenario_config.get('ensemble', hist_ensemble)
            
            logger.info(f"Extracting {scenario_name} data: {scen_model}, {scen_start_year}-{scen_end_year}")
            
            # Determine which time period(s) to use based on requested years
            time_periods = []
            if scen_start_year >= 2015 and scen_start_year <= 2044:
                time_periods.append("2015_2044")
            if (scen_start_year <= 2044 and scen_end_year >= 2045) or (scen_start_year >= 2045 and scen_start_year <= 2074):
                time_periods.append("2045_2074")
            if scen_end_year >= 2075:
                time_periods.append("2075_2100")
            
            logger.info(f"Using time periods: {time_periods} for years {scen_start_year}-{scen_end_year}")
            
            # Try each variant of the scenario name
            scenario_variants = [
                scenario_name,                   # Standard naming (e.g., 'ssp245')
                scenario_name.replace('ssp', 'ssp-'),  # Split format (e.g., 'ssp-245')
                # Add dashes between numbers if not present
                ''.join(['ssp'] + ['-' + c if c.isdigit() and i > 0 
                                 and scenario_name[3:][i-1].isdigit() 
                                 else c 
                                 for i, c in enumerate(scenario_name[3:])])
            ]
            
            attempted_variants = []
            data_pieces = []
            
            for variant in scenario_variants:
                attempted_variants.append(variant)
                variant_success = True
                period_data = []
                
                for time_period in time_periods:
                    logger.info(f"Attempting to extract {variant} data for period {time_period}")
                    period_start = int(time_period.split('_')[0])
                    period_end = int(time_period.split('_')[1])
                    
                    # Adjust years to fit within the period
                    extract_start = max(scen_start_year, period_start)
                    extract_end = min(scen_end_year, period_end)
                    
                    if extract_start > extract_end:
                        logger.info(f"Skipping period {time_period} as it doesn't overlap with requested years")
                        continue
                    
                    logger.info(f"Extracting years {extract_start}-{extract_end} from period {time_period}")
                    
                    # Extract the data for this time period
                    pr, tmax, tmin = self.data_importer.LOCA2(
                        start_year=extract_start,
                        end_year=extract_end,
                        cc_model=scen_model,
                        scenario=variant,
                        ensemble=scen_ensemble,
                        cc_time_step='daily',
                        time_period=time_period  # Pass the specific time period
                    )
                    
                    period_data.append((pr, tmax, tmin))
                    logger.info(f"Successfully extracted {variant} data for period {time_period}, shapes: {pr.shape}, {tmax.shape}, {tmin.shape}")
                
                if period_data:
                    # If we successfully got data for all required periods
                    data_pieces = period_data
                    logger.info(f"Successfully extracted all data using variant '{variant}'")
                    break
            
            # Combine data from different time periods if needed
            if len(data_pieces) == 1:
                scen_pr, scen_tmax, scen_tmin = data_pieces[0]
            else:
                # Concatenate along the time axis
                scen_pr = np.concatenate([piece[0] for piece in data_pieces], axis=0)
                scen_tmax = np.concatenate([piece[1] for piece in data_pieces], axis=0)
                scen_tmin = np.concatenate([piece[2] for piece in data_pieces], axis=0)
            
            # Store scenario data
            self.scenario_data[scenario_name] = {
                'pr': scen_pr,
                'tmax': scen_tmax,
                'tmin': scen_tmin,
                'config': {
                    'start_year': scen_start_year,
                    'end_year': scen_end_year,
                    'model': scen_model,
                    'ensemble': scen_ensemble
                }
            }
            
            logger.info(f"Scenario {scenario_name} data extracted with final shapes: PR {scen_pr.shape}, TMAX {scen_tmax.shape}, TMIN {scen_tmin.shape}")
        
        self._calculate_spatial_means()
        
        return len(self.scenario_data) > 0
    
    def _calculate_spatial_means(self):
        """Calculate spatial means across all datasets for time series analysis."""
        # Calculate historical means
        if self.historical_data:
            self.historical_means = {
                'pr': np.nanmean(self.historical_data['pr'].astype(float), axis=(1, 2)),
                'tmax': np.nanmean(self.historical_data['tmax'].astype(float), axis=(1, 2)),
                'tmin': np.nanmean(self.historical_data['tmin'].astype(float), axis=(1, 2))
            }
            
            # Add tmean
            if 'tmax' in self.historical_means and 'tmin' in self.historical_means:
                self.historical_means['tmean'] = (self.historical_means['tmax'] + self.historical_means['tmin']) / 2
        
        # Calculate scenario means
        for scenario_name, scenario_data in self.scenario_data.items():
            self.scenario_means[scenario_name] = {
                'pr': np.nanmean(scenario_data['pr'].astype(float), axis=(1, 2)),
                'tmax': np.nanmean(scenario_data['tmax'].astype(float), axis=(1, 2)),
                'tmin': np.nanmean(scenario_data['tmin'].astype(float), axis=(1, 2))
            }
            
            # Add tmean
            if 'tmax' in self.scenario_means[scenario_name] and 'tmin' in self.scenario_means[scenario_name]:
                self.scenario_means[scenario_name]['tmean'] = (
                    self.scenario_means[scenario_name]['tmax'] + 
                    self.scenario_means[scenario_name]['tmin']
                ) / 2
    
    def calculate_climate_change_metrics(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Calculate climate change metrics between historical and scenario periods.
        
        Returns:
            Dictionary with climate change metrics for each scenario and variable
        """
        if not self.historical_means or not self.scenario_means:
            logger.warning("Cannot calculate metrics without data")
            return {}
        
        self.deltas = {}
        
        for scenario_name, scenario_means in self.scenario_means.items():
            self.deltas[scenario_name] = {}
            
            for var_name in ['pr', 'tmax', 'tmin', 'tmean']:
                if var_name in self.historical_means and var_name in scenario_means:
                    hist_data = self.historical_means[var_name]
                    scen_data = scenario_means[var_name]
                    
                    # Calculate basic statistics
                    hist_mean = np.nanmean(hist_data)
                    scen_mean = np.nanmean(scen_data)
                    abs_change = scen_mean - hist_mean
                    
                    if abs(hist_mean) > 1e-10:  # Avoid division by zero
                        pct_change = (abs_change / hist_mean) * 100
                    else:
                        pct_change = np.nan
                    
                    # Calculate variability metrics
                    hist_std = np.nanstd(hist_data)
                    scen_std = np.nanstd(scen_data)
                    variability_change = scen_std - hist_std
                    
                    if abs(hist_std) > 1e-10:
                        variability_pct = (variability_change / hist_std) * 100
                    else:
                        variability_pct = np.nan
                    
                    # Calculate extremes
                    hist_p90 = np.nanpercentile(hist_data, 90)
                    hist_p10 = np.nanpercentile(hist_data, 10)
                    scen_p90 = np.nanpercentile(scen_data, 90)
                    scen_p10 = np.nanpercentile(scen_data, 10)
                    
                    p90_change = scen_p90 - hist_p90
                    p10_change = scen_p10 - hist_p10
                    
                    # NEW: Analyze extreme events
                    extreme_metrics = {}
                    
                    if var_name == 'pr':
                        # For precipitation: heavy rainfall days and consecutive dry days
                        # Determine thresholds based on historical data
                        heavy_rain_threshold = np.nanpercentile(hist_data, 95)  # 95th percentile
                        
                        # Count days exceeding threshold
                        hist_heavy_rain_days = np.sum(hist_data > heavy_rain_threshold) / len(hist_data) * 100
                        scen_heavy_rain_days = np.sum(scen_data > heavy_rain_threshold) / len(scen_data) * 100
                        
                        # Analyze dry spells (consecutive days with precipitation below a threshold)
                        # Note: This is approximate using spatial means, more accurate calculation
                        # would use complete daily precipitation data
                        dry_day_threshold = 1.0  # mm/day
                        
                        # Calculate consecutive dry days (simplified due to using spatial means)
                        hist_dry_days = np.where(hist_data < dry_day_threshold)[0]
                        scen_dry_days = np.where(scen_data < dry_day_threshold)[0]
                        
                        hist_dry_spells = self._count_consecutive_events(hist_dry_days)
                        scen_dry_spells = self._count_consecutive_events(scen_dry_days)
                        
                        # Store extreme precipitation metrics
                        extreme_metrics = {
                            'heavy_rain_threshold': heavy_rain_threshold,
                            'hist_heavy_rain_percent': hist_heavy_rain_days,
                            'scen_heavy_rain_percent': scen_heavy_rain_days,
                            'heavy_rain_percent_change': scen_heavy_rain_days - hist_heavy_rain_days,
                            'hist_max_dry_spell': max(hist_dry_spells) if hist_dry_spells else 0,
                            'scen_max_dry_spell': max(scen_dry_spells) if scen_dry_spells else 0,
                            'dry_spell_change': (max(scen_dry_spells) if scen_dry_spells else 0) - 
                                              (max(hist_dry_spells) if hist_dry_spells else 0),
                            'hist_mean_dry_spell': np.mean(hist_dry_spells) if hist_dry_spells else 0,
                            'scen_mean_dry_spell': np.mean(scen_dry_spells) if scen_dry_spells else 0
                        }
                        
                    elif var_name in ['tmax', 'tmin', 'tmean']:
                        # For temperature: heat waves and cold spells
                        if var_name == 'tmax':
                            # Heat wave threshold (95th percentile of historical max temp)
                            heat_threshold = np.nanpercentile(hist_data, 95)
                            hist_hot_days = np.sum(hist_data > heat_threshold) / len(hist_data) * 100
                            scen_hot_days = np.sum(scen_data > heat_threshold) / len(scen_data) * 100
                            
                            # Find consecutive hot days (heat waves)
                            hist_hot_day_indices = np.where(hist_data > heat_threshold)[0]
                            scen_hot_day_indices = np.where(scen_data > heat_threshold)[0]
                            
                            hist_heat_waves = self._count_consecutive_events(hist_hot_day_indices)
                            scen_heat_waves = self._count_consecutive_events(scen_hot_day_indices)
                            
                            extreme_metrics = {
                                'heat_threshold': heat_threshold,
                                'hist_hot_days_percent': hist_hot_days,
                                'scen_hot_days_percent': scen_hot_days,
                                'hot_days_percent_change': scen_hot_days - hist_hot_days,
                                'hist_max_heat_wave': max(hist_heat_waves) if hist_heat_waves else 0,
                                'scen_max_heat_wave': max(scen_heat_waves) if scen_heat_waves else 0,
                                'heat_wave_change': (max(scen_heat_waves) if scen_heat_waves else 0) - 
                                                  (max(hist_heat_waves) if hist_heat_waves else 0),
                                'hist_mean_heat_wave': np.mean(hist_heat_waves) if hist_heat_waves else 0,
                                'scen_mean_heat_wave': np.mean(scen_heat_waves) if scen_heat_waves else 0
                            }
                        
                        elif var_name == 'tmin':
                            # Cold spell threshold (5th percentile of historical min temp)
                            cold_threshold = np.nanpercentile(hist_data, 5)
                            hist_cold_days = np.sum(hist_data < cold_threshold) / len(hist_data) * 100
                            scen_cold_days = np.sum(scen_data < cold_threshold) / len(scen_data) * 100
                            
                            # Find consecutive cold days (cold spells)
                            hist_cold_day_indices = np.where(hist_data < cold_threshold)[0]
                            scen_cold_day_indices = np.where(scen_data < cold_threshold)[0]
                            
                            hist_cold_spells = self._count_consecutive_events(hist_cold_day_indices)
                            scen_cold_spells = self._count_consecutive_events(scen_cold_day_indices)
                            
                            extreme_metrics = {
                                'cold_threshold': cold_threshold,
                                'hist_cold_days_percent': hist_cold_days,
                                'scen_cold_days_percent': scen_cold_days,
                                'cold_days_percent_change': scen_cold_days - hist_cold_days,
                                'hist_max_cold_spell': max(hist_cold_spells) if hist_cold_spells else 0,
                                'scen_max_cold_spell': max(scen_cold_spells) if scen_cold_spells else 0,
                                'cold_spell_change': (max(scen_cold_spells) if scen_cold_spells else 0) - 
                                                   (max(hist_cold_spells) if hist_cold_spells else 0),
                                'hist_mean_cold_spell': np.mean(hist_cold_spells) if hist_cold_spells else 0,
                                'scen_mean_cold_spell': np.mean(scen_cold_spells) if scen_cold_spells else 0
                            }
                    
                    # Store the metrics
                    self.deltas[scenario_name][var_name] = {
                        'historical_mean': hist_mean,
                        'scenario_mean': scen_mean,
                        'absolute_change': abs_change,
                        'percent_change': pct_change,
                        'historical_std': hist_std,
                        'scenario_std': scen_std,
                        'variability_change': variability_change,
                        'variability_pct_change': variability_pct,
                        'historical_p90': hist_p90,
                        'scenario_p90': scen_p90,
                        'p90_change': p90_change,
                        'historical_p10': hist_p10,
                        'scenario_p10': scen_p10, 
                        'p10_change': p10_change,
                        'extreme_events': extreme_metrics  # Add extreme event metrics
                    }
        
        return self.deltas
    
    def _count_consecutive_events(self, indices):
        """
        Helper function to count consecutive events from indices.
        
        Args:
            indices: Array of indices where events occur
            
        Returns:
            List of consecutive event durations
        """
        if len(indices) == 0:
            return []
            
        # Sort indices to ensure proper sequence
        indices = np.sort(indices)
        
        # Calculate differences between consecutive indices
        diffs = np.diff(indices)
        
        # Find breaks in consecutive sequences (where diff > 1)
        breaks = np.where(diffs > 1)[0]
        
        # Prepare list to store lengths of consecutive sequences
        consecutive_lengths = []
        
        # Calculate lengths of consecutive sequences
        prev_break = -1
        for brk in breaks:
            consecutive_lengths.append(indices[brk] - indices[prev_break+1] + 1)
            prev_break = brk
            
        # Add the last sequence
        consecutive_lengths.append(indices[-1] - indices[prev_break+1] + 1)
        
        return consecutive_lengths
    
    def plot_extreme_events_comparison(self, output_path: Optional[str] = None, 
                                     figsize: Tuple[int, int] = (12, 10)) -> Optional[plt.Figure]:
        """
        Create visualizations comparing extreme climate events between historical and future periods.
        
        Args:
            output_path: Path to save the figure (if None, uses default path)
            figsize: Figure size as tuple (width, height)
            
        Returns:
            Matplotlib Figure object or None if error occurs
        """
        if not self.deltas:
            logger.warning("Cannot generate extreme event comparison without metrics")
            return None
            
        # Set default output path if not provided
        if output_path is None:
            output_path = os.path.join(self.output_dir, "extreme_events_comparison.png")
            
        # Create figure with 2x2 subplots for extreme event comparisons
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Colors for scenarios
        colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']
        
        # 1. Heavy Rainfall Days (Precipitation)
        ax0 = axes[0]
        
        # Prepare data for plotting
        scenarios = []
        hist_heavy_rain = []
        scen_heavy_rain = []
        
        for i, (scenario_name, metrics) in enumerate(self.deltas.items()):
            if 'pr' in metrics and 'extreme_events' in metrics['pr']:
                scenarios.append(scenario_name)
                extremes = metrics['pr']['extreme_events']
                hist_heavy_rain.append(extremes.get('hist_heavy_rain_percent', 0))
                scen_heavy_rain.append(extremes.get('scen_heavy_rain_percent', 0))
        
        if scenarios:
            x = np.arange(len(scenarios))
            width = 0.35
            
            ax0.bar(x - width/2, hist_heavy_rain, width, label='Historical', color='gray')
            ax0.bar(x + width/2, scen_heavy_rain, width, label='Future', color='tab:red')
            
            ax0.set_title('Heavy Rainfall Days', fontsize=12)
            ax0.set_xlabel('Scenario')
            ax0.set_ylabel('% of Days > 95th Percentile')
            ax0.set_xticks(x)
            ax0.set_xticklabels(scenarios)
            ax0.legend(loc='best')
            ax0.grid(True, linestyle='--', alpha=0.7)
            
        # 2. Maximum Dry Spells (Precipitation)
        ax1 = axes[1]
        
        # Prepare data for plotting
        hist_dry_spells = []
        scen_dry_spells = []
        
        for i, (scenario_name, metrics) in enumerate(self.deltas.items()):
            if 'pr' in metrics and 'extreme_events' in metrics['pr']:
                extremes = metrics['pr']['extreme_events']
                hist_dry_spells.append(extremes.get('hist_max_dry_spell', 0))
                scen_dry_spells.append(extremes.get('scen_max_dry_spell', 0))
        
        if scenarios:
            ax1.bar(x - width/2, hist_dry_spells, width, label='Historical', color='gray')
            ax1.bar(x + width/2, scen_dry_spells, width, label='Future', color='tab:brown')
            
            ax1.set_title('Maximum Dry Spell Duration', fontsize=12)
            ax1.set_xlabel('Scenario')
            ax1.set_ylabel('Consecutive Days')
            ax1.set_xticks(x)
            ax1.set_xticklabels(scenarios)
            ax1.legend(loc='best')
            ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 3. Heat Wave Duration (Tmax)
        ax2 = axes[2]
        
        # Prepare data for plotting
        scenarios = []
        hist_heat_waves = []
        scen_heat_waves = []
        
        for i, (scenario_name, metrics) in enumerate(self.deltas.items()):
            if 'tmax' in metrics and 'extreme_events' in metrics['tmax']:
                scenarios.append(scenario_name)
                extremes = metrics['tmax']['extreme_events']
                hist_heat_waves.append(extremes.get('hist_max_heat_wave', 0))
                scen_heat_waves.append(extremes.get('scen_max_heat_wave', 0))
        
        if scenarios:
            x = np.arange(len(scenarios))
            
            ax2.bar(x - width/2, hist_heat_waves, width, label='Historical', color='gray')
            ax2.bar(x + width/2, scen_heat_waves, width, label='Future', color='tab:red')
            
            ax2.set_title('Maximum Heat Wave Duration', fontsize=12)
            ax2.set_xlabel('Scenario')
            ax2.set_ylabel('Consecutive Hot Days')
            ax2.set_xticks(x)
            ax2.set_xticklabels(scenarios)
            ax2.legend(loc='best')
            ax2.grid(True, linestyle='--', alpha=0.7)
        
        # 4. Cold Spell Duration (Tmin)
        ax3 = axes[3]
        
        # Prepare data for plotting
        scenarios = []
        hist_cold_spells = []
        scen_cold_spells = []
        
        for i, (scenario_name, metrics) in enumerate(self.deltas.items()):
            if 'tmin' in metrics and 'extreme_events' in metrics['tmin']:
                scenarios.append(scenario_name)
                extremes = metrics['tmin']['extreme_events']
                hist_cold_spells.append(extremes.get('hist_max_cold_spell', 0))
                scen_cold_spells.append(extremes.get('scen_max_cold_spell', 0))
        
        if scenarios:
            x = np.arange(len(scenarios))
            
            ax3.bar(x - width/2, hist_cold_spells, width, label='Historical', color='gray')
            ax3.bar(x + width/2, scen_cold_spells, width, label='Future', color='tab:blue')
            
            ax3.set_title('Maximum Cold Spell Duration', fontsize=12)
            ax3.set_xlabel('Scenario')
            ax3.set_ylabel('Consecutive Cold Days')
            ax3.set_xticks(x)
            ax3.set_xticklabels(scenarios)
            ax3.legend(loc='best')
            ax3.grid(True, linestyle='--', alpha=0.7)
        
        plt.suptitle('Extreme Climate Events: Historical vs Future', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save the figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Extreme events comparison saved to {output_path}")
        
        return fig

    def plot_timeseries_comparison(self, 
                                 output_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (12, 15)) -> Optional[plt.Figure]:
        """
        Create time series plots comparing historical and future climate variables.
        
        Args:
            output_path: Path to save the figure (if None, uses default path)
            figsize: Figure size as tuple (width, height)
            
        Returns:
            Matplotlib Figure object or None if error occurs
        """
        if not self.historical_means or not self.scenario_means:
            logger.warning("Cannot generate time series without data")
            return None
        
        # Set default output path if not provided
        if output_path is None:
            output_path = os.path.join(self.output_dir, "climate_timeseries_comparison.png")
        
        # Get historical years
        hist_start = self.historical_data['config']['start_year']
        hist_end = self.historical_data['config']['end_year']
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=False, 
                                gridspec_kw={'height_ratios': [1, 1, 1.2]})
        
        # Create x axes for plotting
        hist_years = np.arange(hist_start, hist_end + 1)
        
        if self.aggregation == 'monthly':
            # For monthly data, create continuous timeline
            hist_months_total = (hist_end - hist_start + 1) * 12
            hist_x = np.linspace(hist_start, hist_end + 1, hist_months_total)
        else:
            # For annual data
            hist_x = hist_years
        
        # Scenario info for plotting
        scenario_x = {}
        for scenario_name, scenario_data in self.scenario_data.items():
            scen_start = scenario_data['config']['start_year']
            scen_end = scenario_data['config']['end_year']
            
            if self.aggregation == 'monthly':
                # For monthly data, create continuous timeline
                scen_months_total = (scen_end - scen_start + 1) * 12
                scenario_x[scenario_name] = np.linspace(scen_start, scen_end + 1, scen_months_total)
            else:
                # For annual data
                scenario_x[scenario_name] = np.arange(scen_start, scen_end + 1)
        
        # Plot Max Temperature
        ax_tmax = axes[0]
        
        # Plot historical data
        if 'tmax' in self.historical_means:
            hist_data = self.historical_means['tmax']
            hist_x_plot = hist_x[:len(hist_data)]
            ax_tmax.plot(hist_x_plot, hist_data, 'k-', 
                       label='Historical', linewidth=1.5, alpha=0.8)
            
            # Add historical range
            hist_mean = np.mean(hist_data)
            hist_std = np.std(hist_data)
            ax_tmax.axhspan(hist_mean - hist_std, hist_mean + hist_std, 
                          color='gray', alpha=0.2, label='Historical Range (±1σ)')
        
        # Plot each scenario
        colors = ['r', 'b', 'g', 'm', 'c']
        for i, (scenario_name, scenario_means) in enumerate(self.scenario_means.items()):
            if 'tmax' in scenario_means:
                scen_data = scenario_means['tmax']
                scen_x_plot = scenario_x[scenario_name][:len(scen_data)]
                color = colors[i % len(colors)]
                ax_tmax.plot(scen_x_plot, scen_data, color=color, linestyle='-',
                           label=f'{scenario_name}', linewidth=1.5, alpha=0.8)
        
        ax_tmax.set_ylabel('Max Temperature (°C)', fontsize=12)
        ax_tmax.set_title('Maximum Temperature Comparison', fontsize=14)
        ax_tmax.grid(True, linestyle='--', alpha=0.6)
        ax_tmax.legend(loc='upper left', fontsize=10)
        
        # Plot Min Temperature
        ax_tmin = axes[1]
        
        # Plot historical data
        if 'tmin' in self.historical_means:
            hist_data = self.historical_means['tmin']
            hist_x_plot = hist_x[:len(hist_data)]
            ax_tmin.plot(hist_x_plot, hist_data, 'k-', 
                       label='Historical', linewidth=1.5, alpha=0.8)
            
            # Add historical range
            hist_mean = np.mean(hist_data)
            hist_std = np.std(hist_data)
            ax_tmin.axhspan(hist_mean - hist_std, hist_mean + hist_std, 
                          color='gray', alpha=0.2, label='Historical Range (±1σ)')
        
        # Plot each scenario
        for i, (scenario_name, scenario_means) in enumerate(self.scenario_means.items()):
            if 'tmin' in scenario_means:
                scen_data = scenario_means['tmin']
                scen_x_plot = scenario_x[scenario_name][:len(scen_data)]
                color = colors[i % len(colors)]
                ax_tmin.plot(scen_x_plot, scen_data, color=color, linestyle='-',
                           label=f'{scenario_name}', linewidth=1.5, alpha=0.8)
        
        ax_tmin.set_ylabel('Min Temperature (°C)', fontsize=12)
        ax_tmin.set_title('Minimum Temperature Comparison', fontsize=14)
        ax_tmin.grid(True, linestyle='--', alpha=0.6)
        ax_tmin.legend(loc='upper left', fontsize=10)
        
        # Plot Precipitation
        ax_pr = axes[2]
        
        # Plot historical data
        if 'pr' in self.historical_means:
            hist_data = self.historical_means['pr']
            hist_x_plot = hist_x[:len(hist_data)]
            ax_pr.plot(hist_x_plot, hist_data, 'k-', 
                     label='Historical', linewidth=1.5, alpha=0.8)
            
            # Add historical range
            hist_mean = np.mean(hist_data)
            hist_std = np.std(hist_data)
            ax_pr.axhspan(hist_mean - hist_std, hist_mean + hist_std, 
                        color='gray', alpha=0.2, label='Historical Range (±1σ)')
        
        # Plot each scenario
        for i, (scenario_name, scenario_means) in enumerate(self.scenario_means.items()):
            if 'pr' in scenario_means:
                scen_data = scenario_means['pr']
                scen_x_plot = scenario_x[scenario_name][:len(scen_data)]
                color = colors[i % len(colors)]
                ax_pr.plot(scen_x_plot, scen_data, color=color, linestyle='-',
                         label=f'{scenario_name}', linewidth=1.5, alpha=0.8)
        
        ax_pr.set_ylabel('Precipitation (mm/day)', fontsize=12)
        ax_pr.set_title('Precipitation Comparison', fontsize=14)
        ax_pr.grid(True, linestyle='--', alpha=0.6)
        ax_pr.legend(loc='upper left', fontsize=10)
        
        # Overall title and layout
        plt.suptitle('Climate Change Comparison: Historical vs Future Scenarios', 
                   fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save the plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Time series comparison saved to {output_path}")
        
        return fig
    
    def plot_seasonal_cycle_comparison(self,
                                     variable: str = 'pr',
                                     scenario_name: Optional[str] = None,
                                     output_path: Optional[str] = None,
                                     figsize: Tuple[int, int] = (10, 7)) -> Optional[plt.Figure]:
        """
        Create a seasonal cycle comparison plot for a specified climate variable.
        
        Args:
            variable: Climate variable to plot ('pr', 'tmax', 'tmin', or 'tmean')
            scenario_name: Name of scenario to compare (if None, uses first available)
            output_path: Path to save the figure (if None, uses default path)
            figsize: Figure size as tuple (width, height)
            
        Returns:
            Matplotlib Figure object or None if error occurs
        """
        if not self.historical_means or not self.scenario_means:
            logger.warning("Cannot generate seasonal cycle without data")
            return None
        
        # Only monthly aggregation makes sense for seasonal cycle
        if self.aggregation != 'monthly':
            logger.warning("Seasonal cycle requires monthly data aggregation")
            return None
            
        # Set default output path if not provided
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"seasonal_cycle_{variable}.png")
        
        # If no scenario specified, use the first one
        if scenario_name is None and self.scenario_means:
            scenario_name = list(self.scenario_means.keys())[0]
            
        if scenario_name not in self.scenario_means:
            logger.warning(f"Scenario {scenario_name} not found")
            return None
            
        if variable not in self.historical_means:
            logger.warning(f"Variable {variable} not found in historical data")
            return None
            
        if variable not in self.scenario_means[scenario_name]:
            logger.warning(f"Variable {variable} not found in scenario data")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get historical data and calculate monthly averages
        hist_data = self.historical_means[variable]
        hist_months_per_year = 12
        num_years = len(hist_data) // hist_months_per_year
        
        if num_years < 1:
            logger.warning("Not enough historical data for seasonal cycle")
            return None
            
        # Reshape to [years, months] and calculate monthly means
        hist_monthly = np.zeros(hist_months_per_year)
        for i in range(len(hist_data)):
            month_idx = i % hist_months_per_year
            hist_monthly[month_idx] += hist_data[i]
            
        # Divide by number of years to get average
        hist_monthly /= num_years
        
        # Get scenario data and calculate monthly averages
        scen_data = self.scenario_means[scenario_name][variable]
        scen_months_per_year = 12
        scen_num_years = len(scen_data) // scen_months_per_year
        
        if scen_num_years < 1:
            logger.warning("Not enough scenario data for seasonal cycle")
            return None
            
        # Reshape to [years, months] and calculate monthly means
        scen_monthly = np.zeros(scen_months_per_year)
        for i in range(len(scen_data)):
            month_idx = i % scen_months_per_year
            scen_monthly[month_idx] += scen_data[i]
            
        # Divide by number of years to get average
        scen_monthly /= scen_num_years
        
        # Setup x-axis (months)
        months = np.arange(1, 13)
        month_names = [calendar.month_abbr[m] for m in months]
        
        # Plot historical data
        ax.plot(months, hist_monthly, 'ko-', label='Historical', linewidth=2, markersize=6)
        
        # Plot scenario data
        ax.plot(months, scen_monthly, 'ro-', label=scenario_name, linewidth=2, markersize=6)
        
        # Calculate and plot difference if requested
        ax_twin = ax.twinx()
        diff = scen_monthly - hist_monthly
        ax_twin.bar(months, diff, alpha=0.3, color='blue', label='Difference')
        ax_twin.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel('Month', fontsize=12)
        
        # Variable-specific settings
        if variable == 'pr':
            ax.set_ylabel('Precipitation (mm/day)', fontsize=12)
            ax_twin.set_ylabel('Difference (mm/day)', fontsize=12, color='blue')
            title = 'Precipitation Seasonal Cycle'
        elif variable in ['tmax', 'tmin', 'tmean']:
            label = 'Maximum' if variable == 'tmax' else 'Minimum' if variable == 'tmin' else 'Mean'
            ax.set_ylabel(f'{label} Temperature (°C)', fontsize=12)
            ax_twin.set_ylabel('Difference (°C)', fontsize=12, color='blue')
            title = f'{label} Temperature Seasonal Cycle'
        else:
            ax.set_ylabel(variable, fontsize=12)
            ax_twin.set_ylabel(f'Difference', fontsize=12, color='blue')
            title = f'{variable} Seasonal Cycle'
        
        ax.set_title(f'{title}: Historical vs {scenario_name}', fontsize=14)
        
        # Set x-ticks to month names
        ax.set_xticks(months)
        ax.set_xticklabels(month_names)
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.6)
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc='best')
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Seasonal cycle comparison saved to {output_path}")
        
        return fig
    
    def plot_spatial_change_maps(self,
                               variable: str = 'tmean',
                               scenario_name: Optional[str] = None,
                               output_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (15, 5),
                               export_data: bool = False) -> Optional[plt.Figure]:
        """
        Create spatial maps showing climate change between historical and future periods.
        
        Args:
            variable: Climate variable to plot ('pr', 'tmax', 'tmin', or 'tmean')
            scenario_name: Name of scenario to compare (if None, uses first available)
            output_path: Path to save the figure (if None, uses default path)
            figsize: Figure size as tuple (width, height)
            export_data: Whether to export the spatial data as GeoTIFF or CSV
            
        Returns:
            Matplotlib Figure object or None if error occurs
        """
        if not self.historical_data or not self.scenario_data:
            logger.warning("Cannot generate spatial maps without data")
            return None
            
        # Set default output path if not provided
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"spatial_change_{variable}.png")
            
        # If no scenario specified, use the first one
        if scenario_name is None and self.scenario_data:
            scenario_name = list(self.scenario_data.keys())[0]
            
        if scenario_name not in self.scenario_data:
            logger.warning(f"Scenario {scenario_name} not found")
            return None
        
        # For tmean, calculate from tmax and tmin
        if variable == 'tmean':
            if 'tmax' not in self.historical_data or 'tmin' not in self.historical_data:
                logger.warning("Cannot calculate historical tmean without tmax and tmin")
                return None
                
            if 'tmax' not in self.scenario_data[scenario_name] or 'tmin' not in self.scenario_data[scenario_name]:
                logger.warning("Cannot calculate scenario tmean without tmax and tmin")
                return None
                
            hist_data = (self.historical_data['tmax'] + self.historical_data['tmin']) / 2
            scen_data = (self.scenario_data[scenario_name]['tmax'] + self.scenario_data[scenario_name]['tmin']) / 2
        else:
            if variable not in self.historical_data or variable not in self.scenario_data[scenario_name]:
                logger.warning(f"Variable {variable} not found in both datasets")
                return None
                
            hist_data = self.historical_data[variable]
            scen_data = self.scenario_data[scenario_name][variable]
        
        # Calculate temporal means for both datasets
        hist_mean = np.nanmean(hist_data, axis=0)
        scen_mean = np.nanmean(scen_data, axis=0)
        
        # Calculate the difference
        diff = scen_mean - hist_mean
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Determine appropriate colormaps and value limits
        if variable == 'pr':
            cmap_var = 'Blues'
            cmap_diff = 'BrBG'  # Brown-Blue-Green (brown for drying, green for wetting)
            title = 'Precipitation'
            units = 'mm/day'
            # Use percentiles to avoid outliers
            vmin_hist = np.nanpercentile(hist_mean, 2)
            vmax_hist = np.nanpercentile(hist_mean, 98)
            vmin_scen = np.nanpercentile(scen_mean, 2)
            vmax_scen = np.nanpercentile(scen_mean, 98)
            
            # For difference, use symmetric limits
            abs_max_diff = max(abs(np.nanpercentile(diff, 2)), abs(np.nanpercentile(diff, 98)))
            vmin_diff = -abs_max_diff
            vmax_diff = abs_max_diff
            
        else:  # Temperature variables
            cmap_var = 'RdYlBu_r'  # Red-Yellow-Blue (reverse for temp)
            cmap_diff = 'RdBu_r'   # Red-Blue (red for warming, blue for cooling)
            if variable == 'tmax':
                title = 'Maximum Temperature'
            elif variable == 'tmin':
                title = 'Minimum Temperature'
            else:
                title = 'Mean Temperature'
            units = '°C'
            
            # Use percentiles to avoid outliers
            vmin_hist = np.nanpercentile(hist_mean, 2)
            vmax_hist = np.nanpercentile(hist_mean, 98)
            vmin_scen = np.nanpercentile(scen_mean, 2)
            vmax_scen = np.nanpercentile(scen_mean, 98)
            
            # For difference, use symmetric limits for temperature
            abs_max_diff = max(abs(np.nanpercentile(diff, 2)), abs(np.nanpercentile(diff, 98)))
            vmin_diff = -abs_max_diff
            vmax_diff = abs_max_diff
        
        # Create masked arrays to handle missing data
        hist_mask = np.isnan(hist_mean) | (hist_mean == -999)
        scen_mask = np.isnan(scen_mean) | (scen_mean == -999)
        diff_mask = hist_mask | scen_mask
        
        hist_masked = np.ma.masked_array(hist_mean, mask=hist_mask)
        scen_masked = np.ma.masked_array(scen_mean, mask=scen_mask)
        diff_masked = np.ma.masked_array(diff, mask=diff_mask)
        
        # Plot historical data
        im0 = axes[0].imshow(hist_masked, cmap=cmap_var, vmin=vmin_hist, vmax=vmax_hist)
        axes[0].set_title(f"Historical {title}\n({self.historical_data['config']['start_year']}-{self.historical_data['config']['end_year']})")
        plt.colorbar(im0, ax=axes[0], label=units, shrink=0.8)
        
        # Plot scenario data
        im1 = axes[1].imshow(scen_masked, cmap=cmap_var, vmin=vmin_scen, vmax=vmax_scen)
        axes[1].set_title(f"{scenario_name} {title}\n({self.scenario_data[scenario_name]['config']['start_year']}-{self.scenario_data[scenario_name]['config']['end_year']})")
        plt.colorbar(im1, ax=axes[1], label=units, shrink=0.8)
        
        # Plot difference
        im2 = axes[2].imshow(diff_masked, cmap=cmap_diff, vmin=vmin_diff, vmax=vmax_diff)
        axes[2].set_title(f"Change in {title}\n({scenario_name} minus Historical)")
        plt.colorbar(im2, ax=axes[2], label=f"Δ {units}", shrink=0.8)
        
        # Remove axis ticks for cleaner look
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            
        # Add overall title
        plt.suptitle(f"Spatial Climate Change Analysis: {title}", fontsize=16)
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Spatial change map saved to {output_path}")
        
        # Export spatial data if requested
        if export_data:
            # Export as CSV for easy analysis
            data_output_path = os.path.splitext(output_path)[0] + ".csv"
            try:
                # Create flattened data for export
                rows, cols = hist_mean.shape
                export_data = []
                
                for r in range(rows):
                    for c in range(cols):
                        if not (np.isnan(hist_mean[r,c]) or np.isnan(scen_mean[r,c])):
                            export_data.append({
                                'row': r,
                                'col': c,
                                f'historical_{variable}': hist_mean[r,c],
                                f'future_{variable}': scen_mean[r,c],
                                f'change_{variable}': diff[r,c]
                            })
                
                # Convert to DataFrame and save
                import pandas as pd
                df = pd.DataFrame(export_data)
                df.to_csv(data_output_path, index=False)
                logger.info(f"Spatial data exported to {data_output_path}")
                
                # Create a hot-spot identification summary
                if variable in ['tmean', 'tmax', 'tmin']:
                    # For temperature, identify hotspots (areas with highest warming)
                    threshold = np.nanpercentile(diff, 90)  # Top 10% warming areas
                    hotspot_count = np.sum(diff > threshold)
                    logger.info(f"Identified {hotspot_count} temperature hot-spots (>{threshold:.2f}°C warming)")
                    
                elif variable == 'pr':
                    # For precipitation, identify both wetting and drying hotspots
                    wet_threshold = np.nanpercentile(diff, 90)  # Top 10% wetting areas
                    dry_threshold = np.nanpercentile(diff, 10)  # Top 10% drying areas
                    wet_count = np.sum(diff > wet_threshold)
                    dry_count = np.sum(diff < dry_threshold)
                    logger.info(f"Identified {wet_count} precipitation wetting hot-spots (>{wet_threshold:.2f}mm/day increase)")
                    logger.info(f"Identified {dry_count} precipitation drying hot-spots (<{dry_threshold:.2f}mm/day decrease)")
                
            except Exception as e:
                logger.error(f"Failed to export spatial data: {e}")
        
        return fig

    def generate_climate_change_report(self, output_path: Optional[str] = None, include_hotspot_analysis: bool = True) -> str:
        """
        Generate a comprehensive climate change report with visualizations and analysis.
        
        Args:
            output_path: Path to save the report (if None, uses default path)
            include_hotspot_analysis: Whether to include hot-spot identification in spatial analysis
            
        Returns:
            Path to the generated report
        """
        if not output_path:
            output_path = os.path.join(self.output_dir, 'climate_change_report.md')
            
        if not self.historical_data or not self.scenario_data:
            logger.warning("Cannot generate report without data")
            return ""
            
        # Calculate metrics if not already done
        if not self.deltas:
            self.calculate_climate_change_metrics()
            
        # Generate visualizations
        timeseries_path = os.path.join(self.output_dir, 'climate_timeseries.png')
        self.plot_timeseries_comparison(output_path=timeseries_path)
        
        spatial_paths = {}
        hotspot_data = {}
        for var in ['pr', 'tmax', 'tmin', 'tmean']:
            spatial_path = os.path.join(self.output_dir, f'spatial_{var}.png')
            try:
                self.plot_spatial_change_maps(variable=var, output_path=spatial_path, export_data=include_hotspot_analysis)
                spatial_paths[var] = spatial_path
                
                # Check if CSV export was created
                csv_path = os.path.splitext(spatial_path)[0] + ".csv"
                if os.path.exists(csv_path):
                    hotspot_data[var] = csv_path
            except Exception as e:
                logger.warning(f"Could not generate spatial map for {var}: {e}")
        
        seasonal_paths = {}
        if self.aggregation == 'monthly':
            for var in ['pr', 'tmax', 'tmin']:
                seasonal_path = os.path.join(self.output_dir, f'seasonal_{var}.png')
                try:
                    self.plot_seasonal_cycle_comparison(variable=var, output_path=seasonal_path)
                    seasonal_paths[var] = seasonal_path
                except Exception as e:
                    logger.warning(f"Could not generate seasonal cycle plot for {var}: {e}")
        
        # Generate extreme event analysis
        extreme_events_path = os.path.join(self.output_dir, 'extreme_events_comparison.png')
        try:
            self.plot_extreme_events_comparison(output_path=extreme_events_path)
        except Exception as e:
            logger.warning(f"Could not generate extreme events comparison: {e}")
            extreme_events_path = None
        
        # Generate multi-scenario analysis if bbox is available
        multi_scenario_results = None
        if 'bounding_box' in self.config:
            try:
                logger.info("Generating multi-scenario analysis...")
                multi_scenario_results = self.integrate_multi_scenario_analysis(
                    self.config['bounding_box'],
                    self.output_dir
                )
                logger.info("Multi-scenario analysis completed successfully")
            except Exception as e:
                logger.error(f"Error in multi-scenario analysis: {e}")
        
        # Write report
        try:
            with open(output_path, 'w') as f:
                # Header
                f.write("# Climate Change Analysis Report\n\n")
                
                # Introduction
                f.write("## Introduction\n\n")
                f.write("This report presents an analysis of projected climate change based on LOCA2 ")
                f.write("(Localized Constructed Analog) climate model data. It compares historical climate ")
                f.write("conditions with future scenarios to identify key changes in temperature and precipitation patterns.\n\n")
                
                # Data sources
                f.write("## Data Sources\n\n")
                f.write("### Historical Data\n\n")
                hist_config = self.historical_data['config']
                f.write(f"- **Climate Model**: {hist_config['model']}\n")
                f.write(f"- **Ensemble**: {hist_config['ensemble']}\n")
                f.write(f"- **Time Period**: {hist_config['start_year']} to {hist_config['end_year']}\n")
                f.write(f"- **Temporal Resolution**: {self.aggregation}\n\n")
                
                # Scenario data
                f.write("### Future Scenarios\n\n")
                for scenario_name, scenario_data in self.scenario_data.items():
                    scen_config = scenario_data['config']
                    f.write(f"#### {scenario_name}\n\n")
                    f.write(f"- **Climate Model**: {scen_config['model']}\n")
                    f.write(f"- **Ensemble**: {scen_config['ensemble']}\n")
                    f.write(f"- **Time Period**: {scen_config['start_year']} to {scen_config['end_year']}\n\n")
                
                # Overall climate change metrics
                f.write("## Climate Change Summary\n\n")
                f.write("### Key Climate Change Metrics\n\n")
                
                # Create a summary table for each scenario
                for scenario_name, metrics in self.deltas.items():
                    f.write(f"#### {scenario_name}\n\n")
                    
                    # Create a markdown table with key metrics
                    f.write("| Variable | Historical | Future | Change | % Change |\n")
                    f.write("|----------|------------|--------|--------|----------|\n")
                    
                    for var_name, var_metrics in metrics.items():
                        var_label = {
                            'pr': 'Precipitation (mm/day)',
                            'tmax': 'Max Temperature (°C)',
                            'tmin': 'Min Temperature (°C)',
                            'tmean': 'Mean Temperature (°C)'
                        }.get(var_name, var_name)
                        
                        hist_val = var_metrics['historical_mean']
                        fut_val = var_metrics['scenario_mean']
                        abs_change = var_metrics['absolute_change']
                        pct_change = var_metrics['percent_change']
                        
                        # Format the values
                        hist_fmt = f"{hist_val:.2f}"
                        fut_fmt = f"{fut_val:.2f}"
                        
                        if abs_change >= 0:
                            abs_fmt = f"+{abs_change:.2f}"
                        else:
                            abs_fmt = f"{abs_change:.2f}"
                            
                        if pct_change >= 0:
                            pct_fmt = f"+{pct_change:.1f}%"
                        else:
                            pct_fmt = f"{pct_change:.1f}%"
                        
                        f.write(f"| {var_label} | {hist_fmt} | {fut_fmt} | {abs_fmt} | {pct_fmt} |\n")
                    
                    f.write("\n")
                
                # Time series visualization
                f.write("## Time Series Comparison\n\n")
                f.write("The following figure shows the time series comparison of historical and future climate variables:\n\n")
                f.write(f"![Time Series Comparison]({os.path.basename(timeseries_path)})\n\n")
                f.write("This visualization illustrates how temperature and precipitation are projected to change over time.\n\n")
                
                # Spatial patterns
                f.write("## Spatial Patterns of Change\n\n")
                f.write("The following maps show the spatial distribution of climate variables and their projected changes:\n\n")
                
                for var_name, path in spatial_paths.items():
                    var_label = {
                        'pr': 'Precipitation',
                        'tmax': 'Maximum Temperature',
                        'tmin': 'Minimum Temperature',
                        'tmean': 'Mean Temperature'
                    }.get(var_name, var_name)
                    
                    f.write(f"### {var_label}\n\n")
                    f.write(f"![Spatial Change in {var_label}]({os.path.basename(path)})\n\n")
                    
                    # Add interpretation based on variable
                    if var_name == 'pr':
                        f.write("The precipitation maps show how rainfall patterns are projected to shift across the region. ")
                        f.write("Areas with increased precipitation may experience more frequent flooding events, ")
                        f.write("while areas with decreased precipitation may face increased drought risk.\n\n")
                    elif var_name == 'tmean':
                        f.write("The temperature maps illustrate the spatial distribution of warming across the region. ")
                        f.write("Higher temperatures can affect evapotranspiration rates, crop suitability, ")
                        f.write("and water resource management requirements.\n\n")
                
                # Seasonal cycles (if available)
                if seasonal_paths:
                    f.write("## Seasonal Cycle Changes\n\n")
                    f.write("The following figures show how the seasonal cycle of climate variables is projected to change:\n\n")
                    
                    for var_name, path in seasonal_paths.items():
                        var_label = {
                            'pr': 'Precipitation',
                            'tmax': 'Maximum Temperature',
                            'tmin': 'Minimum Temperature'
                        }.get(var_name, var_name)
                        
                        f.write(f"### {var_label} Seasonal Cycle\n\n")
                        f.write(f"![{var_label} Seasonal Cycle]({os.path.basename(path)})\n\n")
                        
                        # Add interpretation based on variable
                        if var_name == 'pr':
                            f.write("Changes in the seasonal distribution of precipitation can have significant impacts on agriculture, ")
                            f.write("water resources management, and ecosystem function. The timing of seasonal precipitation is often ")
                            f.write("as important as the total amount for many applications.\n\n")
                        elif var_name in ['tmax', 'tmin']:
                            f.write("Shifts in the seasonal temperature cycle can affect growing season length, ")
                            f.write("plant development stages, and heat stress risk during critical periods. ")
                            f.write("Understanding these changes is crucial for agricultural planning and adaptation.\n\n")
                
                # Add extreme events section after seasonal cycles
                if extreme_events_path and os.path.exists(extreme_events_path):
                    f.write("## Extreme Event Analysis\n\n")
                    f.write("This section analyzes changes in climate extremes between historical and future periods, ")
                    f.write("focusing on events that often have the greatest impacts on agriculture, water resources, and infrastructure.\n\n")
                    
                    f.write("![Extreme Climate Events]({0})\n\n".format(os.path.basename(extreme_events_path)))
                    
                    f.write("### Key Findings from Extreme Event Analysis\n\n")
                    
                    # Extract extreme event metrics for reporting
                    has_heavy_rain_increase = False
                    has_dry_spell_increase = False
                    has_heat_wave_increase = False
                    has_cold_spell_decrease = False
                    
                    # Extract values from first scenario (or loop through all if needed)
                    scenario_name = list(self.deltas.keys())[0]
                    metrics = self.deltas[scenario_name]
                    
                    if 'pr' in metrics and 'extreme_events' in metrics['pr']:
                        pr_extremes = metrics['pr']['extreme_events']
                        heavy_rain_change = pr_extremes.get('heavy_rain_percent_change', 0)
                        dry_spell_change = pr_extremes.get('dry_spell_change', 0)
                        
                        has_heavy_rain_increase = heavy_rain_change > 1  # 1% threshold
                        has_dry_spell_increase = dry_spell_change > 0
                        
                        if has_heavy_rain_increase:
                            f.write(f"- **Increased Heavy Rainfall Events**: The analysis shows a {heavy_rain_change:.1f}% increase ")
                            f.write(f"in days with heavy rainfall (exceeding {pr_extremes.get('heavy_rain_threshold', 0):.1f} mm/day). ")
                            f.write("This suggests a higher risk of flash flooding, soil erosion, and water management challenges.\n\n")
                        
                        if has_dry_spell_increase:
                            f.write(f"- **Longer Dry Spells**: Maximum dry spell duration is projected to increase by {dry_spell_change:.1f} days. ")
                            f.write("Longer periods without precipitation may increase drought risk and irrigation requirements.\n\n")
                    
                    if 'tmax' in metrics and 'extreme_events' in metrics['tmax']:
                        tmax_extremes = metrics['tmax']['extreme_events']
                        heat_wave_change = tmax_extremes.get('heat_wave_change', 0)
                        
                        has_heat_wave_increase = heat_wave_change > 0
                        
                        if has_heat_wave_increase:
                            f.write(f"- **Extended Heat Waves**: Heat wave duration is projected to increase by {heat_wave_change:.1f} days. ")
                            f.write("Longer extreme heat events may increase heat stress for crops and livestock, ")
                            f.write("reduce water availability, and strain energy systems.\n\n")
                    
                    if 'tmin' in metrics and 'extreme_events' in metrics['tmin']:
                        tmin_extremes = metrics['tmin']['extreme_events']
                        cold_spell_change = tmin_extremes.get('cold_spell_change', 0)
                        
                        has_cold_spell_decrease = cold_spell_change < 0
                        
                        if has_cold_spell_decrease:
                            f.write(f"- **Shorter Cold Spells**: Cold spell duration is projected to decrease by {abs(cold_spell_change):.1f} days. ")
                            f.write("Reduced cold periods may affect winter dormancy requirements for some crops and allow ")
                            f.write("greater survival of pests and diseases.\n\n")
                    
                    f.write("### Implications for Agriculture and Water Management\n\n")
                    f.write("Changes in extreme events often have more significant impacts than changes in average conditions. ")
                    f.write("These projected shifts in climate extremes suggest the need for:\n\n")
                    
                    f.write("- **Improved stormwater management** to handle more intense rainfall events\n")
                    f.write("- **Enhanced water storage capacity** to buffer against longer dry spells\n")
                    f.write("- **Heat-tolerant crop varieties** that can withstand more extended heat waves\n")
                    f.write("- **Diversified farming systems** to improve resilience to climate extremes\n")
                    f.write("- **Updated infrastructure design standards** that account for changing extreme event patterns\n\n")
                
                # Multi-scenario analysis section if available
                if multi_scenario_results:
                    analysis = multi_scenario_results.get('multi_scenario_analysis', {})
                    summary_table = analysis.get('summary_table', None)
                    
                    f.write("\n## Multi-Scenario Climate Change Analysis\n\n")
                    f.write("This section presents results from analyzing multiple climate scenarios ")
                    f.write("across different time periods to provide a comprehensive view of potential future conditions.\n\n")
                    
                    # Add information about the scenarios analyzed
                    f.write("### Scenarios Analyzed\n\n")
                    f.write("- **Historical**: Baseline climate data (1950-2015)\n")
                    f.write("- **SSP2-4.5**: Lower emissions scenario\n")
                    f.write("- **SSP3-7.0**: Medium emissions scenario\n")
                    f.write("- **SSP5-8.5**: Higher emissions scenario\n\n")
                    
                    f.write("### Time Periods Considered\n\n")
                    f.write("- **Near-term**: 2016-2045\n")
                    f.write("- **Mid-century**: 2045-2075\n")
                    f.write("- **Late-century**: 2075-2100\n\n")
                    
                    # Add summary table if available
                    if summary_table is not None:
                        f.write("### Climate Change Summary Table\n\n")
                        f.write("The following table summarizes projected changes in climate variables ")
                        f.write("across different scenarios and time periods compared to the historical baseline:\n\n")
                        
                        # Convert DataFrame to Markdown table
                        f.write(summary_table.to_markdown(index=False) + "\n\n")
                    
                    # Add visualization references
                    f.write("### Multi-Scenario Visualizations\n\n")
                    f.write("![Climate Change Summary](multi_scenario_climate_change_summary.png)\n\n")
                    f.write("*Figure: Comparison of temperature and precipitation changes across scenarios and time periods.*\n\n")
                    
                    f.write("![Spatial Changes](multi_scenario_spatial_changes_worst_case.png)\n\n")
                    f.write("*Figure: Spatial distribution of climate changes for the high-emission scenario in the late century period.*\n\n")
                    
                    # Add interpretation of results
                    f.write("### Key Findings from Multi-Scenario Analysis\n\n")
                    f.write("1. **Emission Scenario Impact**: Higher emission scenarios consistently show more pronounced climate changes, ")
                    f.write("particularly for temperature increases.\n\n")
                    f.write("2. **Temporal Progression**: Climate changes generally intensify over time, with the most significant changes ")
                    f.write("observed in the late-century period (2075-2100).\n\n")
                    f.write("3. **Spatial Variability**: Some regions within the study area show greater sensitivity to climate change, ")
                    f.write("with notable 'hot spots' of temperature increase or precipitation change.\n\n")
                    f.write("4. **Consistency Across Scenarios**: While magnitudes differ, the general patterns of change are consistent ")
                    f.write("across scenarios, suggesting robust directional signals despite uncertainty in precise amounts.\n\n")
                
                # Climate change implications
                f.write("## Climate Change Implications\n\n")
                
                # Temperature implications
                f.write("### Temperature Implications\n\n")
                
                # Check if we have substantial warming
                has_warming = False
                warming_amount = 0
                if 'tmean' in self.deltas[list(self.deltas.keys())[0]]:
                    warming_amount = self.deltas[list(self.deltas.keys())[0]]['tmean']['absolute_change']
                    has_warming = warming_amount > 0.5  # More than 0.5°C warming
                
                if has_warming:
                    f.write(f"The analysis shows a projected warming of approximately {warming_amount:.1f}°C in mean temperature. Implications include:\n\n")
                    f.write("- **Growing Season**: Potential extension of the growing season, allowing for longer crop development periods\n")
                    f.write("- **Heat Stress**: Increased risk of heat stress for crops, livestock, and human populations\n")
                    f.write("- **Water Demand**: Higher evapotranspiration rates leading to increased irrigation requirements\n")
                    f.write("- **Pests and Diseases**: Potential shifts in pest and disease distribution and life cycles\n")
                    f.write("- **Crop Suitability**: Changes in optimal crop varieties and potential for new crops in the region\n\n")
                else:
                    f.write("The analysis shows minimal changes in temperature patterns. However, even small temperature ")
                    f.write("changes can have cascading effects on agricultural systems, especially during critical growth stages.\n\n")
                
                # Precipitation implications
                f.write("### Precipitation Implications\n\n")
                
                # Check if we have substantial precipitation changes
                has_precip_change = False
                precip_change_amount = 0
                precip_change_percent = 0
                if 'pr' in self.deltas[list(self.deltas.keys())[0]]:
                    precip_change_amount = self.deltas[list(self.deltas.keys())[0]]['pr']['absolute_change']
                    precip_change_percent = self.deltas[list(self.deltas.keys())[0]]['pr']['percent_change']
                    has_precip_change = abs(precip_change_percent) > 5  # More than 5% change
                
                if has_precip_change:
                    if precip_change_amount > 0:
                        f.write(f"The analysis shows a projected **increase** in precipitation of approximately {precip_change_amount:.2f} mm/day ({precip_change_percent:.1f}%). Implications include:\n\n")
                        f.write("- **Flooding Risk**: Potential increase in flooding events and soil erosion\n")
                        f.write("- **Soil Moisture**: Improved soil moisture availability for crop growth\n")
                        f.write("- **Drainage Requirements**: Enhanced need for adequate drainage systems\n")
                        f.write("- **Water Quality**: Potential increase in nutrient leaching and water quality issues\n")
                        f.write("- **Disease Pressure**: Higher humidity potentially increasing crop disease pressure\n\n")
                    else:
                        f.write(f"The analysis shows a projected **decrease** in precipitation of approximately {abs(precip_change_amount):.2f} mm/day ({abs(precip_change_percent):.1f}%). Implications include:\n\n")
                        f.write("- **Drought Risk**: Increased frequency and severity of drought conditions\n")
                        f.write("- **Irrigation Demand**: Higher supplemental irrigation requirements\n")
                        f.write("- **Crop Selection**: Need for more drought-tolerant crop varieties\n")
                        f.write("- **Water Storage**: Increased importance of efficient water storage systems\n")
                        f.write("- **Soil Conservation**: Greater emphasis on soil moisture conservation practices\n\n")
                else:
                    f.write("The analysis shows minimal changes in overall precipitation amounts. However, changes in precipitation timing, ")
                    f.write("intensity, and seasonal distribution may still have significant agricultural impacts.\n\n")
                
                # Adaptation recommendations
                f.write("## Adaptation Recommendations\n\n")
                f.write("Based on the climate change projections analyzed in this report, the following adaptation measures are recommended:\n\n")
                
                f.write("### Agricultural Adaptations\n\n")
                f.write("- Evaluate and adjust crop selection and varieties for future climate conditions\n")
                f.write("- Review and optimize planting calendars to align with shifting seasonal patterns\n")
                f.write("- Invest in irrigation infrastructure and water-efficient technologies\n")
                f.write("- Implement soil conservation practices to enhance resilience to both wet and dry extremes\n")
                f.write("- Diversify farming systems to reduce climate-related risks\n\n")
                
                f.write("### Water Resource Management\n\n")
                f.write("- Evaluate water storage and distribution systems for changing precipitation patterns\n")
                f.write("- Update flood and drought management plans based on projected changes\n")
                f.write("- Implement watershed management practices that enhance resilience to climate extremes\n")
                f.write("- Consider water allocation strategies that account for changing water availability\n\n")
                
                # Data limitations and considerations
                f.write("## Data Limitations and Considerations\n\n")
                f.write("This analysis is subject to several limitations that should be considered when interpreting the results:\n\n")
                
                f.write("- **Model Uncertainty**: Climate models have inherent uncertainties, especially at local and regional scales\n")
                f.write("- **Emissions Scenarios**: Future climate depends on global greenhouse gas emissions trajectories\n")
                f.write("- **Natural Variability**: Natural climate variability may mask or amplify climate change signals\n")
                f.write("- **Extreme Events**: Changes in extreme events may be more impactful than changes in averages\n")
                f.write("- **Resolution**: The spatial resolution of climate models may not capture local microclimates\n\n")
                
                # Methodology
                f.write("## Methodology\n\n")
                f.write("This report is based on data from the LOCA2 (Localized Constructed Analog) downscaled climate projections. ")
                f.write("The analysis includes the following steps:\n\n")
                
                f.write("1. Extraction of historical and future scenario data from LOCA2 datasets\n")
                f.write("2. Calculation of spatial means and temporal aggregation\n")
                f.write("3. Computation of climate change metrics (absolute and percentage changes)\n")
                f.write("4. Visualization of time series, spatial patterns, and seasonal cycles\n")
                f.write("5. Multi-scenario analysis across different time periods and emissions pathways\n")
                f.write("6. Analysis and interpretation of climate change implications\n\n")
                
                # Conclusion
                f.write("## Conclusion\n\n")
                f.write("Climate change is projected to bring significant changes to temperature and precipitation patterns in the study region. ")
                f.write("These changes will require thoughtful adaptation strategies in agriculture, water management, and related sectors. ")
                
                if multi_scenario_results:
                    f.write("The multi-scenario analysis demonstrates that while there are uncertainties in the precise magnitude of changes, ")
                    f.write("the direction and general patterns of change are consistent across different emissions pathways. ")
                
                f.write("Continued monitoring and periodic updates to this analysis are recommended as climate science and projections evolve.\n\n")
                
                # Footer with date
                f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d')}*\n")
                
            logger.info(f"Climate change report generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating climate change report: {e}")
            return ""

    def integrate_multi_scenario_analysis(self, bbox, output_dir=None):
        """
        Integrate multi-scenario LOCA2 analysis results into the climate change report.
        
        Args:
            bbox: Bounding box for the area of interest
            output_dir: Directory to save outputs
            
        Returns:
            Dict containing analysis results
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get multi-scenario climate data
        results = full_climate_change_data(bbox)
        
        # Analyze climate changes and create visualizations directly in the output directory
        analysis_results = analyze_climate_changes(results, output_dir, prefix="multi_scenario_")
        
        # No need to copy files since they are already in the right directory
        # Just ensure the filenames are correct for the report
        expected_files = [
            "multi_scenario_climate_change_summary.png", 
            "multi_scenario_spatial_changes_worst_case.png"
        ]
        
        for file_name in expected_files:
            file_path = os.path.join(output_dir, file_name)
            if os.path.exists(file_path):
                logger.info(f"Generated {file_name} in {output_dir}")
            else:
                logger.warning(f"Expected file {file_name} not found in {output_dir}")
        
        # Return the analysis results for inclusion in the report
        return {
            'multi_scenario_results': results,
            'multi_scenario_analysis': analysis_results
        }

# Update the main method to use the consolidated approach
if __name__ == "__main__":
    # Example usage
    config = {
        "RESOLUTION": 250,
        "huc8": None,
        "video": False,
        "aggregation": "monthly",
        'bounding_box': [-85.444332, 43.658148, -85.239256, 44.164683],  # min_longitude, min_latitude, max_longitude, max_latitude
        'output_dir': 'climate_change_results',
        'use_synthetic_data_fallback': True  # Fall back to synthetic data if real data not available
    }
    
    # Create analysis object
    analysis = ClimateChangeAnalysis(config)
    
    # Define historical data config
    historical_config = {
        'start_year': 2000,
        'end_year': 2012,
        'model': 'ACCESS-CM2',
        'ensemble': 'r1i1p1f1'
    }
    
    # Define scenario data configs
    scenario_configs = [
        {
            'name': 'ssp245',  # Middle-of-the-road scenario
            'start_year': 2050,
            'end_year': 2062,  # Same length as historical for fair comparison
            'model': 'ACCESS-CM2',
            'ensemble': 'r1i1p1f1'
        }
    ]
    
    # Extract data
    success = analysis.extract_data(historical_config, scenario_configs)
    
    if success:
        print("Data extraction successful.")
        
        # Calculate climate change metrics
        metrics = analysis.calculate_climate_change_metrics()
        print("Climate change metrics calculated.")
        
        # Generate plots and comprehensive report with multi-scenario analysis
        report_path = analysis.generate_climate_change_report()
        print(f"Comprehensive report generated: {report_path}")
    else:
        print("Data extraction failed.")

