"""
Advanced solar radiation analysis for NSRDB data.

This module provides specialized functions for analyzing solar radiation data,
including heat wave probability, solar energy potential, and advanced metrics.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import calendar
import scipy.stats as stats
from datetime import datetime, timedelta
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import logging
from utils.plot_utils import safe_figure, save_figure
import os 

# Configure logger
logger = logging.getLogger(__name__)

def calculate_clear_sky_radiation(latitude: float, day_of_year: int) -> float:
    """
    Calculate theoretical clear-sky radiation for a given latitude and day.
    Simple model based on day of year and latitude.
    
    Args:
        latitude: Latitude in degrees (positive for Northern hemisphere)
        day_of_year: Day of the year (1-366)
        
    Returns:
        Estimated clear sky radiation in W/m²
    """
    # Convert latitude to radians
    lat_rad = latitude * np.pi / 180.0
    
    # Calculate solar declination angle
    declination = 23.45 * np.sin(np.pi * (284 + day_of_year) / 182.5) * np.pi / 180.0
    
    # Calculate day length in hours
    day_length = 24 * np.arccos(-np.tan(lat_rad) * np.tan(declination)) / np.pi
    
    # Simple clear sky model (approximate)
    solar_constant = 1367  # W/m²
    atmosphere_transmissivity = 0.7  # Average clear sky transmissivity
    
    # Calculate average daily insolation on horizontal surface
    # This is a simplification and doesn't account for all factors
    mean_elevation_angle = np.pi/4  # 45 degrees average elevation during daylight
    
    # Clear sky radiation (daily average)
    clear_sky = solar_constant * atmosphere_transmissivity * np.sin(mean_elevation_angle) * (day_length/24.0)
    
    return clear_sky

def identify_heat_wave_periods(data: np.ndarray, threshold_percentile: float = 90, 
                              min_consecutive_days: int = 3) -> List[Tuple[int, int]]:
    """
    Identify heat wave periods based on high solar radiation.
    
    Args:
        data: 1D array of daily solar radiation values
        threshold_percentile: Percentile threshold for defining high radiation
        min_consecutive_days: Minimum consecutive days to qualify as a heat wave
        
    Returns:
        List of (start_idx, end_idx) tuples indicating heat wave periods
    """
    # Calculate threshold value
    threshold = np.nanpercentile(data, threshold_percentile)
    
    # Create binary array where True represents days above threshold
    above_threshold = data > threshold
    
    # Identify periods of consecutive days above threshold
    heat_waves = []
    in_heat_wave = False
    start_idx = 0
    
    for i, val in enumerate(above_threshold):
        if val and not in_heat_wave:
            # Start of potential heat wave
            in_heat_wave = True
            start_idx = i
        elif not val and in_heat_wave:
            # End of potential heat wave
            if (i - start_idx) >= min_consecutive_days:
                heat_waves.append((start_idx, i - 1))
            in_heat_wave = False
    
    # Check if we ended while still in a heat wave
    if in_heat_wave and (len(above_threshold) - start_idx) >= min_consecutive_days:
        heat_waves.append((start_idx, len(above_threshold) - 1))
    
    return heat_waves

def calculate_heat_wave_statistics(data: np.ndarray, dates: pd.DatetimeIndex, 
                                 threshold_percentile: float = 90,
                                 min_consecutive_days: int = 3) -> Dict:
    """
    Calculate statistics about heat waves based on solar radiation.
    
    Args:
        data: 1D array of daily solar radiation values
        dates: DatetimeIndex corresponding to the data values
        threshold_percentile: Percentile threshold for defining high radiation
        min_consecutive_days: Minimum consecutive days to qualify as a heat wave
        
    Returns:
        Dictionary with heat wave statistics
    """
    # Identify heat wave periods
    heat_waves = identify_heat_wave_periods(
        data, threshold_percentile, min_consecutive_days
    )
    
    if not heat_waves:
        return {
            "count": 0,
            "avg_duration": 0,
            "max_duration": 0,
            "total_days": 0,
            "probability": 0,
            "threshold_value": np.nanpercentile(data, threshold_percentile),
            "freq_by_month": {},
            "avg_intensity": 0,
            "max_intensity": 0,
            "events": []
        }
    
    # Calculate statistics
    durations = [end - start + 1 for start, end in heat_waves]
    total_days = sum(durations)
    
    # Calculate average intensity (how much above threshold)
    threshold = np.nanpercentile(data, threshold_percentile)
    intensities = []
    heat_wave_events = []
    
    months = [0] * 12  # Count of heat wave starts by month
    
    for start, end in heat_waves:
        # Get dates and values for this heat wave
        wave_dates = dates[start:end+1]
        wave_values = data[start:end+1]
        
        # Record month of heat wave start
        start_month = wave_dates[0].month - 1  # 0-indexed
        months[start_month] += 1
        
        # Calculate intensity (average exceedance above threshold)
        intensity = np.mean(wave_values) - threshold
        intensities.append(intensity)
        
        # Store event details
        heat_wave_events.append({
            "start_date": wave_dates[0].strftime("%Y-%m-%d"),
            "end_date": wave_dates[-1].strftime("%Y-%m-%d"),
            "duration": end - start + 1,
            "avg_value": float(np.mean(wave_values)),
            "max_value": float(np.max(wave_values)),
            "intensity": float(intensity)
        })
    
    # Convert months count to dictionary with month names
    month_names = list(calendar.month_abbr)[1:]
    freq_by_month = {month: count for month, count in zip(month_names, months)}
    
    # Calculate probability (percentage of days in heat wave)
    probability = total_days / len(data) * 100 if len(data) > 0 else 0
    
    return {
        "count": len(heat_waves),
        "avg_duration": np.mean(durations),
        "max_duration": np.max(durations),
        "total_days": total_days,
        "probability": probability,
        "threshold_value": threshold,
        "freq_by_month": freq_by_month,
        "avg_intensity": np.mean(intensities) if intensities else 0,
        "max_intensity": np.max(intensities) if intensities else 0,
        "events": heat_wave_events
    }

def calculate_solar_energy_potential(data: np.ndarray) -> Dict:
    """
    Calculate solar energy potential metrics from solar radiation data.
    
    Args:
        data: Daily solar radiation values in W/m²
        
    Returns:
        Dictionary with solar energy potential metrics
    """
    # Convert W/m² to kWh/m²/day
    # For daily average radiation, multiply by 24 hours and divide by 1000
    kwh_per_day = data * 24 / 1000
    
    # Standard PV panel efficiency range (15-22% for modern panels)
    pv_efficiency_low = 0.15
    pv_efficiency_high = 0.22
    
    # System losses (inverters, wiring, dust, etc.) - typically 10-20%
    system_efficiency = 0.85
    
    # Calculate PV generation potential (kWh/m²/day)
    pv_potential_low = kwh_per_day * pv_efficiency_low * system_efficiency
    pv_potential_high = kwh_per_day * pv_efficiency_high * system_efficiency
    
    # Calculate reliability metrics
    good_day_threshold = 3.0  # kWh/m²/day, threshold for a "good" solar day
    reliability = np.sum(kwh_per_day >= good_day_threshold) / len(kwh_per_day) * 100
    
    # Calculate variability
    variability = np.std(kwh_per_day) / np.mean(kwh_per_day) if np.mean(kwh_per_day) > 0 else 0
    
    # Calculate monthly averages
    monthly_potential = np.mean(kwh_per_day)
    
    return {
        "daily_mean_kwh": float(np.mean(kwh_per_day)),
        "daily_min_kwh": float(np.min(kwh_per_day)),
        "daily_max_kwh": float(np.max(kwh_per_day)),
        "annual_kwh_per_m2": float(np.sum(kwh_per_day)),
        "pv_potential_low_kwh": float(np.mean(pv_potential_low)),
        "pv_potential_high_kwh": float(np.mean(pv_potential_high)),
        "reliability_percent": float(reliability),
        "variability": float(variability),
        "monthly_kwh_per_m2": float(monthly_potential * 30),  # Approximate monthly production
        "percentile_90_kwh": float(np.percentile(kwh_per_day, 90)),
        "percentile_10_kwh": float(np.percentile(kwh_per_day, 10))
    }

def calculate_clear_sky_index(data: np.ndarray, latitudes: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Calculate clear sky index (ratio of actual to theoretical clear sky radiation).
    
    Args:
        data: 2D array of solar radiation values [time, locations]
        latitudes: 1D array of latitudes for each location
        dates: DatetimeIndex corresponding to the time dimension
        
    Returns:
        Array of clear sky indices
    """
    # Create array to store clear sky values
    clear_sky_values = np.zeros_like(data)
    
    # Calculate clear sky radiation for each day and location
    for t, date in enumerate(dates):
        day_of_year = date.dayofyear
        for i, lat in enumerate(latitudes):
            clear_sky_values[t, i] = calculate_clear_sky_radiation(lat, day_of_year)
    
    # Calculate clear sky index (actual / theoretical)
    # Avoid division by zero
    clear_sky_index = np.divide(
        data,
        clear_sky_values,
        out=np.zeros_like(data),
        where=clear_sky_values != 0
    )
    
    return clear_sky_index

def plot_heat_wave_analysis(data: np.ndarray, dates: pd.DatetimeIndex, 
                           heat_wave_stats: Dict, output_path: Optional[str] = None) -> plt.Figure:
    """
    Create visualization of heat wave frequency and intensity.
    
    Args:
        data: Daily solar radiation values
        dates: Corresponding dates for the data
        heat_wave_stats: Dictionary with heat wave statistics
        output_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    with safe_figure(figsize=(12, 9)) as fig:
        gs = gridspec.GridSpec(3, 2)
        
        # Plot 1: Time series with heat waves highlighted
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(dates, data, color='grey', alpha=0.7, label='Solar Radiation')
        
        # Highlight heat wave periods
        threshold = heat_wave_stats['threshold_value']
        ax1.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label=f'{threshold:.1f} W/m² Threshold')
        
        for event in heat_wave_stats['events']:
            start_date = datetime.strptime(event['start_date'], "%Y-%m-%d")
            end_date = datetime.strptime(event['end_date'], "%Y-%m-%d")
            ax1.axvspan(start_date, end_date, color='red', alpha=0.2)
        
        ax1.set_title("Solar Radiation with Heat Wave Periods Highlighted")
        ax1.set_ylabel("Solar Radiation (W/m²)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Monthly distribution of heat waves
        ax2 = fig.add_subplot(gs[1, 0])
        month_names = list(calendar.month_abbr)[1:]
        month_counts = [heat_wave_stats['freq_by_month'].get(m, 0) for m in month_names]
        
        ax2.bar(month_names, month_counts, color='orange')
        ax2.set_title("Heat Wave Frequency by Month")
        ax2.set_ylabel("Number of Heat Waves")
        ax2.set_xlabel("Month")
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Plot 3: Duration distribution
        ax3 = fig.add_subplot(gs[1, 1])
        durations = [event['duration'] for event in heat_wave_stats['events']]
        
        if durations:
            bins = range(min(durations), max(durations) + 2)
            ax3.hist(durations, bins=bins, color='darkred', edgecolor='black')
        else:
            ax3.text(0.5, 0.5, "No heat waves detected", 
                    horizontalalignment='center', verticalalignment='center')
        
        ax3.set_title("Heat Wave Duration Distribution")
        ax3.set_xlabel("Duration (days)")
        ax3.set_ylabel("Frequency")
        ax3.grid(True, axis='y', alpha=0.3)
        
        # Plot 4: Key statistics
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        stats_text = (
            f"Total Number of Heat Waves: {heat_wave_stats['count']}\n"
            f"Average Duration: {heat_wave_stats['avg_duration']:.1f} days\n"
            f"Maximum Duration: {heat_wave_stats['max_duration']} days\n"
            f"Percentage of Days in Heat Wave: {heat_wave_stats['probability']:.1f}%\n"
            f"Average Intensity: {heat_wave_stats['avg_intensity']:.1f} W/m² above threshold"
        )
        
        ax4.text(0.5, 0.5, stats_text, ha='center', va='center', 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        
        if output_path:
            save_figure(fig, output_path)
        
        return fig

def plot_solar_energy_potential(data: np.ndarray, dates: pd.DatetimeIndex, 
                               solar_potential: Dict, 
                               output_path: Optional[str] = None) -> plt.Figure:
    """
    Create visualization of solar energy potential metrics.
    
    Args:
        data: Daily solar radiation values in W/m²
        dates: Corresponding dates for the data
        solar_potential: Dictionary with solar energy potential metrics
        output_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Convert W/m² to kWh/m²/day
    kwh_per_day = data * 24 / 1000
    
    # Calculate monthly averages
    df = pd.DataFrame({'date': dates, 'kwh': kwh_per_day})
    df['year'] = df.date.dt.year
    df['month'] = df.date.dt.month
    monthly_avg = df.groupby('month')['kwh'].mean()
    
    with safe_figure(figsize=(12, 10)) as fig:
        gs = gridspec.GridSpec(3, 1, height_ratios=[1.5, 1, 1])
        
        # Plot 1: Annual solar energy potential
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(dates, kwh_per_day, color='orange', alpha=0.7)
        
        # Add moving average
        window = min(30, len(kwh_per_day))
        if window > 0:
            moving_avg = pd.Series(kwh_per_day).rolling(window=window).mean().values
            ax1.plot(dates, moving_avg, color='red', linewidth=2, label=f'{window}-day Moving Average')
        
        ax1.set_title("Daily Solar Energy Potential")
        ax1.set_ylabel("Energy Potential (kWh/m²/day)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Monthly averages
        ax2 = fig.add_subplot(gs[1])
        month_names = list(calendar.month_abbr)[1:]
        monthly_values = [monthly_avg.get(i+1, 0) for i in range(12)]
        
        ax2.bar(month_names, monthly_values, color='gold')
        ax2.set_title("Average Monthly Solar Energy Potential")
        ax2.set_ylabel("Energy Potential (kWh/m²/day)")
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Plot 3: PV potential visualization
        ax3 = fig.add_subplot(gs[2])
        ax3.axis('off')
        
        # Create a visual representation of the energy production
        daily_kwh = solar_potential['daily_mean_kwh']
        annual_kwh = solar_potential['annual_kwh_per_m2']
        pv_low = solar_potential['pv_potential_low_kwh']
        pv_high = solar_potential['pv_potential_high_kwh']
        reliability = solar_potential['reliability_percent']
        
        # Prepare stats text
        stats_text = (
            f"Daily Average: {daily_kwh:.2f} kWh/m²/day\n"
            f"Annual Total: {annual_kwh:.1f} kWh/m²/year\n\n"
            f"PV Generation Potential: {pv_low:.2f} - {pv_high:.2f} kWh/m²/day\n\n"
            f"Reliability: {reliability:.1f}% of days have good solar potential\n"
            f"Variability: {solar_potential['variability']:.2f} (coefficient of variation)\n\n"
            f"10th Percentile: {solar_potential['percentile_10_kwh']:.2f} kWh/m²/day\n"
            f"90th Percentile: {solar_potential['percentile_90_kwh']:.2f} kWh/m²/day"
        )
        
        ax3.text(0.5, 0.5, stats_text, ha='center', va='center', 
                fontsize=12, bbox=dict(facecolor='lightyellow', alpha=0.5))
        
        plt.tight_layout()
        
        if output_path:
            save_figure(fig, output_path)
        
        return fig

def calculate_radiation_extremes(data: np.ndarray, dates: pd.DatetimeIndex) -> Dict:
    """
    Calculate extreme value statistics for solar radiation.
    
    Args:
        data: Daily solar radiation values
        dates: Corresponding dates for the data
        
    Returns:
        Dictionary with extreme value statistics
    """
    # Create DataFrame for easier processing
    df = pd.DataFrame({'date': dates, 'radiation': data})
    df['year'] = df.date.dt.year
    df['month'] = df.date.dt.month
    df['day'] = df.date.dt.day
    
    # Calculate percentiles
    p95 = np.percentile(data, 95)
    p99 = np.percentile(data, 99)
    p05 = np.percentile(data, 5)
    p01 = np.percentile(data, 1)
    
    # Calculate return periods and levels
    # Sort data in descending order for high extremes
    sorted_data = np.sort(data)[::-1]
    n = len(sorted_data)
    # Empirical return periods (in days)
    return_periods = n / (np.arange(n) + 1)
    
    # Calculate number of extreme days per year
    extreme_high_days = np.sum(data > p95)
    extreme_low_days = np.sum(data < p05)
    years = df.year.nunique()
    
    # Most extreme periods
    # Find days with highest radiation
    top_days = df.nlargest(5, 'radiation')
    # Find days with lowest radiation
    bottom_days = df.nsmallest(5, 'radiation')
    
    # Calculate longest runs above p95 and below p05
    df['high_extreme'] = df.radiation > p95
    df['low_extreme'] = df.radiation < p05
    
    # Count consecutive days for high extremes
    df['high_run_id'] = (df.high_extreme != df.high_extreme.shift()).cumsum()
    high_runs = df[df.high_extreme].groupby('high_run_id').size()
    longest_high_run = high_runs.max() if not high_runs.empty else 0
    
    # Count consecutive days for low extremes
    df['low_run_id'] = (df.low_extreme != df.low_extreme.shift()).cumsum()
    low_runs = df[df.low_extreme].groupby('low_run_id').size()
    longest_low_run = low_runs.max() if not low_runs.empty else 0
    
    return {
        "percentile_95": float(p95),
        "percentile_99": float(p99),
        "percentile_05": float(p05),
        "percentile_01": float(p01),
        "extreme_high_days_per_year": float(extreme_high_days / years),
        "extreme_low_days_per_year": float(extreme_low_days / years),
        "longest_high_extreme_run": int(longest_high_run),
        "longest_low_extreme_run": int(longest_low_run),
        "highest_radiation_days": top_days[['date', 'radiation']].to_dict('records'),
        "lowest_radiation_days": bottom_days[['date', 'radiation']].to_dict('records'),
        "return_period_1yr_level": float(sorted_data[int(n/365) - 1 if n >= 365 else 0]),
        "return_period_5yr_level": float(sorted_data[int(n/(365*5)) - 1 if n >= 365*5 else 0])
    }

def analyze_climate_correlations(radiation: np.ndarray, humidity: np.ndarray, 
                               wind: np.ndarray, dates: pd.DatetimeIndex) -> Dict:
    """
    Analyze correlations between solar radiation and other climate variables.
    
    Args:
        radiation: Daily solar radiation values
        humidity: Daily relative humidity values
        wind: Daily wind speed values
        dates: Corresponding dates for the data
        
    Returns:
        Dictionary with correlation statistics
    """
    # Create DataFrame with all variables
    df = pd.DataFrame({
        'date': dates,
        'radiation': radiation,
        'humidity': humidity,
        'wind': wind
    })
    
    df['year'] = df.date.dt.year
    df['month'] = df.date.dt.month
    
    # Calculate overall correlations
    corr_rad_hum = np.corrcoef(radiation, humidity)[0, 1]
    corr_rad_wind = np.corrcoef(radiation, wind)[0, 1]
    corr_hum_wind = np.corrcoef(humidity, wind)[0, 1]
    
    # Calculate monthly correlations
    monthly_corr = {}
    for month in range(1, 13):
        month_data = df[df.month == month]
        if len(month_data) > 10:  # Ensure enough data points
            month_rad = month_data.radiation.values
            month_hum = month_data.humidity.values
            month_wind = month_data.wind.values
            
            monthly_corr[month] = {
                'rad_hum': np.corrcoef(month_rad, month_hum)[0, 1],
                'rad_wind': np.corrcoef(month_rad, month_wind)[0, 1],
                'hum_wind': np.corrcoef(month_hum, month_wind)[0, 1]
            }
    
    # Calculate solar/aridity index
    # This is a simplified version (radiation / humidity)
    # Higher values indicate more arid conditions with high radiation
    df['solar_aridity_index'] = df.radiation / (df.humidity + 0.1)  # Add small value to avoid division by zero
    
    # Calculate statistics for this index
    aridity_mean = df.solar_aridity_index.mean()
    aridity_std = df.solar_aridity_index.std()
    aridity_trend = np.polyfit(np.arange(len(df.solar_aridity_index)), df.solar_aridity_index, 1)[0]
    
    return {
        "correlation_radiation_humidity": float(corr_rad_hum),
        "correlation_radiation_wind": float(corr_rad_wind),
        "correlation_humidity_wind": float(corr_hum_wind),
        "monthly_correlations": monthly_corr,
        "solar_aridity_index_mean": float(aridity_mean),
        "solar_aridity_index_std": float(aridity_std),
        "solar_aridity_index_trend": float(aridity_trend)
    }

def plot_climate_correlations(radiation: np.ndarray, humidity: np.ndarray,
                            wind: np.ndarray, dates: pd.DatetimeIndex,
                            correlation_stats: Dict,
                            output_path: Optional[str] = None) -> plt.Figure:
    """
    Create visualization of climate variable correlations.
    
    Args:
        radiation: Daily solar radiation values
        humidity: Daily relative humidity values
        wind: Daily wind speed values
        dates: Corresponding dates for the data
        correlation_stats: Dictionary with correlation statistics
        output_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    with safe_figure(figsize=(12, 10)) as fig:
        gs = gridspec.GridSpec(2, 2)
        
        # Plot 1: Scatter plot of radiation vs humidity
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(radiation, humidity, alpha=0.4, c='purple', s=15)
        ax1.set_xlabel("Solar Radiation (W/m²)")
        ax1.set_ylabel("Relative Humidity (%)")
        ax1.set_title(f"Correlation: {correlation_stats['correlation_radiation_humidity']:.2f}")
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(radiation, humidity, 1)
        p = np.poly1d(z)
        ax1.plot(sorted(radiation), p(sorted(radiation)), "r--", alpha=0.8)
        
        # Plot 2: Scatter plot of radiation vs wind
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(radiation, wind, alpha=0.4, c='teal', s=15)
        ax2.set_xlabel("Solar Radiation (W/m²)")
        ax2.set_ylabel("Wind Speed (m/s)")
        ax2.set_title(f"Correlation: {correlation_stats['correlation_radiation_wind']:.2f}")
        ax2.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(radiation, wind, 1)
        p = np.poly1d(z)
        ax2.plot(sorted(radiation), p(sorted(radiation)), "r--", alpha=0.8)
        
        # Plot 3: Monthly correlations
        ax3 = fig.add_subplot(gs[1, 0])
        months = sorted(correlation_stats['monthly_correlations'].keys())
        rad_hum_corrs = [correlation_stats['monthly_correlations'][m]['rad_hum'] for m in months]
        rad_wind_corrs = [correlation_stats['monthly_correlations'][m]['rad_wind'] for m in months]
        
        month_names = [calendar.month_abbr[m] for m in months]
        x = np.arange(len(months))
        width = 0.35
        
        ax3.bar(x - width/2, rad_hum_corrs, width, label='Radiation-Humidity', color='purple')
        ax3.bar(x + width/2, rad_wind_corrs, width, label='Radiation-Wind', color='teal')
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(month_names)
        ax3.set_ylabel("Correlation Coefficient")
        ax3.set_title("Monthly Climate Correlations")
        ax3.legend()
        ax3.grid(True, axis='y', alpha=0.3)
        
        # Plot 4: Solar aridity index time series
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Calculate solar aridity index
        solar_aridity = radiation / (humidity + 0.1)  # Add small value to avoid division by zero
        
        # Plot time series
        ax4.plot(dates, solar_aridity, color='brown', alpha=0.6)
        
        # Add trend line
        y = solar_aridity
        x = np.arange(len(y))
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax4.plot(dates, p(x), "r-", linewidth=2)
        
        ax4.set_ylabel("Solar Aridity Index")
        ax4.set_title(f"Solar Aridity Index (Trend: {correlation_stats['solar_aridity_index_trend']:.3f}/day)")
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            save_figure(fig, output_path)
            
        return fig

def calculate_pv_performance_metrics(radiation: np.ndarray, temperature: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate detailed PV system performance metrics using solar radiation data.
    
    Args:
        radiation: Daily solar radiation values in W/m²
        temperature: Optional ambient temperature values in °C
        
    Returns:
        Dictionary with PV system performance metrics
    """
    # Convert W/m² to kWh/m²/day
    kwh_per_day = radiation * 24 / 1000
    
    # Define standard parameters
    std_pv_capacity = 1.0  # Standard 1 kW system
    std_efficiency = 0.17  # 17% panel efficiency
    area_per_kw = 6.5  # m²/kW for typical systems
    pr = 0.75  # Performance ratio (accounts for system losses)
    
    # Adjust PR based on temperature if available
    if temperature is not None:
        # Apply temperature coefficient (typically -0.4%/°C above 25°C)
        temp_coeff = -0.004
        temp_diff = temperature - 25  # Difference from standard test condition (25°C)
        # Performance ratio adjustment (only apply to temperatures above 25°C)
        temp_adj = np.maximum(0, temp_diff * temp_coeff)
        pr_adjusted = pr - np.mean(temp_adj)
    else:
        pr_adjusted = pr
    
    # Calculate system output
    daily_output_kwh = kwh_per_day * std_pv_capacity * (area_per_kw / 6.5) * pr_adjusted
    annual_output_kwh = np.sum(daily_output_kwh) * (365 / len(daily_output_kwh))  # Scale to annual
    
    # Calculate capacity factor
    capacity_factor = np.mean(daily_output_kwh) / (std_pv_capacity * 24) * 100
    
    # Calculate variability
    daily_variability = np.std(daily_output_kwh) / np.mean(daily_output_kwh) if np.mean(daily_output_kwh) > 0 else 0
    
    # Calculate financial metrics
    electricity_price = 0.12  # $/kWh (typical US average)
    annual_value = annual_output_kwh * electricity_price
    
    # Installation cost
    install_cost_per_watt = 3.0  # $/W (typical residential system cost)
    system_cost = std_pv_capacity * 1000 * install_cost_per_watt  # $ for 1kW system
    
    # Simple payback period
    payback_years = system_cost / annual_value
    
    return {
        "daily_output_kwh": float(np.mean(daily_output_kwh)),
        "annual_output_kwh": float(annual_output_kwh),
        "capacity_factor_percent": float(capacity_factor),
        "daily_variability": float(daily_variability),
        "performance_ratio": float(pr_adjusted),
        "annual_value_usd": float(annual_value),
        "simple_payback_years": float(payback_years)
    }

def simulate_pv_output(radiation: np.ndarray, dates: pd.DatetimeIndex, 
                     system_capacity_kw: float = 10.0,
                     temperature: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Simulate PV system output from solar radiation data.
    
    Args:
        radiation: Daily solar radiation values in W/m²
        dates: Corresponding dates for the data
        system_capacity_kw: PV system capacity in kW
        temperature: Optional ambient temperature values in °C for temperature correction
        
    Returns:
        DataFrame with simulated PV output
    """
    # Convert W/m² to kWh/m²/day
    kwh_per_day = radiation * 24 / 1000
    
    # Create DataFrame for output
    df = pd.DataFrame({
        'date': dates,
        'solar_radiation_wm2': radiation,
        'solar_energy_kwh_m2': kwh_per_day
    })
    
    # Add month and year columns
    df['month'] = df.date.dt.month
    df['year'] = df.date.dt.year
    df['day'] = df.date.dt.day
    df['dayofyear'] = df.date.dt.dayofyear
    
    # Define system parameters
    area_per_kw = 6.5  # m²/kW for typical systems
    system_area = system_capacity_kw * area_per_kw
    base_pr = 0.75  # Base performance ratio
    
    # Apply temperature correction if available
    if temperature is not None:
        df['temperature_c'] = temperature
        # Temperature coefficient (typically -0.4% per °C above 25°C)
        temp_coeff = -0.004
        df['temp_diff'] = df.temperature_c - 25  # Difference from standard test condition
        df['temp_adjustment'] = df.temp_diff.clip(lower=0) * temp_coeff
        df['performance_ratio'] = base_pr - df.temp_adjustment
    else:
        df['performance_ratio'] = base_pr
    
    # Calculate system output
    df['pv_output_kwh'] = df.solar_energy_kwh_m2 * system_capacity_kw * (area_per_kw / 6.5) * df.performance_ratio
    
    # Calculate financial value
    electricity_price = 0.12  # $/kWh
    df['daily_value_usd'] = df.pv_output_kwh * electricity_price
    
    # Calculate monthly and annual aggregates
    monthly_output = df.groupby(['year', 'month']).agg(
        monthly_output_kwh=('pv_output_kwh', 'sum'),
        monthly_value_usd=('daily_value_usd', 'sum'),
        avg_daily_output_kwh=('pv_output_kwh', 'mean')
    ).reset_index()
    
    annual_output = df.groupby('year').agg(
        annual_output_kwh=('pv_output_kwh', 'sum'),
        annual_value_usd=('daily_value_usd', 'sum')
    ).reset_index()
    
    return {
        'daily': df,
        'monthly': monthly_output,
        'annual': annual_output
    }

def plot_pv_simulation(pv_data: Dict, 
                      output_path: Optional[str] = None) -> plt.Figure:
    """
    Create visualization of PV simulation results.
    
    Args:
        pv_data: Dictionary with DataFrames returned by simulate_pv_output
        output_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    daily_df = pv_data['daily']
    monthly_df = pv_data['monthly']
    annual_df = pv_data['annual']
    
    with safe_figure(figsize=(12, 10)) as fig:
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
        
        # Plot 1: Daily PV output
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(daily_df.date, daily_df.pv_output_kwh, color='orange', alpha=0.7)
        
        # Add moving average
        window = 30
        if len(daily_df) > window:
            daily_df['moving_avg'] = daily_df.pv_output_kwh.rolling(window=window).mean()
            ax1.plot(daily_df.date, daily_df.moving_avg, color='red', linewidth=2, 
                   label=f'{window}-day Moving Average')
            
        ax1.set_title("Daily PV System Output")
        ax1.set_ylabel("Energy Output (kWh)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Monthly averages
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Group by month (ignoring year)
        month_avg = daily_df.groupby('month').pv_output_kwh.mean()
        month_names = list(calendar.month_abbr)[1:]
        
        ax2.bar(month_names, [month_avg.get(i+1, 0) for i in range(12)], color='skyblue')
        ax2.set_title("Average Monthly PV Output")
        ax2.set_ylabel("Daily Output (kWh)")
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Plot 3: Annual output
        ax3 = fig.add_subplot(gs[1, 1])
        if not annual_df.empty:
            ax3.bar(annual_df.year, annual_df.annual_output_kwh, color='green')
            ax3.set_title("Annual PV System Output")
            ax3.set_ylabel("Annual Output (kWh)")
            ax3.grid(True, axis='y', alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "Insufficient data for annual analysis",
                    ha='center', va='center')
        
        # Plot 4: System performance summary
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Calculate key metrics
        total_kwh = annual_df.annual_output_kwh.sum() if not annual_df.empty else daily_df.pv_output_kwh.sum()
        avg_daily = daily_df.pv_output_kwh.mean()
        best_month = month_avg.idxmax()
        worst_month = month_avg.idxmin()
        best_month_output = month_avg.max()
        worst_month_output = month_avg.min()
        
        system_capacity = daily_df.pv_output_kwh.max() / daily_df.performance_ratio.mean() / daily_df.solar_energy_kwh_m2.max()
        capacity_factor = avg_daily / (system_capacity * 24) * 100
        
        # Create summary text
        summary_text = (
            f"PV System Summary (Estimated {system_capacity:.1f} kW capacity)\n\n"
            f"Average Daily Output: {avg_daily:.2f} kWh\n"
            f"Total Production: {total_kwh:.0f} kWh\n"
            f"Capacity Factor: {capacity_factor:.1f}%\n\n"
            f"Best Month: {calendar.month_name[best_month]} ({best_month_output:.2f} kWh/day average)\n"
            f"Worst Month: {calendar.month_name[worst_month]} ({worst_month_output:.2f} kWh/day average)\n"
            f"Summer/Winter Ratio: {best_month_output/worst_month_output:.1f}\n\n"
            f"Estimated Annual Value: ${daily_df.daily_value_usd.sum() * 365/len(daily_df):.0f}"
        )
        
        ax4.text(0.5, 0.5, summary_text, ha='center', va='center',
                fontsize=12, bbox=dict(facecolor='lightgreen', alpha=0.5))
        
        plt.tight_layout()
        
        if output_path:
            save_figure(fig, output_path)
            
        return fig

def analyze_nsrdb_ghi_data(ghi_data: np.ndarray, dates: pd.DatetimeIndex, latitudes: np.ndarray,
                         output_dir: str = "solar_analysis") -> Dict:
    """
    Perform comprehensive analysis of NSRDB solar radiation (GHI) data.
    
    Args:
        ghi_data: Solar radiation values array [time, locations]
        dates: DatetimeIndex corresponding to the time dimension
        latitudes: Array of latitudes for the locations
        output_dir: Directory to save output visualizations
        
    Returns:
        Dictionary with analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate spatial mean solar radiation for each day
    spatial_mean_ghi = np.nanmean(ghi_data, axis=1)
    
    # 1. Heat wave analysis
    heat_wave_stats = calculate_heat_wave_statistics(
        spatial_mean_ghi, dates, threshold_percentile=90, min_consecutive_days=3
    )
    
    heat_wave_plot_path = os.path.join(output_dir, "heat_wave_analysis.png")
    plot_heat_wave_analysis(spatial_mean_ghi, dates, heat_wave_stats, heat_wave_plot_path)
    
    # 2. Solar energy potential
    energy_potential = calculate_solar_energy_potential(spatial_mean_ghi)
    
    energy_plot_path = os.path.join(output_dir, "solar_energy_potential.png")
    plot_solar_energy_potential(spatial_mean_ghi, dates, energy_potential, energy_plot_path)
    
    # 3. PV system performance
    pv_metrics = calculate_pv_performance_metrics(spatial_mean_ghi)
    
    # 4. Calculate clear sky index if latitudes are available
    if latitudes is not None and latitudes.size > 0:
        clear_sky_idx = calculate_clear_sky_index(ghi_data, latitudes, dates)
        clear_sky_mean = np.nanmean(clear_sky_idx, axis=1)
    else:
        clear_sky_mean = np.array([])
    
    # 5. Calculate radiation extremes
    extremes = calculate_radiation_extremes(spatial_mean_ghi, dates)
    
    # 6. PV simulation
    sim_results = simulate_pv_output(spatial_mean_ghi, dates, system_capacity_kw=10.0)
    
    pv_sim_path = os.path.join(output_dir, "pv_simulation.png")
    plot_pv_simulation(sim_results, pv_sim_path)
    
    # Return comprehensive results
    return {
        "heat_wave_analysis": heat_wave_stats,
        "solar_energy_potential": energy_potential,
        "pv_performance_metrics": pv_metrics,
        "radiation_extremes": extremes,
        "pv_simulation_daily": sim_results['daily'].to_dict('records')[:5],  # First 5 days as sample
        "visualization_paths": {
            "heat_wave_plot": heat_wave_plot_path,
            "energy_potential_plot": energy_plot_path,
            "pv_simulation_plot": pv_sim_path
        }
    }

def generate_solar_report_content(analysis_results: Dict, location_name: str = "Study Area") -> str:
    """
    Generate markdown content for a comprehensive solar radiation analysis report.
    
    Args:
        analysis_results: Results from analyze_nsrdb_ghi_data function
        location_name: Name of the location being analyzed
        
    Returns:
        Markdown string for the report
    """
    # Extract data from analysis results
    hw = analysis_results['heat_wave_analysis']
    se = analysis_results['solar_energy_potential']
    pv = analysis_results['pv_performance_metrics']
    ex = analysis_results['radiation_extremes']
    
    # Format report header
    report = f"# Solar Radiation Analysis Report for {location_name}\n\n"
    report += f"*Generated on {datetime.now().strftime('%Y-%m-%d')}*\n\n"
    
    # 1. Executive Summary
    report += "## Executive Summary\n\n"
    report += f"The analysis area receives an average of {se['daily_mean_kwh']:.2f} kWh/m²/day of solar energy "
    report += f"({se['annual_kwh_per_m2']:.0f} kWh/m²/year). "
    report += f"This translates to a photovoltaic generation potential of {pv['annual_output_kwh']:.0f} kWh per year "
    report += f"for a standard 1 kW system, with an estimated capacity factor of {pv['capacity_factor_percent']:.1f}%. "
    
    # Add extreme event summary
    report += f"The area experiences {hw['count']} extended high solar radiation periods ('heat waves') per year, "
    report += f"with an average duration of {hw['avg_duration']:.1f} days.\n\n"
    
    # 2. Solar Resource Assessment
    report += "## Solar Resource Assessment\n\n"
    
    report += "### Solar Energy Availability\n\n"
    report += f"- **Daily Average:** {se['daily_mean_kwh']:.2f} kWh/m²/day\n"
    report += f"- **Annual Total:** {se['annual_kwh_per_m2']:.1f} kWh/m²/year\n"
    report += f"- **Variability:** {se['variability']:.2f} (coefficient of variation)\n"
    report += f"- **90th Percentile:** {se['percentile_90_kwh']:.2f} kWh/m²/day\n"
    report += f"- **10th Percentile:** {se['percentile_10_kwh']:.2f} kWh/m²/day\n\n"
    
    # Add PV potential section
    report += "### Photovoltaic Potential\n\n"
    report += f"- **Performance Ratio:** {pv['performance_ratio']:.2f}\n"
    report += f"- **Capacity Factor:** {pv['capacity_factor_percent']:.1f}%\n"
    report += f"- **Annual Output (1 kW system):** {pv['annual_output_kwh']:.0f} kWh\n"
    report += f"- **Estimated Value:** ${pv['annual_value_usd']:.0f} per year\n"
    report += f"- **Simple Payback Period:** {pv['simple_payback_years']:.1f} years\n\n"
    
    # 3. Extreme Events Analysis
    report += "## Extreme Solar Radiation Analysis\n\n"
    
    report += "### High Radiation Periods\n\n"
    report += f"- **Number of Heat Waves:** {hw['count']}\n"
    report += f"- **Average Duration:** {hw['avg_duration']:.1f} days\n"
    report += f"- **Maximum Duration:** {hw['max_duration']} days\n"
    report += f"- **Percentage of Days in Heat Wave:** {hw['probability']:.1f}%\n\n"
    
    # Add monthly distribution
    report += "### Monthly Distribution of Heat Wave Events\n\n"
    report += "| Month | Number of Events |\n"
    report += "|-------|----------------|\n"
    for month, count in hw['freq_by_month'].items():
        report += f"| {month} | {count} |\n"
    report += "\n"
    
    # 4. Extreme Value Analysis
    report += "### Extreme Value Analysis\n\n"
    report += f"- **Extreme High Days per Year:** {ex['extreme_high_days_per_year']:.1f}\n"
    report += f"- **Extreme Low Days per Year:** {ex['extreme_low_days_per_year']:.1f}\n"
    report += f"- **Longest High Radiation Run:** {ex['longest_high_extreme_run']} days\n"
    report += f"- **Longest Low Radiation Run:** {ex['longest_low_extreme_run']} days\n\n"
    
    # 5. Visualizations
    report += "## Visualizations\n\n"
    
    for name, path in analysis_results['visualization_paths'].items():
        # Convert the filepath to just the filename for embedding in the markdown
        filename = os.path.basename(path)
        report += f"### {name.replace('_', ' ').title()}\n\n"
        report += f"![{name}]({filename})\n\n"
    
    # 6. Recommendations
    report += "## Recommendations\n\n"
    
    # Add recommendations based on analysis results
    if pv['capacity_factor_percent'] > 18:
        report += "- **Excellent solar potential:** This location is well-suited for solar PV installations with above-average energy yield.\n"
    elif pv['capacity_factor_percent'] > 15:
        report += "- **Good solar potential:** This location has favorable conditions for solar PV installations.\n"
    else:
        report += "- **Moderate solar potential:** Solar installations are feasible but will have lower yields compared to sunnier regions.\n"
    
    if hw['probability'] > 15:
        report += "- **Consider heat management:** The high frequency of intense solar radiation periods may require additional cooling for PV systems.\n"
    
    if se['variability'] > 0.5:
        report += "- **High variability noted:** Consider energy storage solutions to manage the variable solar resource.\n"
    else:
        report += "- **Relatively stable solar resource:** The area shows consistent solar radiation patterns.\n"
    
    report += "\n"
    
    # 7. Methodology
    report += "## Methodology\n\n"
    report += "This analysis was performed using data from the National Solar Radiation Database (NSRDB), "
    report += "which provides satellite-derived estimates of solar radiation and meteorological parameters. "
    report += "The analysis includes statistical processing, heat wave identification based on the 90th percentile threshold, "
    report += "and photovoltaic performance modeling using standard assumptions for system efficiency and losses.\n\n"
    
    report += "- Heat waves are defined as periods of at least 3 consecutive days above the 90th percentile of solar radiation.\n"
    report += "- PV performance calculations assume a standard performance ratio of 0.75 and typical crystalline silicon modules.\n"
    report += "- Extreme value analysis is based on the 95th and 5th percentiles of the observed data.\n\n"
    
    return report

if __name__ == "__main__":
    print("NSRDB Solar Analysis module loaded. Import to use functions.")
