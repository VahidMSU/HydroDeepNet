import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Tuple, Union
import datetime
from matplotlib.dates import DateFormatter
import glob

class StreamflowAnalyzer:
    """
    A class to analyze streamflow data, separate baseflow, and calculate trends.
    
    This class implements several baseflow separation methods including:
    - Digital Filter (Lyne and Hollick)
    - Recursive Digital Filter
    - WHAT (Web-based Hydrograph Analysis Tool) method
    - Local Minimum Method
    """
    
    def __init__(self, data_dir: str, start_year: int = 2000, end_year: int = 2005):
        """
        Initialize the StreamflowAnalyzer.
        
        Parameters:
        -----------
        data_dir : str
            Path to the directory containing streamflow data
        start_year, end_year : int
            Start and end year for the analysis
        """
        self.data_dir = data_dir
        self.start_year = start_year
        self.end_year = end_year
        self.stations = {}
        self.results = {}
        
    def load_data(self):
        """
        Load streamflow data from all stations in the data directory.
        """
        print(f"Loading streamflow data from {self.data_dir}")
        
        # Get all subdirectories (one for each watershed model)
        try:
            watersheds = [name for name in os.listdir(self.data_dir) 
                          if os.path.isdir(os.path.join(self.data_dir, name))]
            
            for watershed in watersheds:
                streamflow_dir = os.path.join(self.data_dir, watershed, "streamflow_data")
                
                if not os.path.exists(streamflow_dir):
                    continue
                
                # Find all daily streamflow files
                daily_files = glob.glob(os.path.join(streamflow_dir, "*_daily.csv"))
                
                for file_path in daily_files:
                    # Extract station name from filename
                    station_name = os.path.basename(file_path).replace("_daily.csv", "")
                    
                    # Read the data
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Check if required columns exist
                        if 'streamflow' not in df.columns or 'date' not in df.columns:
                            print(f"Missing required columns in {file_path}")
                            continue
                            
                        # Convert date to datetime
                        df['date'] = pd.to_datetime(df['date'])
                        
                        # Filter by year range
                        df = df[(df['date'].dt.year >= self.start_year) & 
                                (df['date'].dt.year < self.end_year)]
                        
                        # Store the data
                        self.stations[station_name] = {
                            'watershed': watershed,
                            'data': df,
                            'file_path': file_path
                        }
                        
                        print(f"Loaded {len(df)} records for station {station_name} in watershed {watershed}")
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
            
            print(f"Loaded data for {len(self.stations)} stations")
            return len(self.stations)
        except Exception as e:
            print(f"Error loading streamflow data: {e}")
            return 0
    
    def separate_baseflow_digital_filter(self, streamflow: np.ndarray, 
                                         filter_parameter: float = 0.925, 
                                         passes: int = 3) -> np.ndarray:
        """
        Separate baseflow from streamflow using the digital filter method (Lyne and Hollick, 1979).
        
        Parameters:
        -----------
        streamflow : np.ndarray
            Array of streamflow values
        filter_parameter : float
            Filter parameter, typically between 0.9 and 0.95
        passes : int
            Number of filter passes (usually 3)
            
        Returns:
        --------
        np.ndarray
            Baseflow values
        """
        # Make a copy to prevent modifying the original data
        streamflow = np.copy(streamflow)
        
        # Replace NaNs with zeros for filtering
        streamflow = np.nan_to_num(streamflow, nan=0.0)
        
        # Initialize baseflow array
        baseflow = np.zeros_like(streamflow)
        
        # Forward pass
        filtered_quickflow = np.zeros_like(streamflow)
        for i in range(1, len(streamflow)):
            filtered_quickflow[i] = filter_parameter * filtered_quickflow[i-1] + \
                                   (1 + filter_parameter) / 2 * (streamflow[i] - streamflow[i-1])
        
        # Calculate baseflow (ensure non-negative values and not exceeding streamflow)
        baseflow = streamflow - filtered_quickflow
        baseflow = np.maximum(baseflow, 0)  # Ensure non-negative baseflow
        baseflow = np.minimum(baseflow, streamflow)  # Baseflow cannot exceed streamflow
        
        # For multiple passes
        if passes > 1:
            temp_baseflow = np.copy(baseflow)
            for _ in range(passes - 1):
                filtered_baseflow = np.zeros_like(temp_baseflow)
                for i in range(1, len(temp_baseflow)):
                    filtered_baseflow[i] = filter_parameter * filtered_baseflow[i-1] + \
                                         (1 + filter_parameter) / 2 * (temp_baseflow[i] - temp_baseflow[i-1])
                temp_baseflow = filtered_baseflow
                # Ensure constraints are maintained
                temp_baseflow = np.maximum(temp_baseflow, 0)
                temp_baseflow = np.minimum(temp_baseflow, streamflow)
            baseflow = temp_baseflow
        
        return baseflow
    
    def separate_baseflow_recursive_filter(self, 
                                          streamflow: np.ndarray, 
                                          alpha: float = 0.980) -> np.ndarray:
        """
        Separate baseflow using a recursive digital filter similar to the method
        described by Eckhardt (2005).
        
        Parameters:
        -----------
        streamflow : np.ndarray
            Array of streamflow values
        alpha : float
            Filter parameter (typically 0.98 for perennial streams)
            
        Returns:
        --------
        np.ndarray
            Baseflow values
        """
        # Replace NaNs with zeros for filtering
        streamflow = np.nan_to_num(np.copy(streamflow), nan=0.0)
        
        # Initialize baseflow array with first value
        n = len(streamflow)
        baseflow = np.zeros(n)
        baseflow[0] = streamflow[0]  # Initial condition
        
        # Define BFImax (maximum baseflow index)
        # 0.80 for perennial streams with porous aquifers
        # 0.50 for ephemeral streams with porous aquifers
        # 0.25 for perennial streams with hard rock aquifers
        BFImax = 0.80  # For perennial streams with porous aquifers
        
        # Apply the recursive filter
        for i in range(1, n):
            bf_filtered = (1 - BFImax) * alpha * baseflow[i-1] + (1 - alpha) * BFImax * streamflow[i]
            baseflow[i] = min(bf_filtered, streamflow[i])  # Baseflow cannot exceed streamflow
        
        # Ensure non-negative values
        baseflow = np.maximum(baseflow, 0)
        
        return baseflow
    
    def separate_baseflow_local_minimum(self, 
                                       streamflow: np.ndarray, 
                                       dates: np.ndarray,
                                       interval_days: int = 5) -> np.ndarray:
        """
        Separate baseflow using the local minimum method.
        
        Parameters:
        -----------
        streamflow : np.ndarray
            Array of streamflow values
        dates : np.ndarray
            Array of corresponding dates
        interval_days : int
            Interval (in days) for determining local minima
            
        Returns:
        --------
        np.ndarray
            Baseflow values
        """
        # Replace NaNs with zeros for processing
        streamflow = np.nan_to_num(np.copy(streamflow), nan=0.0)
        n = len(streamflow)
        
        # Convert dates to numerical value (days since start)
        if isinstance(dates[0], (datetime.datetime, np.datetime64, pd.Timestamp)):
            # Convert numpy.datetime64 to Python datetime objects if needed
            if isinstance(dates[0], np.datetime64):
                dates = pd.to_datetime(dates)
            
            # Use pandas to calculate days difference (safer than manual calculation)
            start_date = dates[0]
            date_nums = np.array([(d - start_date).total_seconds() / (24*3600) 
                                if hasattr(d - start_date, 'total_seconds') 
                                else (d - start_date) / np.timedelta64(1, 'D') 
                                for d in dates])
        else:
            # If dates are already numerical, use them directly
            date_nums = np.array(dates)
        
        # Find local minima
        local_min_indices = signal.argrelextrema(streamflow, np.less_equal, order=interval_days)[0]
        
        # Add endpoints
        if 0 not in local_min_indices:
            local_min_indices = np.append([0], local_min_indices)
        if n-1 not in local_min_indices:
            local_min_indices = np.append(local_min_indices, [n-1])
        
        # Sort indices
        local_min_indices = np.sort(local_min_indices)
        
        # Extract values at local minima
        local_min_values = streamflow[local_min_indices]
        local_min_dates = date_nums[local_min_indices]
        
        # Interpolate baseflow between local minima
        baseflow = np.interp(date_nums, local_min_dates, local_min_values)
        
        # Ensure baseflow doesn't exceed streamflow
        baseflow = np.minimum(baseflow, streamflow)
        
        return baseflow
    
    def calculate_trends(self, time_series: np.ndarray, dates: np.ndarray = None) -> Dict:
        """
        Calculate trends in a time series.
        
        Parameters:
        -----------
        time_series : np.ndarray
            Array of values
        dates : np.ndarray, optional
            Array of corresponding dates (if None, uses array indices)
            
        Returns:
        --------
        Dict
            Dictionary with trend statistics
        """
        # Handle no-value data points (marked as -1)
        mask = time_series != -1
        if np.any(~mask):
            print(f"Filtered out {np.sum(~mask)} no-value (-1) data points from time series")
            time_series = time_series[mask]
            if dates is not None:
                dates = dates[mask]
                
        # Remove NaN values
        if dates is not None:
            valid_mask = ~np.isnan(time_series)
            valid_series = time_series[valid_mask]
            valid_dates = dates[valid_mask]
        else:
            valid_mask = ~np.isnan(time_series)
            valid_series = time_series[valid_mask]
            valid_dates = np.arange(len(time_series))[valid_mask]
        
        # If insufficient valid data, return null results
        if len(valid_series) < 2:
            return {
                'slope': np.nan,
                'intercept': np.nan,
                'r_squared': np.nan,
                'p_value': np.nan,
                'trend_percent': np.nan,
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan
            }
        
        # Convert dates to numeric values if they're datetime objects
        if dates is not None and isinstance(valid_dates[0], (datetime.datetime, np.datetime64, pd.Timestamp)):
            # Convert numpy.datetime64 to Python datetime objects if needed
            if isinstance(valid_dates[0], np.datetime64):
                valid_dates = pd.to_datetime(valid_dates)
            
            # Calculate numeric dates (days since first date) with proper handling of timedelta objects
            start_date = valid_dates[0]
            numeric_dates = np.array([(d - start_date).total_seconds() / (24*3600) 
                                    if hasattr(d - start_date, 'total_seconds') 
                                    else (d - start_date) / np.timedelta64(1, 'D') 
                                    for d in valid_dates])
        else:
            numeric_dates = valid_dates
            
        # Reshape for sklearn
        X = numeric_dates.reshape(-1, 1)
        y = valid_series
        
        # Calculate linear regression
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # Calculate R² and p-value
        y_pred = model.predict(X)
        ss_total = np.sum((y - np.mean(y))**2)
        ss_residual = np.sum((y - y_pred)**2)
        r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
        
        # Calculate p-value using t-test
        n = len(y)
        df = n - 2  # degrees of freedom
        # Standard error of the slope
        se = np.sqrt(ss_residual / df / np.sum((X.flatten() - np.mean(X.flatten()))**2))
        t_stat = slope / se
        from scipy import stats
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))
        
        # Calculate trend as percentage change over the series
        if len(numeric_dates) > 1:
            time_range = numeric_dates[-1] - numeric_dates[0]
            total_change = slope * time_range
            if np.mean(y) != 0:
                trend_percent = (total_change / np.mean(y)) * 100
            else:
                trend_percent = np.nan
        else:
            trend_percent = np.nan
        
        # Basic statistics
        mean = np.mean(valid_series)
        median = np.median(valid_series)
        std = np.std(valid_series)
        min_val = np.min(valid_series)
        max_val = np.max(valid_series)
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'p_value': p_value,
            'trend_percent': trend_percent,
            'mean': mean,
            'median': median,
            'std': std,
            'min': min_val,
            'max': max_val
        }
    
    def analyze_all_stations(self, method: str = 'digital_filter'):
        """
        Analyze all stations using the specified baseflow separation method.
        
        Parameters:
        -----------
        method : str
            Baseflow separation method: 'digital_filter', 'recursive_filter', or 'local_minimum'
        """
        print(f"Analyzing {len(self.stations)} stations using {method} method")
        
        for station_name, station_info in self.stations.items():
            try:
                # Get the data
                df = station_info['data']
                
                # Extract streamflow and dates
                streamflow = df['streamflow'].values
                dates = df['date'].values
                
                # Apply baseflow separation
                if method == 'digital_filter':
                    baseflow = self.separate_baseflow_digital_filter(streamflow)
                elif method == 'recursive_filter':
                    baseflow = self.separate_baseflow_recursive_filter(streamflow)
                elif method == 'local_minimum':
                    baseflow = self.separate_baseflow_local_minimum(streamflow, dates)
                else:
                    print(f"Unknown method: {method}, using digital filter")
                    baseflow = self.separate_baseflow_digital_filter(streamflow)
                
                # Calculate quickflow (surface runoff)
                quickflow = streamflow - baseflow
                
                # Calculate baseflow index
                bfi = np.sum(baseflow) / np.sum(streamflow) if np.sum(streamflow) > 0 else np.nan
                
                # Calculate trends for streamflow, baseflow, and quickflow
                streamflow_trend = self.calculate_trends(streamflow, dates)
                baseflow_trend = self.calculate_trends(baseflow, dates)
                quickflow_trend = self.calculate_trends(quickflow, dates)
                
                # Calculate seasonal trends
                df_with_flows = df.copy()
                df_with_flows['baseflow'] = baseflow
                df_with_flows['quickflow'] = quickflow
                df_with_flows['month'] = df_with_flows['date'].dt.month
                
                # Calculate monthly averages
                monthly_avg = df_with_flows.groupby('month').agg({
                    'streamflow': 'mean',
                    'baseflow': 'mean',
                    'quickflow': 'mean'
                }).reset_index()
                
                # Define seasons
                season_map = {
                    1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring',
                    6: 'summer', 7: 'summer', 8: 'summer', 9: 'fall', 10: 'fall',
                    11: 'fall', 12: 'winter'
                }
                df_with_flows['season'] = df_with_flows['date'].dt.month.map(season_map)
                
                # Calculate seasonal averages
                seasonal_avg = df_with_flows.groupby('season').agg({
                    'streamflow': 'mean',
                    'baseflow': 'mean',
                    'quickflow': 'mean'
                }).reset_index()
                
                # Store results
                self.results[station_name] = {
                    'watershed': station_info['watershed'],
                    'data': df_with_flows,
                    'baseflow_method': method,
                    'baseflow_index': bfi,
                    'streamflow_trend': streamflow_trend,
                    'baseflow_trend': baseflow_trend,
                    'quickflow_trend': quickflow_trend,
                    'monthly_averages': monthly_avg,
                    'seasonal_averages': seasonal_avg
                }
                
                print(f"Analysis complete for station {station_name}")
            except Exception as e:
                print(f"Error analyzing station {station_name}: {e}")
        
        print(f"Analysis completed for {len(self.results)} stations")
        return len(self.results)
    
    def generate_plots(self, output_dir: str = './Michigan/figs/streamflow'):
        """
        Generate plots for all analyzed stations.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the plots
        
        Returns:
        --------
        dict
            Dictionary with paths to generated plots
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        plot_paths = {}
        
        for station_name, results in self.results.items():
            try:
                # Create a subdirectory for this station
                station_dir = os.path.join(output_dir, station_name)
                os.makedirs(station_dir, exist_ok=True)
                
                # Get the data
                df = results['data']
                
                # ===== Hydrograph with baseflow separation =====
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Plot streamflow
                ax.plot(df['date'], df['streamflow'], color='blue', alpha=0.6, label='Total Streamflow')
                
                # Plot baseflow
                ax.plot(df['date'], df['baseflow'], color='green', alpha=0.8, label='Baseflow')
                
                # Fill between baseflow and streamflow for quickflow
                ax.fill_between(df['date'], df['baseflow'], df['streamflow'], 
                               color='skyblue', alpha=0.4, label='Quickflow')
                
                # Add title and labels
                ax.set_title(f'Streamflow Hydrograph for {station_name}\nBaseflow Separation using {results["baseflow_method"].replace("_", " ").title()}',
                            fontsize=14, fontweight='bold')
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Flow (cubic feet per second)', fontsize=12)
                
                # Format x-axis
                ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
                fig.autofmt_xdate()
                
                # Add grid
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Add baseflow index and trend information
                bfi = results['baseflow_index']
                streamflow_trend = results['streamflow_trend']['trend_percent']
                baseflow_trend = results['baseflow_trend']['trend_percent']
                
                # Add text box with summary statistics
                txt = (f"Baseflow Index: {bfi:.3f}\n"
                      f"Streamflow Trend: {streamflow_trend:.1f}%\n"
                      f"Baseflow Trend: {baseflow_trend:.1f}%")
                
                ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Add legend
                ax.legend(loc='upper right')
                
                # Save the figure
                hydrograph_path = os.path.join(station_dir, f'{station_name}_hydrograph.png')
                plt.tight_layout()
                plt.savefig(hydrograph_path, dpi=300)
                plt.close()
                
                # ===== Monthly Averages =====
                monthly_avg = results['monthly_averages']
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Month names for x-axis
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                # Plot monthly averages
                ax.bar(monthly_avg['month'], monthly_avg['quickflow'], 
                      color='skyblue', label='Quickflow')
                ax.bar(monthly_avg['month'], monthly_avg['baseflow'], 
                      bottom=0, color='green', label='Baseflow')
                
                # Set x-ticks to month names
                ax.set_xticks(range(1, 13))
                ax.set_xticklabels(month_names)
                
                # Add title and labels
                ax.set_title(f'Monthly Average Flows for {station_name}', 
                            fontsize=14, fontweight='bold')
                ax.set_xlabel('Month', fontsize=12)
                ax.set_ylabel('Average Flow (cubic feet per second)', fontsize=12)
                
                # Add grid
                ax.grid(True, linestyle='--', alpha=0.7, axis='y')
                
                # Add legend
                ax.legend()
                
                # Save the figure
                monthly_path = os.path.join(station_dir, f'{station_name}_monthly_avg.png')
                plt.tight_layout()
                plt.savefig(monthly_path, dpi=300)
                plt.close()
                
                # ===== Seasonal Averages =====
                seasonal_avg = results['seasonal_averages']
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Reorder seasons
                season_order = ['winter', 'spring', 'summer', 'fall']
                seasonal_avg = seasonal_avg.set_index('season').reindex(season_order).reset_index()
                
                # Plot seasonal averages
                ax.bar(seasonal_avg['season'], seasonal_avg['quickflow'], 
                      color='skyblue', label='Quickflow')
                ax.bar(seasonal_avg['season'], seasonal_avg['baseflow'], 
                      bottom=0, color='green', label='Baseflow')
                
                # Add title and labels
                ax.set_title(f'Seasonal Average Flows for {station_name}', 
                            fontsize=14, fontweight='bold')
                ax.set_xlabel('Season', fontsize=12)
                ax.set_ylabel('Average Flow (cubic feet per second)', fontsize=12)
                
                # Add grid
                ax.grid(True, linestyle='--', alpha=0.7, axis='y')
                
                # Add legend
                ax.legend()
                
                # Save the figure
                seasonal_path = os.path.join(station_dir, f'{station_name}_seasonal_avg.png')
                plt.tight_layout()
                plt.savefig(seasonal_path, dpi=300)
                plt.close()
                
                # Store paths
                plot_paths[station_name] = {
                    'hydrograph': os.path.relpath(hydrograph_path, start='./Michigan'),
                    'monthly': os.path.relpath(monthly_path, start='./Michigan'),
                    'seasonal': os.path.relpath(seasonal_path, start='./Michigan')
                }
                
                print(f"Generated plots for station {station_name}")
            except Exception as e:
                print(f"Error generating plots for station {station_name}: {e}")
        
        return plot_paths
    
    def generate_summary_table(self, output_dir: str = './Michigan'):
        """
        Generate a summary table of streamflow analysis results.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the table
            
        Returns:
        --------
        str
            Path to the generated HTML table
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create summary data
        summary_data = []
        
        for station_name, results in self.results.items():
            try:
                # Get key results
                watershed = results['watershed']
                bfi = results['baseflow_index']
                streamflow_trend = results['streamflow_trend']
                baseflow_trend = results['baseflow_trend']
                quickflow_trend = results['quickflow_trend']
                
                # Determine if trends are significant (p-value < 0.05)
                streamflow_significant = streamflow_trend['p_value'] < 0.05
                baseflow_significant = baseflow_trend['p_value'] < 0.05
                quickflow_significant = quickflow_trend['p_value'] < 0.05
                
                # Calculate streamflow components
                streamflow_mean = streamflow_trend['mean']
                baseflow_mean = baseflow_trend['mean']
                quickflow_mean = quickflow_trend['mean']
                
                # Calculate baseflow percentage
                baseflow_percent = (baseflow_mean / streamflow_mean * 100) if streamflow_mean > 0 else np.nan
                
                # Add to summary data
                summary_data.append({
                    'Station': station_name,
                    'Watershed': watershed,
                    'Streamflow Mean': streamflow_mean,
                    'Baseflow Mean': baseflow_mean,
                    'Quickflow Mean': quickflow_mean,
                    'Baseflow Index': bfi,
                    'Baseflow %': baseflow_percent,
                    'Streamflow Trend (%)': streamflow_trend['trend_percent'],
                    'Baseflow Trend (%)': baseflow_trend['trend_percent'],
                    'Quickflow Trend (%)': quickflow_trend['trend_percent'],
                    'Streamflow Trend Significant': streamflow_significant,
                    'Baseflow Trend Significant': baseflow_significant,
                    'Quickflow Trend Significant': quickflow_significant
                })
            except Exception as e:
                print(f"Error adding {station_name} to summary: {e}")
        
        # Create a DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by watershed and then by station name
        if not summary_df.empty:
            summary_df = summary_df.sort_values(['Watershed', 'Station'])
            
            # Create HTML table
            html_table = summary_df.to_html(
                index=False,
                float_format="%.2f",
                classes="table table-striped table-hover",
                escape=False  # Allow HTML in cells
            )
            
            # Add styling for significant trends
            html_table = html_table.replace('>True<', ' style="background-color: rgba(0, 128, 0, 0.2)">Yes<')
            html_table = html_table.replace('>False<', ' style="background-color: rgba(169, 169, 169, 0.2)">No<')
            
            # Save to file
            table_path = os.path.join(output_dir, 'streamflow_analysis_summary.html')
            with open(table_path, 'w') as f:
                f.write(html_table)
            
            print(f"Saved summary table to {table_path}")
            return os.path.basename(table_path)
        else:
            print("No summary data to export")
            return None
    
    def run_full_analysis(self, output_dir: str = './Michigan'):
        """
        Run the full streamflow analysis workflow.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save output files
            
        Returns:
        --------
        dict
            Results of the analysis
        """
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        figs_dir = os.path.join(output_dir, 'figs/streamflow')
        os.makedirs(figs_dir, exist_ok=True)
        
        # Load data
        num_stations = self.load_data()
        if num_stations == 0:
            print("No streamflow data found. Analysis aborted.")
            return None
        
        # Run the analysis
        num_analyzed = self.analyze_all_stations(method='digital_filter')
        if num_analyzed == 0:
            print("No stations could be analyzed. Analysis aborted.")
            return None
        
        # Generate plots
        plot_paths = self.generate_plots(figs_dir)
        
        # Generate summary table
        table_path = self.generate_summary_table(output_dir)
        
        # Return results
        return {
            'num_stations': num_stations,
            'num_analyzed': num_analyzed,
            'plot_paths': plot_paths,
            'table_path': table_path,
            'results': self.results
        }


def analyze_streamflow(base_path='/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12', 
                       start_year=2000, end_year=2005):
    """
    Run a streamflow analysis on the given watershed models.
    
    Parameters:
    -----------
    base_path : str
        Path to the watershed models
    start_year, end_year : int
        Start and end year for the analysis
        
    Returns:
    --------
    dict
        Results of the analysis
    """
    # Create analyzer and run analysis
    analyzer = StreamflowAnalyzer(base_path, start_year=start_year, end_year=end_year)
    results = analyzer.run_full_analysis()
    
    return results


if __name__ == "__main__":
    # Example usage
    base_path = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12"
    start_year = 2000
    end_year = 2005
    
    # Run analysis
    results = analyze_streamflow(base_path, start_year, end_year)
    
    if results:
        print(f"Analysis complete for {results['num_analyzed']} stations")
        print(f"Summary table: {results['table_path']}")
    else:
        print("Analysis failed")