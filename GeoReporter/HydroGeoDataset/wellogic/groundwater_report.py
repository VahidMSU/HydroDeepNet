"""
Groundwater properties analysis and report generation.

This module provides functionality to generate comprehensive reports analyzing
groundwater properties data from the EBK (Empirical Bayesian Kriging) dataset.
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import logging
from pathlib import Path
import itertools
import matplotlib.pyplot as plt
try:
    from config import AgentConfig

    from wellogic.groundwater_utilities import (
        GROUNDWATER_VARIABLES, extract_groundwater_data, get_groundwater_spatial_stats,
        create_groundwater_maps, create_groundwater_error_maps, create_groundwater_histograms,
        create_groundwater_correlation_matrix, compare_groundwater_variables,
        export_groundwater_data_to_csv
    )
except ImportError:
    try:
        from config import AgentConfig

        from HydroGeoDataset.wellogic.groundwater_utilities import (
            GROUNDWATER_VARIABLES, extract_groundwater_data, get_groundwater_spatial_stats,
            create_groundwater_maps, create_groundwater_error_maps, create_groundwater_histograms,
            create_groundwater_correlation_matrix, compare_groundwater_variables,
            export_groundwater_data_to_csv
        )
    except ImportError:
        from GeoReporter.config import AgentConfig

        from GeoReporter.HydroGeoDataset.wellogic.groundwater_utilities import (
            GROUNDWATER_VARIABLES, extract_groundwater_data, get_groundwater_spatial_stats,
            create_groundwater_maps, create_groundwater_error_maps, create_groundwater_histograms,
            create_groundwater_correlation_matrix, compare_groundwater_variables,
            export_groundwater_data_to_csv
        )

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def generate_groundwater_report(data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                              stats: Dict[str, Dict[str, float]],
                              bounding_box: Optional[Tuple[float, float, float, float]] = None,
                              output_dir: str = 'groundwater_report',
                              use_us_units: bool = True) -> str:
    """
    Generate a comprehensive groundwater properties report with visualizations.
    
    Args:
        data: Dictionary with groundwater data (output, error) arrays
        stats: Dictionary with calculated statistics for each variable
        bounding_box: Optional [min_lon, min_lat, max_lon, max_lat] of the region
        output_dir: Directory to save report files
        use_us_units: Whether to display units in US system (feet) or SI (meters)
        
    Returns:
        Path to the generated report file
    """
    if not data:
        logger.warning("No data to generate report")
        return ""
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Define file paths
        maps_path = os.path.join(output_dir, "groundwater_maps.png")
        error_maps_path = os.path.join(output_dir, "groundwater_error_maps.png")
        histograms_path = os.path.join(output_dir, "groundwater_histograms.png")
        correlation_path = os.path.join(output_dir, "groundwater_correlation.png")
        stats_path = os.path.join(output_dir, "groundwater_stats.csv")
        report_path = os.path.join(output_dir, "groundwater_report.md")
        
        # Generate visualizations
        create_groundwater_maps(data, output_path=maps_path, use_us_units=use_us_units)
        create_groundwater_error_maps(data, output_path=error_maps_path, use_us_units=use_us_units)
        create_groundwater_histograms(data, output_path=histograms_path, use_us_units=use_us_units)
        create_groundwater_correlation_matrix(data, output_path=correlation_path)
        
        # Export statistics
        export_groundwater_data_to_csv(data, stats, stats_path)
        
        # Generate comparison plots for key pairs
        key_pairs = [
            ('H_COND_1', 'TRANSMSV_1'),  # K vs T for upper aquifer
            ('H_COND_2', 'TRANSMSV_2'),  # K vs T for lower aquifer
            ('AQ_THK_1', 'H_COND_1'),    # Thickness vs K for upper aquifer
            ('AQ_THK_2', 'H_COND_2'),    # Thickness vs K for lower aquifer
        ]
        
        comparison_plots = {}
        for var1, var2 in key_pairs:
            if var1 in data and var2 in data:
                plot_path = os.path.join(output_dir, f"compare_{var1}_{var2}.png")
                compare_groundwater_variables(data, var1, var2, output_path=plot_path)
                comparison_plots[(var1, var2)] = plot_path
        
        # Generate markdown report
        with open(report_path, 'w') as f:
            # Header
            f.write("# Groundwater Properties Analysis Report\n\n")
            
            # Basic information
            f.write("## Overview\n\n")
            
            if bounding_box:
                f.write(f"**Region:** Lat [{bounding_box[1]:.4f}, {bounding_box[3]:.4f}], ")
                f.write(f"Lon [{bounding_box[0]:.4f}, {bounding_box[2]:.4f}]\n\n")
            
            # Data availability
            f.write("**Available Variables:**\n\n")
            for var_name in data.keys():
                if var_name in GROUNDWATER_VARIABLES:
                    f.write(f"- {GROUNDWATER_VARIABLES[var_name]['description']} ({GROUNDWATER_VARIABLES[var_name]['units']})\n")
            f.write("\n")
            
            # Summary statistics for each variable
            f.write("## Summary Statistics\n\n")
            
            for var_name, var_stats in stats.items():
                if var_name in GROUNDWATER_VARIABLES:
                    var_info = GROUNDWATER_VARIABLES[var_name]
                    units = var_info['units'] if use_us_units else var_info['units_si']
                    
                    f.write(f"### {var_info['description']} ({var_name})\n\n")
                    f.write(f"**Mean:** {var_stats['mean']:.4f} {units}\n\n")
                    f.write(f"**Median:** {var_stats['median']:.4f} {units}\n\n")
                    f.write(f"**Range:** {var_stats['min']:.4f} - {var_stats['max']:.4f} {units}\n\n")
                    f.write(f"**Standard Deviation:** {var_stats['std']:.4f} {units}\n\n")
                    f.write(f"**Coefficient of Variation:** {var_stats['cv']:.2f}%\n\n")
                    f.write(f"**Coverage:** {var_stats['coverage']:.2f}% ({var_stats['valid_cells']} of {var_stats['total_cells']} cells)\n\n")
                    f.write(f"**Mean Error:** {var_stats['error_mean']:.4f} {units}\n\n")
            
            # Spatial distribution
            f.write("## Spatial Distribution\n\n")
            f.write("The maps below show the spatial distribution of groundwater properties across the study area.\n\n")
            f.write(f"![Groundwater Maps]({os.path.basename(maps_path)})\n\n")
            
            # Error analysis
            f.write("## Estimation Uncertainty\n\n")
            f.write("These maps show the standard error of the kriging estimates, indicating the level of confidence in the data.\n\n")
            f.write(f"![Error Maps]({os.path.basename(error_maps_path)})\n\n")
            
            # Statistical distributions
            f.write("## Statistical Distributions\n\n")
            f.write("Histograms showing the frequency distribution of each parameter.\n\n")
            f.write(f"![Histograms]({os.path.basename(histograms_path)})\n\n")
            
            # Correlation analysis
            f.write("## Property Correlations\n\n")
            f.write("This heatmap shows the Pearson correlation coefficients between different groundwater properties.\n\n")
            f.write(f"![Correlation Matrix]({os.path.basename(correlation_path)})\n\n")
            
            # Specific property relationships
            if comparison_plots:
                f.write("## Property Relationships\n\n")
                
                # K vs T relationships
                if ('H_COND_1', 'TRANSMSV_1') in comparison_plots:
                    f.write("### Hydraulic Conductivity vs. Transmissivity (Upper Aquifer)\n\n")
                    f.write("Transmissivity is theoretically the product of hydraulic conductivity and aquifer thickness.\n\n")
                    plot_path = comparison_plots[('H_COND_1', 'TRANSMSV_1')]
                    f.write(f"![K vs T Upper]({os.path.basename(plot_path)})\n\n")
                
                if ('H_COND_2', 'TRANSMSV_2') in comparison_plots:
                    f.write("### Hydraulic Conductivity vs. Transmissivity (Lower Aquifer)\n\n")
                    f.write("The relationship between K and T in the lower aquifer may differ from the upper aquifer.\n\n")
                    plot_path = comparison_plots[('H_COND_2', 'TRANSMSV_2')]
                    f.write(f"![K vs T Lower]({os.path.basename(plot_path)})\n\n")
                
                # Thickness vs K relationships
                if ('AQ_THK_1', 'H_COND_1') in comparison_plots:
                    f.write("### Aquifer Thickness vs. Hydraulic Conductivity (Upper Aquifer)\n\n")
                    plot_path = comparison_plots[('AQ_THK_1', 'H_COND_1')]
                    f.write(f"![Thickness vs K Upper]({os.path.basename(plot_path)})\n\n")
                
                if ('AQ_THK_2', 'H_COND_2') in comparison_plots:
                    f.write("### Aquifer Thickness vs. Hydraulic Conductivity (Lower Aquifer)\n\n")
                    plot_path = comparison_plots[('AQ_THK_2', 'H_COND_2')]
                    f.write(f"![Thickness vs K Lower]({os.path.basename(plot_path)})\n\n")
            
            # Hydrogeologic implications
            f.write("## Hydrogeologic Implications\n\n")
            
            # Water table depth analysis
            if 'SWL' in stats:
                swl_mean = stats['SWL']['mean']
                depth_unit = "ft" if use_us_units else "m"
                
                if (use_us_units and swl_mean < 10) or (not use_us_units and swl_mean < 3):
                    f.write(f"The water table is generally **shallow** in this area (mean depth: {swl_mean:.2f} {depth_unit}). ")
                    f.write("Shallow groundwater may:\n\n")
                    f.write("- Interact with surface water systems\n")
                    f.write("- Be vulnerable to contamination from surface sources\n")
                    f.write("- Contribute to baseflow in streams\n")
                    f.write("- Potentially cause seasonal drainage issues\n\n")
                elif (use_us_units and swl_mean < 33) or (not use_us_units and swl_mean < 10):
                    f.write(f"The water table is at a **moderate depth** (mean: {swl_mean:.2f} {depth_unit}). ")
                    f.write("This suggests:\n\n")
                    f.write("- Reasonable access for shallow wells\n")
                    f.write("- Some natural protection from surface contamination\n")
                    f.write("- Potential seasonal interaction with deeper root systems\n\n")
                else:
                    f.write(f"The water table is **deep** in this area (mean: {swl_mean:.2f} {depth_unit}). ")
                    f.write("Deep groundwater conditions suggest:\n\n")
                    f.write("- Greater pumping costs for water extraction\n")
                    f.write("- Good protection from surface contamination\n")
                    f.write("- Limited interaction with surface water systems\n")
                    f.write("- Potential for confined aquifer conditions\n\n")
                
                if 'cv' in stats['SWL'] and stats['SWL']['cv'] > 50:
                    f.write("The high spatial variability in water table depth (CV: {:.2f}%) ".format(stats['SWL']['cv']))
                    f.write("indicates complex hydrogeologic conditions across the area.\n\n")
            
            # Upper aquifer analysis
            if 'H_COND_1' in stats and 'AQ_THK_1' in stats and 'TRANSMSV_1' in stats:
                k_mean = stats['H_COND_1']['mean']
                t_mean = stats['TRANSMSV_1']['mean']
                thk_mean = stats['AQ_THK_1']['mean']
                
                k_unit = "ft/day" if use_us_units else "m/day"
                t_unit = "ft²/day" if use_us_units else "m²/day"
                thk_unit = "ft" if use_us_units else "m"
                
                # Adjust classification thresholds for US units
                if use_us_units:
                    k_class = "low" if k_mean < 3.3 else "moderate" if k_mean < 33 else "high" if k_mean < 164 else "very high"
                    t_class = "low" if t_mean < 165 else "moderate" if t_mean < 1640 else "high"
                else:
                    k_class = "low" if k_mean < 1 else "moderate" if k_mean < 10 else "high" if k_mean < 50 else "very high"
                    t_class = "low" if t_mean < 50 else "moderate" if t_mean < 500 else "high"
                
                f.write(f"The upper aquifer has **{k_class}** hydraulic conductivity ({k_mean:.2f} {k_unit}) ")
                f.write(f"and **{t_class}** transmissivity ({t_mean:.2f} {t_unit}). ")
                f.write(f"With an average thickness of {thk_mean:.2f} {thk_unit}, this suggests:\n\n")
                
                # Well yield implications
                if t_mean < 50:
                    f.write("- Low potential well yield (suitable for domestic use only)\n")
                elif t_mean < 500:
                    f.write("- Moderate potential well yield (suitable for small community or agricultural use)\n")
                else:
                    f.write("- High potential well yield (suitable for municipal or industrial use)\n")
                
                # Groundwater flow implications
                if k_mean < 1:
                    f.write("- Slow groundwater movement and recharge rates\n")
                elif k_mean < 10:
                    f.write("- Moderate groundwater movement rates\n")
                else:
                    f.write("- Rapid groundwater movement and potential for fast contaminant transport\n")
                    
                # Aquifer material implications
                if k_mean < 0.1:
                    f.write("- Material likely consists of silt, clay, or shale\n\n")
                elif k_mean < 1:
                    f.write("- Material likely consists of silty sand or fine sand\n\n")
                elif k_mean < 10:
                    f.write("- Material likely consists of medium to coarse sand\n\n")
                elif k_mean < 100:
                    f.write("- Material likely consists of coarse sand and gravel\n\n")
                else:
                    f.write("- Material likely consists of clean gravel or karstic limestone\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            f.write("Based on the groundwater properties analysis, the following recommendations are provided:\n\n")
            
            # Well development recommendations
            f.write("### Well Development\n\n")
            
            if any(var in stats for var in ['H_COND_1', 'TRANSMSV_1']):
                if stats.get('H_COND_1', {}).get('mean', 0) > 5 and stats.get('TRANSMSV_1', {}).get('mean', 0) > 200:
                    f.write("- The upper aquifer has good hydraulic properties for well development\n")
                    f.write("- Target depth: {:.1f} - {:.1f} m\n".format(
                        stats.get('SWL', {}).get('mean', 5) + 5, 
                        stats.get('AQ_THK_1', {}).get('mean', 30) + stats.get('SWL', {}).get('mean', 5)
                    ))
                else:
                    f.write("- The upper aquifer has limited productivity; consider exploring deeper options\n")
            
            if any(var in stats for var in ['H_COND_2', 'TRANSMSV_2']):
                if stats.get('H_COND_2', {}).get('mean', 0) > 5 and stats.get('TRANSMSV_2', {}).get('mean', 0) > 200:
                    f.write("- The lower aquifer has good hydraulic properties and may be a viable target\n")
                else:
                    f.write("- The lower aquifer has limited productivity\n")
            
            # Groundwater management recommendations
            f.write("\n### Groundwater Management\n\n")
            
            if 'SWL' in stats and stats['SWL']['mean'] < 5:
                f.write("- Monitor shallow groundwater levels to prevent over-extraction\n")
                f.write("- Consider potential impacts on surface water features\n")
                f.write("- Implement wellhead protection measures due to vulnerability\n")
            else:
                f.write("- Establish monitoring wells to track long-term water level trends\n")
                f.write("- Develop sustainable extraction limits based on aquifer properties\n")
            
            # Future investigations
            f.write("\n### Further Investigations\n\n")
            f.write("- Conduct aquifer tests to verify estimated hydraulic properties\n")
            f.write("- Analyze groundwater quality to assess suitability for intended uses\n")
            f.write("- Develop numerical groundwater flow models for scenario analysis\n")
            
            # Data source and methodology
            f.write("\n## Data Source and Methodology\n\n")
            f.write("This analysis is based on the Empirical Bayesian Kriging (EBK) interpolation of groundwater properties. ")
            f.write("EBK is a geostatistical interpolation method that accounts for the uncertainty in semivariogram estimation ")
            f.write("through a process of subsetting and simulation.\n\n")
            
            f.write("**Processing steps:**\n\n")
            f.write("1. Extraction of interpolated data from HDF5 database\n")
            f.write("2. Spatial subsetting to the region of interest\n")
            f.write("3. Statistical analysis and visualization\n")
            f.write("4. Assessment of property relationships and implications\n\n")
            
            # Limitations
            f.write("### Limitations\n\n")
            f.write("- The analysis is based on interpolated data rather than direct measurements\n")
            f.write("- Kriging uncertainty may be high in areas with sparse well data\n")
            f.write("- Local heterogeneity may not be captured at the analysis resolution\n")
            f.write("- Temporal variations in water levels are not addressed\n\n")
            
            # Data export information
            f.write("## Data Export\n\n")
            f.write(f"The complete dataset has been exported to CSV format. Access the data at: [{os.path.basename(stats_path)}]({os.path.basename(stats_path)})\n\n")
            
            # Report generation information
            f.write("---\n\n")
            f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')}*\n")
        
        logger.info(f"Report successfully generated: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Error generating groundwater report: {e}", exc_info=True)
        return ""

class GroundwaterAnalyzer:
    """
    Class for analyzing groundwater data, generating visualizations, and reports.
    """
    
    def __init__(self, database_path: Optional[str] = None, 
                bounding_box: Optional[Tuple[float, float, float, float]] = None,
                use_us_units: bool = True):
        """
        Initialize the groundwater analyzer.
        
        Args:
            database_path: Path to the HDF5 file with groundwater data
            bounding_box: Optional [min_lon, min_lat, max_lon, max_lat] for spatial subset
            use_us_units: Whether to display units in US system (feet) or SI (meters)
        """
        self.database_path = database_path or AgentConfig.HydroGeoDataset_ML_250_path
        self.bounding_box = bounding_box
        self.use_us_units = use_us_units
        self.data = {}
        self.stats = {}
        
    def extract_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Extract groundwater data for the specified region.
        
        Returns:
            Dictionary with extracted groundwater data
        """
        self.data = extract_groundwater_data(
            database_path=self.database_path,
            bounding_box=self.bounding_box
        )
        
        if self.data:
            self.stats = get_groundwater_spatial_stats(self.data, use_us_units=self.use_us_units)
            logger.info(f"Extracted groundwater data for {len(self.data)} variables")
        else:
            logger.warning("No groundwater data extracted")
            
        return self.data
    
    def generate_report(self, output_dir: str = 'groundwater_results') -> str:
        """
        Generate a comprehensive report of the groundwater data.
        
        Args:
            output_dir: Directory to save report files
            
        Returns:
            Path to the generated report file
        """
        if not self.data:
            logger.warning("No data available. Call extract_data() first.")
            return ""
            
        return generate_groundwater_report(
            data=self.data,
            stats=self.stats,
            bounding_box=self.bounding_box,
            output_dir=output_dir,
            use_us_units=self.use_us_units
        )
    
    def create_visualizations(self, output_dir: str = 'groundwater_results') -> Dict[str, str]:
        """
        Create all groundwater visualizations and save them to disk.
        
        Args:
            output_dir: Directory to save visualization files
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        if not self.data:
            logger.warning("No data available. Call extract_data() first.")
            return {}
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        outputs = {}
        
        # Generate maps
        maps_path = os.path.join(output_dir, "groundwater_maps.png")
        if create_groundwater_maps(self.data, output_path=maps_path, use_us_units=self.use_us_units):
            outputs['maps'] = maps_path
            
        # Generate error maps
        error_maps_path = os.path.join(output_dir, "groundwater_error_maps.png")
        if create_groundwater_error_maps(self.data, output_path=error_maps_path, use_us_units=self.use_us_units):
            outputs['error_maps'] = error_maps_path
            
        # Generate histograms
        histograms_path = os.path.join(output_dir, "groundwater_histograms.png")
        if create_groundwater_histograms(self.data, output_path=histograms_path, use_us_units=self.use_us_units):
            outputs['histograms'] = histograms_path
            
        # Generate correlation matrix
        correlation_path = os.path.join(output_dir, "groundwater_correlation.png")
        if create_groundwater_correlation_matrix(self.data, output_path=correlation_path):
            outputs['correlation'] = correlation_path
            
        # Generate comparison plots for key pairs
        key_pairs = [
            ('H_COND_1', 'TRANSMSV_1'),  # K vs T for upper aquifer
            ('H_COND_2', 'TRANSMSV_2'),  # K vs T for lower aquifer
            ('AQ_THK_1', 'H_COND_1'),    # Thickness vs K for upper aquifer
            ('AQ_THK_2', 'H_COND_2'),    # Thickness vs K for lower aquifer
        ]
        
        for var1, var2 in key_pairs:
            if var1 in self.data and var2 in self.data:
                plot_path = os.path.join(output_dir, f"compare_{var1}_{var2}.png")
                if compare_groundwater_variables(self.data, var1, var2, output_path=plot_path):
                    outputs[f'compare_{var1}_{var2}'] = plot_path
                    
        # Export statistics
        stats_path = os.path.join(output_dir, "groundwater_stats.csv")
        if export_groundwater_data_to_csv(self.data, self.stats, stats_path):
            outputs['stats'] = stats_path
            
        return outputs
    
    def export_data(self, output_path: str) -> bool:
        """
        Export groundwater data statistics to CSV.
        
        Args:
            output_path: Path to save the CSV file
            
        Returns:
            Boolean indicating success
        """
        if not self.data or not self.stats:
            logger.warning("No data available. Call extract_data() first.")
            return False
            
        return export_groundwater_data_to_csv(self.data, self.stats, output_path)
    
    def compare_variables(self, var1: str, var2: str, output_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Create a scatter plot comparing two groundwater variables.
        
        Args:
            var1: First variable to compare
            var2: Second variable to compare
            output_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib Figure object or None if error occurs
        """
        if not self.data:
            logger.warning("No data available. Call extract_data() first.")
            return None
            
        if var1 not in self.data or var2 not in self.data:
            logger.warning(f"Variables {var1} and/or {var2} not available in data")
            return None
            
        return compare_groundwater_variables(
            data=self.data,
            var1=var1,
            var2=var2,
            output_path=output_path
        )
    
    def get_available_variables(self) -> List[str]:
        """
        Get list of available groundwater variables in the data.
        
        Returns:
            List of variable names
        """
        return list(self.data.keys())
    
    def get_variable_stats(self, var_name: str) -> Dict:
        """
        Get statistics for a specific groundwater variable.
        
        Args:
            var_name: Variable name to retrieve statistics for
            
        Returns:
            Dictionary with variable statistics or empty dict if not available
        """
        if not self.stats or var_name not in self.stats:
            return {}
            
        return self.stats[var_name]


if __name__ == "__main__":
    # Example usage
    try:
        # Define bounding box [min_lon, min_lat, max_lon, max_lat]
        bbox = [-85.444332, 43.158148, -84.239256, 44.164683]
        
        # Initialize analyzer
        analyzer = GroundwaterAnalyzer(bounding_box=bbox)
        
        # Extract data
        analyzer.extract_data()
        
        # Get list of available variables
        variables = analyzer.get_available_variables()
        if variables:
            print(f"Available variables: {', '.join(variables)}")
            
            # Generate visualization examples
            output_dir = os.path.join(os.getcwd(), "groundwater_results")
            viz_paths = analyzer.create_visualizations(output_dir)
            
            for viz_name, path in viz_paths.items():
                print(f"Generated {viz_name}: {path}")
            
            # Generate comprehensive report
            report_path = analyzer.generate_report(output_dir)
            if report_path:
                print(f"Report generated: {report_path}")
        else:
            print("No groundwater data available for the specified region")
            
    except Exception as e:
        print(f"Error in example execution: {e}")