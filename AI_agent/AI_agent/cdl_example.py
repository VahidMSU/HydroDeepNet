"""
Example script demonstrating CDL data extraction and visualization.

This script shows how to use the CDL_dataset class and cdl_utilities 
module together to extract, analyze, and visualize crop data.
"""
import os
import matplotlib.pyplot as plt
from AI_agent.cdl import CDL_dataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CDL_Example')

def main():
    """Run a complete CDL data extraction and analysis example."""
    
    # Example configuration for southern Michigan
    config = {
        "RESOLUTION": 250,
        "aggregation": "annual",
        "start_year": 2010,
        "end_year": 2020,  # Try to get a decade of data
        "bounding_box": [-85.444332, 43.658148, -85.239256, 44.164683],
    }
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), "cdl_example_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize CDL dataset and extract data
    logger.info("Initializing CDL dataset...")
    cdl = CDL_dataset(config)
    
    logger.info("Extracting CDL data...")
    data = cdl.cdl_trends()
    
    if not data:
        logger.error("No data was extracted. Exiting.")
        return
        
    logger.info(f"Successfully extracted data for {len(data)} years")
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
    # Plot crop trends over time
    fig_trends = cdl.plot_trends(
        output_path=os.path.join(output_dir, "crop_trends.png"),
        top_n=6,
        title="Crop Distribution in Southern Michigan"
    )
    
    if fig_trends:
        plt.figure(fig_trends.number)
        plt.show(block=False)
    
    # Analyze and display crop changes
    logger.info("Analyzing crop changes...")
    changes_df = cdl.analyze_changes()
    
    if not changes_df.empty:
        logger.info("\nTop 5 crop increases:")
        increases = changes_df[changes_df["Change (ha)"] > 0].head(5)
        for _, row in increases.iterrows():
            logger.info(f"  {row['Crop']}: +{row['Change (ha)']:.2f} ha ({row['Status']})")
            
        logger.info("\nTop 5 crop decreases:")
        decreases = changes_df[changes_df["Change (ha)"] < 0].head(5)
        for _, row in decreases.iterrows():
            logger.info(f"  {row['Crop']}: {row['Change (ha)']:.2f} ha ({row['Status']})")
    
    # Export data to CSV
    logger.info("Exporting data to CSV...")
    csv_path = os.path.join(output_dir, "crop_data.csv")
    if cdl.export_data(csv_path):
        logger.info(f"Data exported to {csv_path}")
    
    # Generate comprehensive report
    logger.info("Generating comprehensive report...")
    report_path = cdl.generate_report(
        output_dir=output_dir,
        report_name="Southern_Michigan_CDL_Analysis"
    )
    
    if report_path:
        logger.info(f"Report generated at: {report_path}")
        
    logger.info("Example completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error running example: {e}", exc_info=True)
    
    # Keep plots open if any were created
    plt.show()
