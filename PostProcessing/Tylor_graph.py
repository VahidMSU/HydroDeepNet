import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_taylor_diagram(NAME="40500010102"):
    """
    Plots a Taylor diagram for model comparison based on RMSE, Pearson correlation, and standard deviation.
    The diagram is limited to the angle range from 0° to 180° (positive correlations) and labels the closest model to the origin.
    Different colors are used for different 'cc_model' values.
    
    Parameters:
    model_comparisons (DataFrame): DataFrame containing RMSE, correlation, and standard deviation metrics for each model.
                                   It must include columns: 'NAME', 'std', 'correlation', 'cc_model'.
    """
        # Load the model comparison data
    model_comparisons = pd.read_csv(f"/data/MyDataBase/SWATplus_by_VPUID/0405/huc12/{NAME}/climate_change_analysis/model_comparisions.csv")
    
    # Extract the required columns for plotting the Taylor diagram
    std_devs = model_comparisons['std'].values
    correlations = model_comparisons['correlation'].values
    model_names = model_comparisons['NAME'].values
    cc_model_names = model_comparisons['cc_model'].values

    # Create a unique color map based on cc_model
    unique_cc_models = np.unique(cc_model_names)
    color_map = {cc_model: plt.cm.tab20(i % 20) for i, cc_model in enumerate(unique_cc_models)}

    # Reference standard deviation (assumed as the mean of the model standard deviations for normalization)
    prism_std = np.mean(std_devs)

    # Setup the polar plot for the Taylor diagram
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)

    # Initialize a variable to track the closest point to the origin
    min_distance = float('inf')
    closest_model = None
    closest_coords = (0, 0)

    # Plot each model on the Taylor diagram with colors based on 'cc_model'
    for i, model in enumerate(model_names):
        theta = np.arccos(correlations[i])  # Angle based on Pearson correlation
        r = std_devs[i] / prism_std         # Normalized standard deviation

        if not np.isnan(theta) and not np.isnan(r):
            color = color_map[cc_model_names[i]]  # Assign color based on cc_model
            ax.plot(theta, r, 'o', label=model, markersize=6, color=color)
            
            # Calculate the distance from the origin
            distance = np.sqrt((r * np.cos(theta))**2 + (r * np.sin(theta))**2)

            # Check if this model is the closest to the origin
            if distance < min_distance:
                min_distance = distance
                closest_model = model
                closest_coords = (theta, r)

    # Add a label for the closest model to the origin
    if closest_model:
        ax.annotate(f'Closest: {closest_model}', xy=closest_coords, xytext=(10, 10),
                    textcoords='offset points', arrowprops=dict(facecolor='black', arrowstyle='->'),
                    fontsize=10, color='red')

    # Customize the plot: limit the angular range from 0° to 180°
    ax.set_thetamin(0)   # Set the minimum angle to 0°
    ax.set_thetamax(180)  # Set the maximum angle to 180°

    # Customize the plot appearance
    ax.set_title('Taylor Diagram: Model Performance (0° to 180°)', fontsize=15)
    ax.set_xlabel('Correlation', fontsize=12)
    ax.set_ylabel('Normalized Standard Deviation', fontsize=12)

    # Create a legend for the 'cc_model' color map
    handles = [plt.Line2D([0], [0], marker='o', color=color_map[cc_model], linestyle='', markersize=6, label=cc_model)
               for cc_model in unique_cc_models]
    ax.legend(handles=handles, title='Climate Change Models', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig("taylor_diagram_labeled_closest_colored.png")

if __name__ == "__main__":

    # Plot the Taylor diagram with limited angles, color-coded by 'cc_model', and label the closest point
    plot_taylor_diagram()
