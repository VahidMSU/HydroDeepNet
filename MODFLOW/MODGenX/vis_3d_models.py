"""
3D visualization for MODFLOW models
"""

import os
import numpy as np
import pyvista as pv
from rasterio.transform import Affine
import rasterio
from matplotlib import colormaps
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvistaqt
try:
    from pyvista import trame
    TRAME_AVAILABLE = True
except ImportError:
    TRAME_AVAILABLE = False

# Try to start the virtual X server
try:
    pv.start_xvfb()
    print("Successfully started virtual X server for PyVista")
except Exception as e:
    print(f"Cannot start virtual X server: {e}")
    print("Falling back to matplotlib for visualization")

# Configure PyVista for off-screen rendering
pv.OFF_SCREEN = True
# Use 'trame' for interactive off-screen rendering if available
try:
    pv.global_theme.jupyter_backend = 'trame'
except:
    pass


def read_raster(file_path):
    """Read a raster file and return the data and metadata."""
    with rasterio.open(file_path) as src:
        data = src.read(1)
        transform = src.transform
        nodata = src.nodata
    return data, transform, nodata


def convert_feet_to_meters(data):
    """Convert values from feet to meters."""
    return data * 0.3048


def mask_by_domain(data, domain_data, nodata_value):
    """Mask data outside the domain."""
    masked_data = data.copy()
    masked_data[domain_data == 0] = nodata_value
    return masked_data


def create_terrain_mesh(dem_data, transform, z_values, domain_data=None, nodata_value=None, vertical_exaggeration=5.0):
    """
    Create a 3D terrain-following mesh using DEM data and vertical layer information.
    
    Parameters:
    -----------
    dem_data : numpy.ndarray
        The digital elevation model data.
    transform : affine.Affine
        The geotransform of the DEM.
    z_values : list of numpy.ndarray
        List of Z values for each layer (from top to bottom).
    domain_data : numpy.ndarray, optional
        Domain mask to apply to the data.
    nodata_value : float, optional
        Value to use for areas outside the domain.
    vertical_exaggeration : float, optional
        Factor to exaggerate the vertical dimension. Default is 5.0.
        
    Returns:
    --------
    pyvista.StructuredGrid
        The 3D terrain-following mesh.
    """
    # Get dimensions
    rows, cols = dem_data.shape
    
    # Create a mask for valid DEM values (not 0, NaN, or nodata)
    valid_mask = np.ones_like(dem_data, dtype=bool)
    
    # Explicitly check for exact zeros
    valid_mask[dem_data == 0] = False
    
    # Check for very small values (near zero)
    valid_mask[np.abs(dem_data) < 1e-6] = False
    
    # Check for NaN values
    valid_mask[np.isnan(dem_data)] = False
    
    # Check for nodata values if specified
    if nodata_value is not None:
        valid_mask[dem_data == nodata_value] = False
    
    # Combine with domain mask if provided
    if domain_data is not None:
        domain_mask = domain_data > 0
        valid_mask = np.logical_and(valid_mask, domain_mask)
    
    # Print statistics for debugging
    print(f"Valid DEM values: {valid_mask.sum()} out of {valid_mask.size} ({valid_mask.sum()/valid_mask.size*100:.2f}%)")
    
    # Check if there are any zeros in the valid area (for debugging)
    zeros_count = np.sum((dem_data == 0) & valid_mask)
    if zeros_count > 0:
        print(f"WARNING: Found {zeros_count} zero values in valid DEM area.")
        # Force these to be invalid
        valid_mask[dem_data == 0] = False
    
    # Create coordinate meshgrids
    y, x = np.mgrid[0:rows, 0:cols]
    
    # Apply geotransform to get real-world coordinates
    world_x = transform.c + x * transform.a + y * transform.b
    world_y = transform.f + x * transform.d + y * transform.e
    
    # Make copies of the z_values and modify those (avoid modifying originals)
    masked_z_values = []
    for layer in z_values:
        # Replace invalid areas with NaN
        masked_z = np.where(valid_mask, layer, np.nan)
        masked_z_values.append(masked_z)
    
    # Apply vertical exaggeration to the masked z layers
    exaggerated_z_values = []
    for i in range(len(masked_z_values)):
        # Calculate mean elevation to use as reference (using only valid data)
        valid_data = masked_z_values[i][valid_mask]
        if len(valid_data) > 0:
            mean_elev = np.mean(valid_data)
            # Apply exaggeration relative to mean elevation
            exaggerated_z = (masked_z_values[i] - mean_elev) * vertical_exaggeration + mean_elev
            exaggerated_z_values.append(exaggerated_z)
        else:
            exaggerated_z_values.append(masked_z_values[i])
    
    # Stack layers to create 3D coordinates
    xx = np.repeat(world_x[:, :, np.newaxis], len(z_values), axis=2)
    yy = np.repeat(world_y[:, :, np.newaxis], len(z_values), axis=2)
    
    # Create the z coordinates for each layer using exaggerated values
    zz = np.zeros((rows, cols, len(z_values)))
    for i in range(len(z_values)):
        zz[:, :, i] = exaggerated_z_values[i]
    
    # Create the validity mask array extended to 3D
    valid_mask_3d = np.repeat(valid_mask[:, :, np.newaxis], len(z_values), axis=2)
    
    # Create a new grid including only valid cells
    # Create arrays of x, y, z coordinates for valid cells only
    valid_points_mask = valid_mask_3d.ravel(order="F")
    
    # Create the structured grid
    grid = pv.StructuredGrid(xx, yy, zz)
    
    # Add elevation data as a point array
    grid["Elevation"] = zz.ravel(order="F")
    
    # Add the validity mask as a point array
    grid["ValidMask"] = valid_mask_3d.astype(np.float32).ravel(order="F")
    
    # Extract only cells where all points have valid data
    # Use a stricter threshold (1.0) to ensure all points in a cell are valid
    trimmed_grid = grid.threshold(scalars="ValidMask", value=0.99)
    
    print(f"Original grid had {grid.n_cells} cells, trimmed grid has {trimmed_grid.n_cells} cells")
    
    # Check if any zero values remain in the trimmed grid
    elevation_points = trimmed_grid["Elevation"]
    zero_count = np.sum(np.abs(elevation_points) < 1e-6)
    if zero_count > 0:
        print(f"WARNING: Trimmed grid still has {zero_count} points with zero elevation")
    
    return trimmed_grid


def plot_3d_model(base_path, model_name, save_screenshot=True, interactive=False, vertical_exaggeration=5.0, save_html=True):
    """
    Create a 3D visualization of a MODFLOW model.
    
    Parameters:
    -----------
    base_path : str
        Path to the model directory.
    model_name : str
        Name of the model.
    save_screenshot : bool, optional
        Whether to save a screenshot of the visualization. Default is True.
    interactive : bool, optional
        Whether to try to show the plot interactively. Default is False.
    vertical_exaggeration : float, optional
        Factor to exaggerate the vertical dimension. Default is 5.0.
    save_html : bool, optional
        Whether to save an interactive HTML visualization. Default is True.
    """
    # Extract the watershed name directly from visualize_model.py parameters
    # Parse from path: /data/SWATGenXApp/Users/username/SWATplus_by_VPUID/vpuid/level/name/model/
    parts = base_path.rstrip('/').split('/')
    name = None
    
    # Try to find the part that matches the expected pattern for watershed name
    # Looking for the longest numeric string which is likely the watershed ID
    for part in parts:
        if part.isdigit() or (part.startswith('0') and part[1:].isdigit()):
            if name is None or len(part) > len(name):
                name = part
    
    # Explicit check for the name in the expected position (based on path structure)
    # The watershed name should be 2 levels up from the model directory
    if len(parts) >= 3:
        expected_name_pos = -3  # Two levels up from model directory
        expected_name = parts[expected_name_pos]
        if expected_name.isdigit() or (expected_name.startswith('0') and expected_name[1:].isdigit()):
            name = expected_name
    
    if not name:
        raise ValueError(f"Could not extract watershed name from path: {base_path}")
    
    # Construct file paths
    rasters_dir = os.path.join(base_path, "rasters_input")
    dem_file = os.path.join(base_path, "..", "DEM_250m.tif")
    domain_file = os.path.join(rasters_dir, "domain.tif")
    
    # Construct correct file paths using the watershed name
    aq1_thickness_file = os.path.join(rasters_dir, f"{name}_kriging_output_AQ_THK_1_250m.tif")
    aq2_thickness_file = os.path.join(rasters_dir, f"{name}_kriging_output_AQ_THK_2_250m.tif")
    swl_file = os.path.join(rasters_dir, f"{name}_kriging_output_SWL_250m.tif")
    
    print(f"Using watershed name: {name}")
    print(f"Loading DEM from: {dem_file}")
    print(f"Loading domain from: {domain_file}")
    print(f"Loading AQ1 thickness from: {aq1_thickness_file}")
    print(f"Loading AQ2 thickness from: {aq2_thickness_file}")
    print(f"Loading SWL from: {swl_file}")
    
    # Read raster data
    dem_data, dem_transform, dem_nodata = read_raster(dem_file)
    domain_data, domain_transform, domain_nodata = read_raster(domain_file)
    aq1_data, aq1_transform, aq1_nodata = read_raster(aq1_thickness_file)
    aq2_data, aq2_transform, aq2_nodata = read_raster(aq2_thickness_file)
    swl_data, swl_transform, swl_nodata = read_raster(swl_file)
    
    # Convert aquifer data from feet to meters
    aq1_data_m = convert_feet_to_meters(aq1_data)
    aq2_data_m = convert_feet_to_meters(aq2_data)
    swl_data_m = convert_feet_to_meters(swl_data)
    
    # Create elevation layers
    surface_elev = dem_data
    
    # Ground water level
    gwl_elev = dem_data - swl_data_m
    
    # Bottom of first aquifer
    bottom_aq1 = gwl_elev - aq1_data_m
    
    # Bottom of second aquifer
    bottom_aq2 = bottom_aq1 - aq2_data_m
    
    # Create a mask of valid DEM points for visualization clarity
    dem_mask = np.ones_like(dem_data, dtype=bool)
    dem_mask[dem_data == 0] = False
    dem_mask[np.isnan(dem_data)] = False
    if dem_nodata is not None:
        dem_mask[dem_data == dem_nodata] = False
    
    # Also mask by domain if available
    if domain_data is not None:
        domain_mask = domain_data > 0
        dem_mask = np.logical_and(dem_mask, domain_mask)
    
    # Apply the mask to all elevation layers
    masked_surface_elev = np.where(dem_mask, surface_elev, np.nan)
    masked_gwl_elev = np.where(dem_mask, gwl_elev, np.nan)
    masked_bottom_aq1 = np.where(dem_mask, bottom_aq1, np.nan)
    masked_bottom_aq2 = np.where(dem_mask, bottom_aq2, np.nan)
    
    # Create layer elevations with masked values (from top to bottom)
    z_values = [masked_surface_elev, masked_gwl_elev, masked_bottom_aq1, masked_bottom_aq2]
    
    # Create the 3D mesh with vertical exaggeration
    try:
        # Try to use PyVista for 3D visualization
        mesh = create_terrain_mesh(
            dem_data, 
            dem_transform, 
            z_values, 
            domain_data, 
            dem_nodata, 
            vertical_exaggeration=vertical_exaggeration
        )
        
        # Create output directory
        output_dir = os.path.join(base_path, "visualization")
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Create the basic static visualization
            plotter = pv.Plotter(off_screen=not interactive)
            
            # Set a single color for the entire mesh based on elevation
            plotter.add_mesh(
                mesh, 
                scalars="Elevation",
                cmap="terrain",
                show_edges=True,
                opacity=0.8,
                clim=[np.nanmin(mesh["Elevation"]), np.nanmax(mesh["Elevation"])],
            )
            
            # Add boundaries of each layer as wireframe
            layer_count = len(z_values)
            colors = ["white", "lightblue", "blue", "darkblue"]
            
            # Extract each layer from the full mesh
            layers = []
            for i in range(layer_count):
                # Extract the i-th layer surface as a wireframe
                layer_pts = mesh.points.reshape(-1, 3)
                layer_elev = mesh["Elevation"].reshape(-1, layer_count)[:, i]
                
                # Create a point cloud for this layer
                layer_cloud = pv.PolyData(layer_pts)
                layer_cloud["Elevation"] = layer_elev
                
                # Add to plotter with appropriate color
                if i == 0:
                    edge_color = "brown"  # Surface
                    edge_width = 2
                    label = "Surface"
                elif i == 1:
                    edge_color = "royalblue"  # Water table
                    edge_width = 2
                    label = "Water Table"
                elif i == 2:
                    edge_color = "teal"  # Bottom of aquifer 1
                    edge_width = 1
                    label = "Bottom of Aquifer 1"
                else:
                    edge_color = "navy"  # Bottom of aquifer 2
                    edge_width = 1
                    label = "Bottom of Aquifer 2"
                
                plotter.add_mesh(
                    layer_cloud, 
                    color=edge_color,
                    line_width=edge_width,
                    style="wireframe",
                    render_points_as_spheres=False,
                    label=label
                )
                
                layers.append(layer_cloud)
            
            # Add a legend
            plotter.add_legend()
            
            # Add a text annotation about vertical exaggeration
            plotter.add_text(
                f"Vertical Exaggeration: {vertical_exaggeration}x", 
                position='upper_left', 
                font_size=12
            )
            
            # Set camera position for best view
            plotter.view_isometric()
            
            # Set a title
            plotter.add_text(f"3D MODFLOW Model: {name}", font_size=20)
            
            # Save screenshot if requested
            if save_screenshot:
                screenshot_path = os.path.join(output_dir, f"{name}_3d_model_ve{int(vertical_exaggeration)}x.png")
                plotter.screenshot(screenshot_path)
                print(f"Screenshot saved to: {screenshot_path}")
            
            # Show plot if interactive
            if interactive:
                plotter.show()
            
            # Close the plotter
            plotter.close()
            
            # Now create an interactive HTML visualization
            if save_html:
                print("Creating interactive HTML visualization...")
                
                if TRAME_AVAILABLE:
                    # Use trame for interactive visualization
                    plotter = pv.Plotter(notebook=True)
                    
                    # Add the mesh with different colors for each layer
                    # Surface to water table (vadose zone)
                    plotter.add_mesh(layers[0], color="tan", opacity=0.7, show_edges=True)
                    
                    # Water table to bottom of first aquifer
                    plotter.add_mesh(layers[1], color="lightblue", opacity=0.6, show_edges=True)
                    
                    # Second aquifer
                    if len(layers) > 2:
                        plotter.add_mesh(layers[2], color="blue", opacity=0.5, show_edges=True)
                    
                    # Add basic controls
                    plotter.add_title(f"3D MODFLOW Model: {name} (VE: {vertical_exaggeration}x)")
                    
                    # Export to HTML
                    html_path = os.path.join(output_dir, f"{name}_3d_model_interactive.html")
                    plotter.export_html(html_path)
                    print(f"Interactive HTML saved to: {html_path}")
                    
                else:
                    # Fallback to pyvista's basic HTML export
                    p = pv.Plotter(notebook=True)
                    
                    # Add the mesh with different colors for each layer
                    p.add_mesh(mesh, scalars="Elevation", cmap="terrain", show_edges=True)
                    
                    # Set a nice camera position
                    p.view_isometric()
                    
                    # Export as a local HTML file
                    html_path = os.path.join(output_dir, f"{name}_3d_model_basic.html")
                    p.export_html(html_path)
                    print(f"Basic HTML visualization saved to: {html_path}")
                    
                    # Create a simplified version with vtkjs export (most compatible)
                    p = pv.Plotter()
                    p.add_mesh(layers[0], color="tan", opacity=0.7, show_edges=True)
                    p.add_mesh(layers[1], color="lightblue", opacity=0.6, show_edges=True)
                    if len(layers) > 2:
                        p.add_mesh(layers[2], color="blue", opacity=0.5, show_edges=True)
                    p.view_isometric()
                    
                    vtkjs_path = os.path.join(output_dir, f"{name}_3d_model")
                    p.export_vtkjs(vtkjs_path)
                    print(f"VTK.js visualization saved to: {vtkjs_path}.vtkjs")
                    
                    # Create a viewer HTML that can load the vtkjs file
                    viewer_html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>3D MODFLOW Model: {name}</title>
                        <script type="text/javascript" src="https://unpkg.com/vtk.js"></script>
                        <style>
                            body {{
                                margin: 0;
                                padding: 0;
                                width: 100vw;
                                height: 100vh;
                                overflow: hidden;
                            }}
                            #viewer {{
                                width: 100%;
                                height: 100%;
                                background-color: #eee;
                            }}
                            #info {{
                                position: absolute;
                                top: 10px;
                                left: 10px;
                                background-color: rgba(255, 255, 255, 0.8);
                                padding: 10px;
                                border-radius: 5px;
                                font-family: Arial, sans-serif;
                            }}
                        </style>
                    </head>
                    <body>
                        <div id="viewer"></div>
                        <div id="info">
                            <h2>3D MODFLOW Model: {name}</h2>
                            <p>Vertical Exaggeration: {vertical_exaggeration}x</p>
                            <p>Click and drag to rotate. Scroll to zoom.</p>
                        </div>
                        <script>
                            const container = document.getElementById('viewer');
                            const viewer = new vtk.Viewer(container);
                            viewer.addModel('{name}_3d_model.vtkjs');
                        </script>
                    </body>
                    </html>
                    """
                    
                    viewer_path = os.path.join(output_dir, f"{name}_3d_model_viewer.html")
                    with open(viewer_path, 'w') as f:
                        f.write(viewer_html)
                    print(f"Simple HTML viewer saved to: {viewer_path}")
            
            pyvista_success = True
            
        except Exception as e:
            print(f"PyVista visualization failed: {e}")
            pyvista_success = False
            
    except Exception as e:
        print(f"Failed to create terrain mesh: {e}")
        pyvista_success = False
    
    # If PyVista failed, use matplotlib as fallback
    if not pyvista_success and (save_screenshot or save_html):
        try:
            print("Using matplotlib for visualization...")
            
            # Get dimensions
            rows, cols = dem_data.shape
            
            # Create coordinate meshgrids
            y, x = np.mgrid[0:rows, 0:cols]
            
            # Apply geotransform to get real-world coordinates
            world_x = dem_transform.c + x * dem_transform.a + y * dem_transform.b
            world_y = dem_transform.f + x * dem_transform.d + y * dem_transform.e
            
            # Create output directory
            output_dir = os.path.join(base_path, "visualization")
            os.makedirs(output_dir, exist_ok=True)
            
            # Create and save a basic contour plot of the DEM
            plt.figure(figsize=(12, 10))
            plt.contourf(world_x, world_y, dem_data, cmap='terrain', levels=20)
            plt.colorbar(label='Elevation (m)')
            plt.title(f'DEM Contour Map: {name}')
            plt.savefig(os.path.join(output_dir, f"{name}_dem_contour.png"))
            plt.close()
            
            # Create a cross-section view
            plt.figure(figsize=(14, 8))
            mid_row = dem_data.shape[0] // 2
            
            # Plot the cross-section
            x_coords = world_x[mid_row, :]
            plt.plot(x_coords, z_values[0][mid_row, :], 'k-', label='Surface')
            plt.plot(x_coords, z_values[1][mid_row, :], 'b-', label='Water Table')
            plt.plot(x_coords, z_values[2][mid_row, :], 'g-', label='Bottom of Aquifer 1')
            plt.plot(x_coords, z_values[3][mid_row, :], 'r-', label='Bottom of Aquifer 2')
            
            plt.fill_between(x_coords, z_values[0][mid_row, :], z_values[1][mid_row, :], color='tan', alpha=0.5, label='Vadose Zone')
            plt.fill_between(x_coords, z_values[1][mid_row, :], z_values[2][mid_row, :], color='lightblue', alpha=0.5, label='Aquifer 1')
            plt.fill_between(x_coords, z_values[2][mid_row, :], z_values[3][mid_row, :], color='blue', alpha=0.5, label='Aquifer 2')
            
            plt.title(f'Cross-section View: {name}')
            plt.xlabel('Distance (m)')
            plt.ylabel('Elevation (m)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{name}_cross_section.png"))
            plt.close()
            
            # Try to create a simplified 3D view using matplotlib
            try:
                # Subsample the data to make it manageable for matplotlib 3D
                subsample = 10  # Use every 10th point
                
                fig = plt.figure(figsize=(14, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                # Plot surface as wireframe
                ax.plot_wireframe(
                    world_x[::subsample, ::subsample],
                    world_y[::subsample, ::subsample],
                    dem_data[::subsample, ::subsample],
                    color='brown', alpha=0.7, label='Surface'
                )
                
                # Plot water table
                ax.plot_wireframe(
                    world_x[::subsample, ::subsample],
                    world_y[::subsample, ::subsample],
                    z_values[1][::subsample, ::subsample],
                    color='blue', alpha=0.4, label='Water Table'
                )
                
                ax.set_title(f'3D Model View: {name}')
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Elevation (m)')
                plt.savefig(os.path.join(output_dir, f"{name}_3d_simple.png"))
                plt.close()
                
                print(f"Matplotlib visualizations saved to: {output_dir}")
                
            except Exception as e:
                print(f"Failed to create 3D matplotlib plot: {e}")
                
        except Exception as e:
            print(f"Matplotlib visualization failed: {e}")
            
    return True


def plot_cross_section(base_path, model_name, start_point, end_point):
    """
    Create a cross-section visualization of a MODFLOW model.
    
    Parameters:
    -----------
    base_path : str
        Path to the model directory.
    model_name : str
        Name of the model.
    start_point : tuple
        (x, y) coordinates of the start point.
    end_point : tuple
        (x, y) coordinates of the end point.
    """
    # This function can be implemented to show cross-sections of the model
    pass
