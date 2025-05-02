"""
Interactive 3D Viewer for MODFLOW Models

This script provides an interactive 3D visualization of MODFLOW models
with controls for toggling layers, adjusting transparency, and exploring the model.
"""

import os
import sys
import argparse
import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5 import QtWidgets, QtCore
import matplotlib.pyplot as plt
from MODGenX.vis_3d_models import read_raster, convert_feet_to_meters, create_terrain_mesh

class MODFLOWViewer:
    """Interactive viewer for MODFLOW models."""
    
    def __init__(self, base_path, model_name, vertical_exaggeration=5.0):
        """
        Initialize the MODFLOW viewer.
        
        Parameters
        ----------
        base_path : str
            Path to the model directory
        model_name : str
            Name of the model
        vertical_exaggeration : float
            Vertical exaggeration factor
        """
        self.base_path = base_path
        self.model_name = model_name
        self.vertical_exaggeration = vertical_exaggeration
        self.mesh = None
        self.layer_actors = []
        self.layer_data = {}
        
        # Initialize UI components
        self.plotter = None
        self.app = None
        
    def load_data(self):
        """Load all necessary data for the visualization."""
        # Extract the watershed name
        parts = self.base_path.rstrip('/').split('/')
        self.name = None
        
        # Try to find the part that matches the expected pattern for watershed name
        for part in parts:
            if part.isdigit() or (part.startswith('0') and part[1:].isdigit()):
                if self.name is None or len(part) > len(self.name):
                    self.name = part
        
        # Check the expected position
        if len(parts) >= 3:
            expected_name_pos = -3
            expected_name = parts[expected_name_pos]
            if expected_name.isdigit() or (expected_name.startswith('0') and expected_name[1:].isdigit()):
                self.name = expected_name
        
        if not self.name:
            raise ValueError(f"Could not extract watershed name from path: {self.base_path}")
        
        # Construct file paths
        rasters_dir = os.path.join(self.base_path, "rasters_input")
        dem_file = os.path.join(self.base_path, "..", "DEM_250m.tif")
        domain_file = os.path.join(rasters_dir, "domain.tif")
        
        # Construct correct file paths using the watershed name
        aq1_thickness_file = os.path.join(rasters_dir, f"{self.name}_kriging_output_AQ_THK_1_250m.tif")
        aq2_thickness_file = os.path.join(rasters_dir, f"{self.name}_kriging_output_AQ_THK_2_250m.tif")
        swl_file = os.path.join(rasters_dir, f"{self.name}_kriging_output_SWL_250m.tif")
        
        print(f"Using watershed name: {self.name}")
        print(f"Loading DEM from: {dem_file}")
        
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
        
        # Create a mask of valid DEM points
        dem_mask = np.ones_like(dem_data, dtype=bool)
        dem_mask[dem_data == 0] = False
        dem_mask[np.isnan(dem_data)] = False
        if dem_nodata is not None:
            dem_mask[dem_data == dem_nodata] = False
        
        # Also mask by domain if available
        if domain_data is not None:
            domain_mask = domain_data > 0
            dem_mask = np.logical_and(dem_mask, domain_mask)
        
        # Create elevation layers
        surface_elev = dem_data
        gwl_elev = dem_data - swl_data_m
        bottom_aq1 = gwl_elev - aq1_data_m
        bottom_aq2 = bottom_aq1 - aq2_data_m
        
        # Apply the mask to all elevation layers
        masked_surface_elev = np.where(dem_mask, surface_elev, np.nan)
        masked_gwl_elev = np.where(dem_mask, gwl_elev, np.nan)
        masked_bottom_aq1 = np.where(dem_mask, bottom_aq1, np.nan)
        masked_bottom_aq2 = np.where(dem_mask, bottom_aq2, np.nan)
        
        # Store layer data for later use
        self.layer_data = {
            'dem_data': dem_data,
            'dem_transform': dem_transform,
            'dem_nodata': dem_nodata,
            'domain_data': domain_data,
            'dem_mask': dem_mask,
            'surface_elev': masked_surface_elev,
            'gwl_elev': masked_gwl_elev,
            'bottom_aq1': masked_bottom_aq1,
            'bottom_aq2': masked_bottom_aq2
        }
        
        # Create layer elevations with masked values (from top to bottom)
        z_values = [masked_surface_elev, masked_gwl_elev, masked_bottom_aq1, masked_bottom_aq2]
        
        # Create the 3D mesh with vertical exaggeration
        self.mesh = create_terrain_mesh(
            dem_data, 
            dem_transform, 
            z_values, 
            domain_data, 
            dem_nodata, 
            vertical_exaggeration=self.vertical_exaggeration
        )
        
        # Calculate the boundaries of each layer for slicing
        self.layer_boundaries = []
        for i in range(len(z_values) - 1):
            layer_top = z_values[i]
            layer_bottom = z_values[i+1]
            z_min = np.nanmin(layer_bottom)
            z_max = np.nanmax(layer_top)
            self.layer_boundaries.append((z_min, z_max))
    
    def create_layer_meshes(self):
        """Create separate meshes for each layer."""
        self.layer_meshes = []
        
        # Layer names and colors
        self.layer_names = ["Vadose Zone", "Aquifer 1", "Aquifer 2"]
        self.layer_colors = ["tan", "lightblue", "blue"]
        self.layer_opacities = [0.7, 0.6, 0.5]
        
        # Extract each layer using clip_box
        for i, (z_min, z_max) in enumerate(self.layer_boundaries):
            mesh_copy = self.mesh.copy()
            
            # Extract this layer using clip_box
            layer_slice = mesh_copy.clip_box(
                bounds=(
                    mesh_copy.bounds[0], mesh_copy.bounds[1],  # xmin, xmax
                    mesh_copy.bounds[2], mesh_copy.bounds[3],  # ymin, ymax
                    z_min, z_max                              # zmin, zmax
                ),
                invert=False
            )
            
            if layer_slice.n_cells > 0:
                print(f"Layer {i} ({self.layer_names[i]}) has {layer_slice.n_cells} cells")
                self.layer_meshes.append(layer_slice)
            else:
                print(f"Layer {i} ({self.layer_names[i]}) has no cells - skipping")
    
    def setup_interactive_window(self):
        """Set up the interactive visualization window with controls."""
        # Create a PyVista plotter with a Qt backend
        self.plotter = BackgroundPlotter(title=f"3D MODFLOW Model: {self.name}")
        
        # Create a menu for tools
        menu_bar = self.plotter.main_menu
        tools_menu = menu_bar.addMenu('Tools')
        
        # Add vertical exaggeration options to the tools menu
        def update_ve(value):
            """Update vertical exaggeration."""
            print(f"Changing vertical exaggeration to {value}")
            # This would require regenerating the mesh which is complex
            # For now, just show a message
            self.plotter.add_text(f"Vertical Exaggeration: {value}x", name="ve_text", position='upper_left')
        
        ve1_action = QtWidgets.QAction('Regenerate with VE=1x', self.plotter.app_window)
        ve1_action.triggered.connect(lambda: update_ve(1.0))
        tools_menu.addAction(ve1_action)
        
        ve5_action = QtWidgets.QAction('Regenerate with VE=5x', self.plotter.app_window)
        ve5_action.triggered.connect(lambda: update_ve(5.0))
        tools_menu.addAction(ve5_action)
        
        ve10_action = QtWidgets.QAction('Regenerate with VE=10x', self.plotter.app_window)
        ve10_action.triggered.connect(lambda: update_ve(10.0))
        tools_menu.addAction(ve10_action)
        
        # Add a menu for exporting HTML visualizations
        export_menu = menu_bar.addMenu('Export')
        
        # Add actions for exporting to HTML
        export_html_action = QtWidgets.QAction('Export as Interactive HTML', self.plotter.app_window)
        export_html_action.triggered.connect(self.export_html)
        export_menu.addAction(export_html_action)
        
        export_vtkjs_action = QtWidgets.QAction('Export as VTK.js', self.plotter.app_window)
        export_vtkjs_action.triggered.connect(self.export_vtkjs)
        export_menu.addAction(export_vtkjs_action)
        
        view_in_browser_action = QtWidgets.QAction('View in Browser', self.plotter.app_window)
        view_in_browser_action.triggered.connect(self.view_in_browser)
        export_menu.addAction(view_in_browser_action)
        
        # Create layer controls
        def make_toggle_func(idx):
            """Create a function that toggles a specific layer's visibility."""
            def toggle_layer(state):
                if idx < len(self.layer_actors):
                    self.layer_actors[idx].SetVisibility(state)
                    self.plotter.render()
            return toggle_layer
        
        def make_opacity_func(idx):
            """Create a function that changes a specific layer's opacity."""
            def set_layer_opacity(value):
                if idx < len(self.layer_actors):
                    opacity = value / 100.0
                    self.plotter.renderer.GetActors().GetItemAsObject(self.layer_actors[idx]).GetProperty().SetOpacity(opacity)
                    self.plotter.render()
            return set_layer_opacity
        
        # Create a dock widget with controls
        dock_widget = QtWidgets.QDockWidget("Layer Controls")
        controls_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        
        # Add controls for each layer
        for i, layer_name in enumerate(self.layer_names):
            if i < len(self.layer_meshes):
                # Layer group
                group_box = QtWidgets.QGroupBox(layer_name)
                group_layout = QtWidgets.QVBoxLayout()
                
                # Visibility checkbox
                check_box = QtWidgets.QCheckBox("Visible")
                check_box.setChecked(True)
                check_box.stateChanged.connect(make_toggle_func(i))
                group_layout.addWidget(check_box)
                
                # Opacity slider
                opacity_layout = QtWidgets.QHBoxLayout()
                opacity_layout.addWidget(QtWidgets.QLabel("Opacity:"))
                opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
                opacity_slider.setRange(0, 100)
                opacity_slider.setValue(int(self.layer_opacities[i] * 100))
                opacity_slider.valueChanged.connect(make_opacity_func(i))
                opacity_layout.addWidget(opacity_slider)
                group_layout.addLayout(opacity_layout)
                
                group_box.setLayout(group_layout)
                layout.addWidget(group_box)
        
        # Add a button to save a screenshot
        save_btn = QtWidgets.QPushButton("Save Screenshot")
        save_btn.clicked.connect(self.save_screenshot)
        layout.addWidget(save_btn)
        
        # Add buttons for HTML export to the controls widget
        html_box = QtWidgets.QGroupBox("Export Options")
        html_layout = QtWidgets.QVBoxLayout()
        
        # Export HTML button
        export_html_btn = QtWidgets.QPushButton("Export as Interactive HTML")
        export_html_btn.clicked.connect(self.export_html)
        html_layout.addWidget(export_html_btn)
        
        # Export VTK.js button
        export_vtkjs_btn = QtWidgets.QPushButton("Export as VTK.js")
        export_vtkjs_btn.clicked.connect(self.export_vtkjs)
        html_layout.addWidget(export_vtkjs_btn)
        
        # View in browser button
        view_browser_btn = QtWidgets.QPushButton("View in Browser")
        view_browser_btn.clicked.connect(self.view_in_browser)
        html_layout.addWidget(view_browser_btn)
        
        html_box.setLayout(html_layout)
        layout.addWidget(html_box)
        
        # Set the layout
        controls_widget.setLayout(layout)
        dock_widget.setWidget(controls_widget)
        
        # Add the dock widget to the window - fixed method
        self.plotter.app_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock_widget)
        
        # Add a text annotation about vertical exaggeration
        self.plotter.add_text(
            f"Vertical Exaggeration: {self.vertical_exaggeration}x", 
            name="ve_text",
            position='upper_left', 
            font_size=12
        )
    
    def visualize(self):
        """Create the interactive visualization."""
        self.load_data()
        self.create_layer_meshes()
        self.setup_interactive_window()
        
        # Add each layer to the plotter
        self.layer_actors = []
        for i, layer_mesh in enumerate(self.layer_meshes):
            actor = self.plotter.add_mesh(
                layer_mesh, 
                color=self.layer_colors[i], 
                opacity=self.layer_opacities[i], 
                show_edges=True,
                name=f"layer_{i}"
            )
            self.layer_actors.append(actor)
        
        # Set the camera for a good initial view
        self.plotter.view_isometric()
        
        # Show the plotter (this will start the event loop)
        self.plotter.show()
    
    def save_screenshot(self):
        """Save a screenshot of the current view."""
        output_dir = os.path.join(self.base_path, "visualization")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get current date/time for filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the screenshot
        filepath = os.path.join(output_dir, f"{self.name}_screenshot_{timestamp}.png")
        self.plotter.screenshot(filepath)
        print(f"Screenshot saved to: {filepath}")
        
        # Show a confirmation
        msg_box = QtWidgets.QMessageBox()
        msg_box.setText(f"Screenshot saved to: {filepath}")
        msg_box.exec_()
    
    def export_html(self):
        """Export the current view as an interactive HTML file."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        output_dir = os.path.join(self.base_path, "visualization")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a new plotter for export (to avoid modifying the current view)
        export_plotter = pv.Plotter(notebook=True)
        
        # Add each layer to the export plotter
        for i, layer_mesh in enumerate(self.layer_meshes):
            # Check if the layer is currently visible
            is_visible = self.layer_actors[i].GetVisibility()
            if is_visible:
                # Get current opacity
                opacity = self.plotter.renderer.GetActors().GetItemAsObject(
                    self.layer_actors[i]).GetProperty().GetOpacity()
                
                # Add to export plotter with same properties
                export_plotter.add_mesh(
                    layer_mesh,
                    color=self.layer_colors[i],
                    opacity=opacity,
                    show_edges=True
                )
        
        # Set the same camera position
        export_plotter.camera_position = self.plotter.camera_position
        
        # Add a title
        export_plotter.add_title(f"3D MODFLOW Model: {self.name} (VE: {self.vertical_exaggeration}x)")
        
        # Export as HTML
        html_path = os.path.join(output_dir, f"{self.name}_interactive_{timestamp}.html")
        export_plotter.export_html(html_path)
        
        # Show confirmation
        msg_box = QtWidgets.QMessageBox()
        msg_box.setText(f"Interactive HTML exported to: {html_path}")
        msg_box.exec_()
    
    def export_vtkjs(self):
        """Export the current view as a VTK.js file and create a viewer HTML."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        output_dir = os.path.join(self.base_path, "visualization")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a new plotter for export
        export_plotter = pv.Plotter()
        
        # Add each layer to the export plotter (only visible ones)
        for i, layer_mesh in enumerate(self.layer_meshes):
            # Check if the layer is currently visible
            is_visible = self.layer_actors[i].GetVisibility()
            if is_visible:
                # Get current opacity
                opacity = self.plotter.renderer.GetActors().GetItemAsObject(
                    self.layer_actors[i]).GetProperty().GetOpacity()
                
                # Add to export plotter with same properties
                export_plotter.add_mesh(
                    layer_mesh,
                    color=self.layer_colors[i],
                    opacity=opacity,
                    show_edges=True
                )
        
        # Set the same camera position
        export_plotter.camera_position = self.plotter.camera_position
        
        # Export as VTK.js
        vtkjs_path = os.path.join(output_dir, f"{self.name}_model_{timestamp}")
        export_plotter.export_vtkjs(vtkjs_path)
        
        # Create a viewer HTML that can load the vtkjs file
        viewer_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>3D MODFLOW Model: {self.name}</title>
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
                .layer-control {{
                    margin-top: 5px;
                }}
            </style>
        </head>
        <body>
            <div id="viewer"></div>
            <div id="info">
                <h2>3D MODFLOW Model: {self.name}</h2>
                <p>Vertical Exaggeration: {self.vertical_exaggeration}x</p>
                <p>Click and drag to rotate. Scroll to zoom.</p>
                <p>Exported on: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
            </div>
            <script>
                const container = document.getElementById('viewer');
                const viewer = new vtk.Viewer(container);
                viewer.addModel('{os.path.basename(vtkjs_path)}.vtkjs');
            </script>
        </body>
        </html>
        """
        
        viewer_path = os.path.join(output_dir, f"{self.name}_viewer_{timestamp}.html")
        with open(viewer_path, 'w') as f:
            f.write(viewer_html)
            
        # Show confirmation
        msg_box = QtWidgets.QMessageBox()
        msg_box.setText(f"VTK.js exported to: {vtkjs_path}.vtkjs\nViewer HTML: {viewer_path}")
        msg_box.exec_()
    
    def view_in_browser(self):
        """Export the current view and open it in a web browser."""
        import webbrowser
        import tempfile
        
        # Create a temporary HTML file
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            temp_html_path = f.name
        
        # Create a new plotter for export
        export_plotter = pv.Plotter(notebook=True)
        
        # Add each layer to the export plotter (only visible ones)
        for i, layer_mesh in enumerate(self.layer_meshes):
            # Check if the layer is currently visible
            is_visible = self.layer_actors[i].GetVisibility()
            if is_visible:
                # Get current opacity
                opacity = self.plotter.renderer.GetActors().GetItemAsObject(
                    self.layer_actors[i]).GetProperty().GetOpacity()
                
                # Add to export plotter with same properties
                export_plotter.add_mesh(
                    layer_mesh,
                    color=self.layer_colors[i],
                    opacity=opacity,
                    show_edges=True
                )
        
        # Set the same camera position
        export_plotter.camera_position = self.plotter.camera_position
        
        # Add a title
        export_plotter.add_title(f"3D MODFLOW Model: {self.name} (VE: {self.vertical_exaggeration}x)")
        
        # Export as HTML
        export_plotter.export_html(temp_html_path)
        
        # Open in browser
        webbrowser.open(f"file://{temp_html_path}")
        
        # Show confirmation
        msg_box = QtWidgets.QMessageBox()
        msg_box.setText(f"Opening in browser: {temp_html_path}")
        msg_box.exec_()


def main():
    """Main function to run the interactive viewer."""
    parser = argparse.ArgumentParser(description="Interactive 3D viewer for MODFLOW models")
    parser.add_argument("--path", default=None, help="Path to the model directory")
    parser.add_argument("--ve", type=float, default=5.0, help="Vertical exaggeration (default: 5.0)")
    args = parser.parse_args()
    
    # Default path if not provided
    if args.path is None:
        name = "04112500"
        username = "vahidr32"
        vpuid = "0405"
        level = "huc12"
        model = "MODFLOW_250m"
        base_path = f"/data/SWATGenXApp/Users/{username}/SWATplus_by_VPUID/{vpuid}/{level}/{name}/{model}/"
    else:
        base_path = args.path
    
    # Check if path exists
    if not os.path.exists(base_path):
        print(f"Error: Path does not exist: {base_path}")
        return 1
    
    # Create and run the viewer
    viewer = MODFLOWViewer(base_path, os.path.basename(base_path), vertical_exaggeration=args.ve)
    viewer.visualize()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
