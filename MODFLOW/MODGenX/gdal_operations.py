### we will have geospatioal operations here
import os
import numpy as np
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import shutil

class gdal_sa:
    def __init__(self, input_raster=None, output_raster=None):
        self.input_raster = input_raster
        self.output_raster = output_raster
        self.env = {}
        
    class Env:
        def __init__(self):
            self.workspace = None
            self.overwriteOutput = True
            self.snapRaster = None
            self.outputCoordinateSystem = None
            self.extent = None
            self.nodata = np.nan
    
    env = Env()
    
    @staticmethod
    def GetRasterProperties_management(raster_path, property_type):
        """Get raster properties similar to arcpy.GetRasterProperties_management"""
        class Result:
            def __init__(self, value):
                self._value = value
            def getOutput(self, index):
                return self._value
                
        ds = gdal.Open(raster_path)
        if property_type == "CELLSIZEX":
            gt = ds.GetGeoTransform()
            cell_size_x = abs(gt[1])
            return Result(str(cell_size_x))
        return Result(None)
    
    @staticmethod
    def Clip_management(in_raster, rectangle, out_raster, in_template_dataset=None, 
                        nodata_value="#", clipping_geometry="NONE", maintain_clipping_extent="NO_MAINTAIN_EXTENT"):
        """Clip a raster using a geometry or extent rectangle"""
        # Open the input raster
        input_ds = gdal.Open(in_raster)
        if input_ds is None:
            raise ValueError(f"Cannot open input raster: {in_raster}")
        
        # Create output directory if it doesn't exist
        out_dir = os.path.dirname(out_raster)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        # Decide if we're doing an extent-based clip or a shape-based clip
        if in_template_dataset is None:
            # Just using rectangle/extent clipping
            # Parse the rectangle string to get coordinates
            if isinstance(rectangle, str):
                minX, minY, maxX, maxY = map(float, rectangle.split())
            else:
                # Assume it's already parsed into coordinates
                minX, minY, maxX, maxY = rectangle
                
            # Create warp options for extent clipping
            options = gdal.WarpOptions(
                outputBounds=[minX, minY, maxX, maxY],
                dstNodata=float(nodata_value) if nodata_value != "#" else None
            )
            
            gdal.Warp(out_raster, input_ds, options=options)
            
        else:
            # Using shape-based clipping with a vector dataset
            clip_ds = ogr.Open(in_template_dataset)
            if clip_ds is None:
                raise ValueError(f"Cannot open clip feature: {in_template_dataset}")
                
            layer = clip_ds.GetLayer()
            
            # Use gdal_warp to clip the raster with the vector
            options = gdal.WarpOptions(
                cutlineDSName=in_template_dataset,
                cropToCutline=True,
                dstNodata=float(nodata_value) if nodata_value != "#" else None
            )
            
            gdal.Warp(out_raster, input_ds, options=options)
            clip_ds = None
        
        # Clean up
        input_ds = None
    
    @staticmethod
    def Delete_management(in_data):
        """Delete a dataset"""
        if os.path.exists(in_data):
            if os.path.isfile(in_data):
                os.remove(in_data)
            elif os.path.isdir(in_data):
                shutil.rmtree(in_data)
    
    @staticmethod
    def ProjectRaster_management(in_raster, out_raster, out_coor_system, 
                                resampling_type="NEAREST", cell_size=None, 
                                geographic_transform=None, in_coor_system=None):
        """Project a raster to a new coordinate system"""
        # Create output directory if it doesn't exist
        out_dir = os.path.dirname(out_raster)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        # Get target spatial reference
        if isinstance(out_coor_system, str):
            sr = osr.SpatialReference()
            sr.ImportFromEPSG(int(out_coor_system.split(':')[-1]))
        else:
            sr = out_coor_system
            
        # Set up resampling algorithm
        resampling_dict = {
            "NEAREST": gdal.GRA_NearestNeighbour,
            "BILINEAR": gdal.GRA_Bilinear,
            "CUBIC": gdal.GRA_Cubic,
            "CUBICSPLINE": gdal.GRA_CubicSpline
        }
        resampling = resampling_dict.get(resampling_type, gdal.GRA_NearestNeighbour)
        
        # Create warp options
        options = gdal.WarpOptions(
            dstSRS=sr.ExportToWkt(),
            resampleAlg=resampling
        )
        
        # Perform the projection
        gdal.Warp(out_raster, in_raster, options=options)
    
    @staticmethod
    def Resample_management(in_raster, out_raster, cell_size, resampling_type="NEAREST"):
        """Resample a raster to a new cell size"""
        # Create output directory if it doesn't exist
        out_dir = os.path.dirname(out_raster)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        # Parse cell size 
        if isinstance(cell_size, str) and " " in cell_size:
            x_cell, y_cell = map(float, cell_size.split())
        else:
            x_cell = y_cell = float(cell_size)
        
        # Set up resampling algorithm
        resampling_dict = {
            "NEAREST": gdal.GRA_NearestNeighbour,
            "BILINEAR": gdal.GRA_Bilinear,
            "CUBIC": gdal.GRA_Cubic,
            "CUBICSPLINE": gdal.GRA_CubicSpline
        }
        resampling = resampling_dict.get(resampling_type, gdal.GRA_NearestNeighbour)
        
        # Open input raster to get extent
        ds = gdal.Open(in_raster)
        gt = ds.GetGeoTransform()
        
        # Calculate new dimensions
        width = ds.RasterXSize
        height = ds.RasterYSize
        x_min = gt[0]
        y_max = gt[3]
        x_max = x_min + width * gt[1]
        y_min = y_max + height * gt[5]
        
        new_width = int((x_max - x_min) / x_cell)
        new_height = int((y_max - y_min) / y_cell)
        
        # Create warp options
        options = gdal.WarpOptions(
            width=new_width,
            height=new_height,
            outputBounds=[x_min, y_min, x_max, y_max],
            resampleAlg=resampling
        )
        
        # Perform the resampling
        gdal.Warp(out_raster, in_raster, options=options)
        
        ds = None
    
    @staticmethod
    def PolygonToRaster_conversion(in_features, value_field, out_raster, cell_assignment="CELL_CENTER", 
                                   priority_field="NONE", cellsize=None):
        """Convert polygon features to a raster dataset"""
        # Get the input shapefile
        vector_ds = ogr.Open(in_features)
        if vector_ds is None:
            raise ValueError(f"Cannot open input shapefile: {in_features}")
            
        layer = vector_ds.GetLayer()
        if layer is None:
            raise ValueError(f"Cannot get layer from shapefile: {in_features}")
            
        # Check if the value field exists in the layer
        layer_defn = layer.GetLayerDefn()
        field_index = layer_defn.GetFieldIndex(value_field)
        if field_index == -1:
            raise ValueError(f"Field '{value_field}' not found in the input shapefile")
        
        # Get the extent of the layer
        x_min, x_max, y_min, y_max = layer.GetExtent()
        
        # If cellsize is not specified, use a default value
        if cellsize is None:
            cellsize = 30.0
        else:
            cellsize = float(cellsize)
        
        # Calculate raster dimensions
        cols = int((x_max - x_min) / cellsize) + 1
        rows = int((y_max - y_min) / cellsize) + 1
        
        # Create the output raster
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(out_raster, cols, rows, 1, gdal.GDT_Float32)
        
        # Set the geotransform
        out_ds.SetGeoTransform((x_min, cellsize, 0, y_max, 0, -cellsize))
        
        # Set the projection from the layer
        sr = layer.GetSpatialRef()
        if sr:
            out_ds.SetProjection(sr.ExportToWkt())
        
        # Initialize the raster with nodata values
        band = out_ds.GetRasterBand(1)
        band.SetNoDataValue(-9999)
        band.Fill(-9999)
        
        # Rasterize the layer with specific options to ensure values are transferred correctly
        gdal.RasterizeLayer(
            out_ds, [1], layer, 
            options=[
                f"ATTRIBUTE={value_field}",
                "ALL_TOUCHED=FALSE"  # Only include cells whose center is within the polygon
            ]
        )
        
        # Flush to disk and close datasets
        out_ds.FlushCache()
        band = None
        out_ds = None
        vector_ds = None
        
        # Verify the output file exists and has valid data
        if not os.path.exists(out_raster):
            print(f"Warning: Output raster file was not created: {out_raster}")
        else:
            check_ds = gdal.Open(out_raster)
            if check_ds is not None:
                check_band = check_ds.GetRasterBand(1)
                stats = check_band.GetStatistics(0, 1)
                print(f"Raster statistics - Min: {stats[0]}, Max: {stats[1]}, Mean: {stats[2]}, StdDev: {stats[3]}")
                check_ds = None
            else:
                print(f"Warning: Cannot open output raster for validation: {out_raster}")

    @staticmethod
    def SpatialReference(epsg_code):
        """Create a spatial reference object from an EPSG code"""
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(epsg_code)
        return sr

    @staticmethod
    def Describe(dataset_path):
        """Similar to arcpy's Describe function, returns object with dataset properties"""
        class DatasetProperties:
            def __init__(self):
                self.spatialReference = None
                self.extent = None
                self.shapeType = None
        
        props = DatasetProperties()
        
        # Check file extension to determine dataset type
        file_ext = os.path.splitext(dataset_path)[1].lower()
        
        if file_ext in ['.tif', '.tiff', '.img', '.bil', '.dem']:
            # Handle raster datasets
            try:
                ds = gdal.Open(dataset_path)
                if ds is None:
                    return props
                
                # Get spatial reference
                wkt = ds.GetProjection()
                srs = osr.SpatialReference()
                srs.ImportFromWkt(wkt)
                props.spatialReference = srs
                
                # Get extent
                gt = ds.GetGeoTransform()
                x_min = gt[0]
                y_max = gt[3]
                x_max = x_min + gt[1] * ds.RasterXSize
                y_min = y_max + gt[5] * ds.RasterYSize
                
                # Create extent object similar to arcpy's extent
                class Extent:
                    def __init__(self, xmin, ymin, xmax, ymax):
                        self.XMin = xmin
                        self.YMin = ymin
                        self.YMax = ymax
                        self.XMax = xmax
                
                props.extent = Extent(x_min, y_min, x_max, y_max)
                
                ds = None
            except Exception as e:
                print(f"Error describing raster: {e}")
        
        elif file_ext in ['.shp', '.dbf', '.gdb']:
            # Handle vector datasets
            try:
                ds = ogr.Open(dataset_path)
                if ds is None:
                    return props
                
                layer = ds.GetLayer(0)
                srs = layer.GetSpatialRef()
                props.spatialReference = srs
                
                x_min, x_max, y_min, y_max = layer.GetExtent()
                props.extent = type('Extent', (), {'XMin': x_min, 'YMin': y_min, 'XMax': x_max, 'YMax': y_max})
                
                # Get shape type
                geom_type = layer.GetGeomType()
                if geom_type == ogr.wkbPoint or geom_type == ogr.wkbMultiPoint:
                    props.shapeType = "Point"
                elif geom_type == ogr.wkbLineString or geom_type == ogr.wkbMultiLineString:
                    props.shapeType = "Polyline"
                elif geom_type == ogr.wkbPolygon or geom_type == ogr.wkbMultiPolygon:
                    props.shapeType = "Polygon"
                
                ds = None
            except Exception as e:
                print(f"Error describing vector: {e}")
        
        return props

# Define an alias for compatibility
arcpy = gdal_sa()

