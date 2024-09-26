import ee
import requests
import h5py
import rasterio
import os
import matplotlib.pyplot as plt

import datetime

def get_monthly_modis_et(lon_min, lat_min, lon_max, lat_max, start_date, end_date, parameter='ET', scale=250, crs='EPSG:26990', width=1849, height=1458):
    # Convert strings to datetime objects
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    # Loop through each month in the specified range
    current = start
    while current <= end:
        # Define the start and end of the current month
        month_start = current.strftime('%Y-%m-%d')
        month_end = (current + datetime.timedelta(days=32)).replace(day=1) - datetime.timedelta(days=1)
        month_end = month_end.strftime('%Y-%m-%d')

        # Call the get_modis_et function for each month
        print(f"Processing {month_start} to {month_end}...")
        url = get_modis_et(lon_min, lat_min, lon_max, lat_max, start_date=month_start, end_date=month_end, scale=scale, crs=crs)

        # Move to the next month
        current = (current + datetime.timedelta(days=32)).replace(day=1)




def get_modis_et(lon_min, lat_min, lon_max, lat_max, start_date='2023-01-01', end_date='2023-06-30', parameter='ET', scale=250, project='ee-vahidr32', crs='EPSG:26990'):
    # Authenticate and initialize Earth Engine
    ee.Authenticate()
    ee.Initialize(project=project)

    # Define the region of interest using the reference raster bounds
    region = ee.Geometry.Polygon([
        [[lon_min, lat_min], [lon_max, lat_min], [lon_max, lat_max], [lon_min, lat_max], [lon_min, lat_min]]
    ])

    # Load the MODIS ET dataset (MOD16A2) and filter by the date range
    try:
        version = '061'
        modisET = ee.ImageCollection('MODIS/061/MOD16A2') \
                .filterDate(start_date, end_date) \
                .select(parameter)
        
                # You can use .mean(), .sum(), or any aggregation method
        clippedET = modisET.mean().clip(region)

        # Export the image to Google Drive or your system, matching the reference raster metadata
        task = ee.batch.Export.image.toDrive(
            image=clippedET,
            description=f'MODIS_ET_{start_date}_to_{end_date}',
        #    scale=scale,
            region=region.getInfo()['coordinates'],
            crs=crs,
            fileFormat='GeoTIFF',
            dimensions=f'{width}x{height}'
        )

        task.start()

        print(f"Export task started for {start_date} to {end_date}. Check your Google Drive for the output file.")

        # Download the file via Earth Engine URL (optional step)
        url = clippedET.getDownloadURL({
        #   'scale': scale,
            'region': region.getInfo()['coordinates'],
            'crs': crs,
            'format': 'GeoTIFF', 
            'dimensions': f'{width}x{height}'
        })

        print('Download URL:', url)
    except Exception as e:
        version = '006' 
        modisET = ee.ImageCollection('MODIS/006/MOD16A2') \
                .filterDate(start_date, end_date) \
                .select(parameter)  
        
        # You can use .mean(), .sum(), or any aggregation method
        clippedET = modisET.mean().clip(region)

        # Export the image to Google Drive or your system, matching the reference raster metadata
        task = ee.batch.Export.image.toDrive(
            image=clippedET,
            description=f'MODIS_ET_{start_date}_to_{end_date}',
        #    scale=scale,
            region=region.getInfo()['coordinates'],
            crs=crs,
            fileFormat='GeoTIFF',
            dimensions=f'{width}x{height}'
        )

        task.start()

        print(f"Export task started for {start_date} to {end_date}. Check your Google Drive for the output file.")

        # Download the file via Earth Engine URL (optional step)
        url = clippedET.getDownloadURL({
        #   'scale': scale,
            'region': region.getInfo()['coordinates'],
            'crs': crs,
            'format': 'GeoTIFF', 
            'dimensions': f'{width}x{height}'
        })

        print('Download URL:', url)


    
    response = requests.get(url)

    os.makedirs("downloads", exist_ok=True)
    save_response_content(response, f"MODIS_ET_{start_date}_to_{end_date}_{version}_250m.tif")

    return url

def plot_image(img, name):
    plt.imshow(img, cmap='viridis')
    plt.colorbar()
    os.makedirs("figs", exist_ok=True)
    plt.savefig(f"figs/{name.replace('.tif', '.png')}")    
    plt.close()

    
def save_response_content(response, name):
    # Download the file from the URL
    
    destination = f"downloads/{name}"
    # Save it to your local machine
    with open(destination, 'wb') as file:
        file.write(response.content)



    # Plot the image and check the projection
    with rasterio.open(destination) as src:
        img = src.read(1)
        print(f"shape: {img.shape}")
        print(f"Projection of the saved image: {src.crs}")
        print(f"Image shape: {img.shape}")
        plot_image(img, name)



def get_raster_info(reference_raster):
    with rasterio.open(reference_raster) as src:
        src = src.read(1)
        ## size
        width = src.shape[1]
        height = src.shape[0]
        


    # Open the reference raster and extract its metadata
    with h5py.File("/data/MyDataBase/HydroGeoDataset_ML_250.h5", "r") as f:
        lon = f['lon_250m'][:]
        lat = f['lat_250m'][:]

        ## plot the raster
        plt.imshow(src, cmap='viridis')
        plt.colorbar()
        plt.savefig("figs/reference_raster.png")
        plt.close() 

    lon_min = lon[lon != -999].min().astype(float) - 0.2456
    lon_max = lon[lon != -999].max().astype(float) 
    lat_min = lat[lat != -999].min().astype(float) 
    lat_max = lat[lat != -999].max().astype(float) + 0.028

    return width, height, lon_min, lon_max, lat_min, lat_max

def cleanup():
    os.system("rm -rf downloads/*")
    os.system("rm -rf figs/*")

if __name__ == "__main__":
    cleanup()
    reference_raster = "/data/MyDataBase/SWATGenXAppData/all_rasters/DEM_250m.tif"
    width, height, lon_min, lon_max, lat_min, lat_max = get_raster_info(reference_raster)

    start_date = '2001-01-01'
    end_date = '2023-12-30'
    get_monthly_modis_et(lon_min, lat_min, lon_max, lat_max, start_date, end_date, parameter='ET', scale=250, crs='EPSG:26990', width=width, height=height)



    print(f"Width: {width}")
    print(f"Height: {height}")
    print(f"lon_min: {lon_min}")
    print(f"lon_max: {lon_max}")
    print(f"lat_min: {lat_min}")
    print(f"lat_max: {lat_max}")

    print("Done!")