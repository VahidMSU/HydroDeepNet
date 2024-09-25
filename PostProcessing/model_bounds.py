###################### creating MODEL_BOUNDS   ################ Actve if you added new models
import os
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from concurrent.futures import ProcessPoolExecutor
from shapely.geometry import MultiPolygon, Polygon
import matplotlib.patches as mpatches

def process_wrapper(BASE_PATH, LEVEL, NAME):
    return process_model_boundary(BASE_PATH, LEVEL, NAME)


def process_model_boundary(BASE_PATH, LEVEL, NAME):
    if NAME == 'log.txt':
        return None
    path = f"{BASE_PATH}/SWAT_input/{LEVEL}/{NAME}/SWAT_MODEL/Watershed/Shapes/SWAT_plus_subbasins_modified.shp"
    if not os.path.exists(path):
        path = f"{BASE_PATH}/SWAT_input/{LEVEL}/{NAME}/SWAT_MODEL/Watershed/Shapes/SWAT_plus_subbasins.shp"

    subbasins= gpd.read_file(path).to_crs('EPSG:26990')
    buffered = subbasins.buffer(100)
    dissolved = buffered.unary_union
    ## remove holes from the geometry
    if dissolved.geom_type == 'MultiPolygon':
        dissolved = dissolved.convex_hull
    # Ensure there are no holes in the geometry
    if isinstance(dissolved, MultiPolygon):
        # Create a single convex hull if multiple polygons are present
        dissolved = dissolved.convex_hull
    elif isinstance(dissolved, Polygon) and not dissolved.is_valid:
        # Fix invalid single polygon
        dissolved = dissolved.buffer(0)

    # Convert to a valid polygon with no holes
    if isinstance(dissolved, Polygon):
        dissolved = Polygon(dissolved.exterior)

    print(f"Number of polygons in {NAME}: {len(dissolved) if isinstance(dissolved, MultiPolygon) else 1}")

    return (LEVEL, NAME, dissolved)

def import_model_bounds(BASE_PATH,LEVELS):

    model_id = []

    for LEVEL in LEVELS:

        NAMES = os.listdir(os.path.join(BASE_PATH, f'SWAT_input/{LEVEL}/'))
        ### only NAME with length less than 10
        NAMES = [NAME for NAME in NAMES if len(NAME) < 10]
        with ProcessPoolExecutor() as executor:
            # Create a list of tuples for the arguments
            args = [(BASE_PATH, LEVEL, NAME) for NAME in NAMES if NAME != 'log.txt']
            results = executor.map(process_wrapper, *zip(*args))

        model_id.extend(result for result in results if result is not None)

    return model_id


def plot_model_bounds(model_bounds, LEVEL):
    # Load and preprocess huc4 data
    huc4 = pd.read_pickle('model_bounds/huc8_model_bounds.pkl').to_crs('EPSG:4326')
    lakes = pd.read_pickle('model_bounds/lakes_model_bounds.pkl').to_crs('EPSG:4326')

    # Create a plot
    fig, ax = plt.subplots(figsize=(15, 10))

    # Plot huc4 boundaries
    huc4_boundary = huc4.boundary
    huc4_boundary.plot(ax=ax, linewidth=1, facecolor='none', edgecolor='black')

    # Convert model_bounds to the same CRS
    model_bounds = model_bounds.to_crs('EPSG:4326')

    # Calculate the area and classify based on the threshold of 1500 sqkm
    model_bounds['AREA'] = model_bounds.to_crs("EPSG:26990").geometry.area / 1e6  # Convert to sqkm

    large_models = model_bounds[model_bounds['AREA'] > 1500]
    small_models = model_bounds[model_bounds['AREA'] <= 1500]

    # Plot large models with yellow boundary with face color off
    large_models.plot(ax=ax, alpha=0.7,  edgecolor='red', linewidth=1, facecolor='red')
    lakes.plot(ax=ax, alpha=1,  edgecolor='black', linewidth=1, facecolor='skyblue')
    # Plot small models with blue boundary
    small_models.plot(ax=ax, alpha=0.6,  edgecolor='blue', linewidth=1, facecolor='blue')

    # Add labels, title, and custom legend
    plt.xlabel('Longitude', fontdict={'fontsize': 15})
    plt.ylabel('Latitude', fontdict={'fontsize': 15})
    plt.xlim(-87, -82)
    plt.ylim(41.5, 46)
    plt.title(f'# {len(model_bounds)} {LEVEL} model bounds')

    # Create custom legend
    huc8_patch = mpatches.Patch(color='black', label='HUC8 boundaries')
    blue_patch = mpatches.Patch(color='blue', label='models with WA <= 1500 km\u00b2')
    red_patch = mpatches.Patch(color='red', label='models with WA > 1500 km\u00b2')
    skyblue_patch = mpatches.Patch(color='skyblue', label='Great Lakes')
    ax.legend(handles=[huc8_patch, blue_patch, red_patch, skyblue_patch], title='Legend')

    # Save the plot
    plt.savefig(f'model_bounds/{LEVEL}_model_bounds.png', dpi=300)



if __name__ == '__main__':


    print('begin')
    BASE_PATH = '/data/MyDataBase/CIWRE-BAE/'
    LEVELS = ['huc12']
    model_bounds = import_model_bounds(BASE_PATH,LEVELS)
    model_bounds = pd.DataFrame(model_bounds, columns=['LEVEL', 'NAME', 'geometry'])
    model_bounds = gpd.GeoDataFrame(model_bounds, geometry='geometry', crs='EPSG:26990')
    model_bounds['AREA'] = model_bounds.geometry.area
    for LEVEL in LEVELS:
        plot_model_bounds(model_bounds, LEVEL)
        print(f"Model bounds columns: {model_bounds.columns}")
        model_bounds.to_pickle(f'model_bounds/model_bounds_{LEVEL}.pkl')
    print(f'model_bounds_{LEVEL}.pkl is saved')