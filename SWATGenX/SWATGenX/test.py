import os
import sys
import pandas as pd

path = "/data/SWATGenXApp/GenXAppData/gSSURGO/VPUID/0407/gSSURGO_0407_30m.tif"

import rasterio
import numpy as np
with rasterio.open(path) as src:
    print(src.crs)
    print(src.transform)
    print(src.count)
    print(src.width)
    print(src.height)
    print(src.dtypes)
    no_data_value = src.nodata
    data = src.read(1)

    ## now plot the raster
    import matplotlib.pyplot as plt
    plt.imshow(np.where(data == no_data_value, np.nan, data))
    plt.colorbar()
    plt.savefig("test.png")