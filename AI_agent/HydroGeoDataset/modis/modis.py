import h5py
import numpy as np
try:
    from config import AgentConfig

    from modis.modis_utilities import (extract_modis_data, MODIS_PRODUCTS)
except ImportError:
    try:
        from config import AgentConfig

        from HydroGeoDataset.modis.modis_utilities import (extract_modis_data, MODIS_PRODUCTS)
    except ImportError:
        from AI_agent.config import AgentConfig

        from AI_agent.HydroGeoDataset.modis.modis_utilities import (extract_modis_data, MODIS_PRODUCTS)
class MODIS_dataset:
    def __init__(self, config):
        self.config = config
        self.database_path = AgentConfig.HydroGeoDataset_ML_250_path
        self.h5_group_name = f"MODIS/{config['data_product']}"
        self.start_year = config['start_year']
        self.end_year = config['end_year']

    def MODIS_ET(self):
        """
        Extract MODIS data for a given period (start_year to end_year).
        
        Returns:
            Numpy array containing the extracted data for the specified period
        """
        if self.start_year is None:
            assert 'start_year' in self.config, "start_year is not provided."
            self.start_year = self.config['start_year']
            
        if self.end_year is None:
            assert 'end_year' in self.config, "end_year is not provided."
            self.end_year = self.config['end_year']

        bounding_box = None
        if 'bounding_box' in self.config and self.config['bounding_box']:
            bounding_box = self.config['bounding_box']
        
        # Use the utility function to extract MODIS data
        return extract_modis_data(
            database_path=self.database_path,
            h5_group_name=self.h5_group_name,
            start_year=self.start_year,
            end_year=self.end_year,
            bounding_box=bounding_box
        )


if __name__ == "__main__":
    ### get MODIS Evapotranspiration data

    config = { 
        "RESOLUTION": 250,
        "huc8": None,
        "video": False,
        "aggregation": "annual",
        "start_year": 2000,
        "end_year": 2003,
        'bounding_box': [-85.444332, 43.658148, -85.239256, 44.164683],
        "data_product": "MOD15A2H_Fpar_500m",  #'MOD09GQ_sur_refl_b01', 'MOD09GQ_sur_refl_b02', 'MOD13Q1_EVI', 'MOD13Q1_NDVI', 'MOD15A2H_Fpar_500m', 'MOD15A2H_Lai_500m', 'MOD16A2_ET',
    }

    importer = MODIS_dataset(config)
    et_modis = importer.MODIS_ET()
    print(et_modis.shape)
