import os 
import h5py


def test_swatplus_output(NAME, ver):
    """ 
    Check if the SWAT+ output files are generated correctly.
    """
    SWATplus_output = os.path.join(base_path, NAME, "SWAT_gwflow_MODEL", "Scenarios", f"verification_stage_{ver}", "SWATplus_output.h5")
    
    with h5py.File(SWATplus_output, "r") as f:
        # Define groups and their expected keys
        group_keys = {
            'Soil': ['alb_30m', 'awc_30m', 'bd_30m', 'caco3_30m', 'carbon_30m', 
                     'clay_30m', 'dp_30m', 'ec_30m', 'ph_30m', 'rock_30m', 
                     'silt_30m', 'soil_30m', 'soil_k_30m'],
            'DEM': ['dem', 'demslp'],
            'Landuse': ['landuse_30m']
        }
        print(f"hru_wb keys: {list(f['hru_wb_30m/2000/2/et'].shape)}")

        # Check each group
        for group, keys in group_keys.items():
            if group not in f:
                print(f"Group '{group}' does not exist for {NAME}")
                raise ValueError(f"Group '{group}' does not exist in {SWATplus_output}")

            print(f"Keys in {group}: {list(f[group].keys())}")
            
            # Check each key within the group
            for key in keys:
                if key not in f[group]:
                    print(f"{key} does not exist in group '{group}' for {NAME}")
                    raise ValueError(f"{key} does not exist in group '{group}'")

        # Validate 30m resolution data exists
        resolution_group = f"hru_wb_30m"
        if resolution_group not in f:
            print(f"{resolution_group} resolution output does not exist for {NAME}")
            raise ValueError(f"{resolution_group} resolution output does not exist")
        
        # Check time-series data within 30m resolution group
        for year in range(2001, 2020):
            for month in range(1, 13):
                time_group = f"{resolution_group}/{year}/{month}/"

                if time_group not in f:
                    print(f"Time group '{time_group}' does not exist for {NAME}")
                    raise ValueError(f"Time group '{time_group}' does not exist")
                
                # Validate required variables
                keys = f[time_group].keys()
                for variable_name in ['et', 'perc', 'precip', 'snofall', 'snomlt', 'surq_gen', 'wateryld']:
                    if variable_name not in keys:
                        print(f"{variable_name} does not exist in {time_group} for {NAME}")
                        raise ValueError(f"{variable_name} does not exist in {time_group}")


if __name__ == "__main__":

    """
    This script checks if the SWAT+ output files are generated correctly.
    if not, raise error.

    safe to run: yes

    """
    base_path = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/"

    NAMES = os.listdir(base_path)
    NAMES.remove("log.txt")
    for NAME in NAMES:
        for ver in range(0, 6):
            test_swatplus_output(NAME, ver)


            