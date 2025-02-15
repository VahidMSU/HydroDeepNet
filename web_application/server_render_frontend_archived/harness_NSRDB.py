import h5py
import os

# Define input path
input_path = "/data/SWATGenXApp/GenXAppData/NSRDB/"
files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.h5') and "filtered" not in f]

# List of keys to keep
keys_to_keep = {"ghi", "wind_speed", "relative_humidity", "time_index", "meta"}

def filter_hdf5(file):
    output_path = file.replace(".h5", "_filtered.h5")

    with h5py.File(file, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        for key in f_in.keys():
            if key in keys_to_keep:
                f_out.create_dataset(key, data=f_in[key])  # No compression
                
                # Copy attributes if available
                for attr in f_in[key].attrs:
                    f_out[key].attrs[attr] = f_in[key].attrs[attr]

        # Copy global attributes
        for attr in f_in.attrs:
            f_out.attrs[attr] = f_in.attrs[attr]

    # Optional: Replace original file with filtered version
    os.remove(file)

    print(f"Filtered: {file} -> {output_path}")

# Process all files
for file in files:
    filter_hdf5(file)

print("Filtering completed for all files.")
