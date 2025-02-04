import h5py
import os
import multiprocessing

# Define file path
input_path = "/data/NSRDB/"

# Get list of HDF5 files
files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.h5')]

def compress_hdf5(file):
    output_path = file.replace('.h5', '_compressed.h5')

    with h5py.File(file, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        for key in f_in.keys():
            dset = f_out.create_dataset(
                key, 
                data=f_in[key],  # Directly copy data without loading into RAM
                compression="gzip", 
                compression_opts=9, 
                chunks=True  # Enable chunking
            )
            # Copy attributes
            for attr in f_in[key].attrs:
                dset.attrs[attr] = f_in[key].attrs[attr]

        # Copy metadata
        for attr in f_in.attrs:
            f_out.attrs[attr] = f_in.attrs[attr]

    os.remove(file)  # Remove original file
    print(f"Compressed: {file} -> {output_path}")

for file in files:
    compress_hdf5(file)

print("Compression completed for all files.")
