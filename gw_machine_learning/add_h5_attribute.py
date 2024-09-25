import h5py
import os
path = '/data/MyDataBase/'

h5_files = [x for x in os.listdir(path) if x.endswith('.h5')]

for h5_file in h5_files:
    with h5py.File(path + h5_file, 'r+') as f:
        f.attrs['author'] = 'Vahid Rafiei'
        f.attrs['email'] = 'rafieiva@msu.edu'
        f.attrs['date'] = '2023-06-18'
        f.attrs['crs'] = 'EPSG:26990'
        f.attrs['version'] = '4.12.0'