import h5py
import matplotlib.pyplot as plt
path = "Z:/PRISM/PRISM.h5"

with h5py.File(path, 'r') as f:
    print(f.keys())
    h = f['ppt']
    print(h['1990'])
    ## plot one slice
    plt.imshow(h['1990'][1,:, :])
    print(len(f['coords/lat'][:]))
    plt.show()