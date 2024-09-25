import numpy as np
from joblib import Parallel, delayed

def get_neighbors(x, y, distance_threshold=1500, k=16, n_jobs=48):
    from sklearn.neighbors import BallTree
    # Stack x and y coordinates into a 2D array
    coords = np.vstack((x, y)).T
    
    # Build the BallTree for fast nearest neighbor search
    tree = BallTree(coords, metric='euclidean')
    
    # Query the tree for the k+1 nearest neighbors (including the point itself)
    dist, ind = tree.query(coords, k=k+1)
    
    # Function to process neighbors for a single node
    def process_node(i):
        return [neighbor for j, neighbor in enumerate(ind[i][1:]) if dist[i][j+1] <= distance_threshold][:k]
    
    # Use Parallel processing with joblib
    neighbors = Parallel(n_jobs=n_jobs)(delayed(process_node)(i) for i in range(len(x)))
    
    return neighbors
