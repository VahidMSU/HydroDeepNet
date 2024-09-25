import contextlib
from libs.utils import get_neighbors
from torch_geometric.data import HeteroData
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
import numpy as np
import os
import contextlib
from torch_geometric.data import HeteroData
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def create_edge_index(df, neighbors):
    """
    Creates the edge index and edge attributes including self-loops.
    """
    # Extract coordinates and DEM_250m values
    centroid_x, centroid_y = df.geometry.x, df.geometry.y
    dem_values = df['DEM_250m'].values

    # Create the edge index: add neighbor connections
    edge_index = [[i, neighbor] for i, neighbor_list in enumerate(neighbors) for neighbor in neighbor_list]

    # Add self-loops: Create edges from each node to itself
    num_nodes = len(df)
    self_loops = [[i, i] for i in range(num_nodes)]
    edge_index += self_loops

    # Create the edge attributes (distance and DEM difference)
    edge_attr = []
    for i, neighbor_list in enumerate(neighbors):
        for neighbor in neighbor_list:
            distance = np.sqrt((centroid_x[i] - centroid_x[neighbor]) ** 2 + (centroid_y[i] - centroid_y[neighbor]) ** 2)
            dem_difference = dem_values[i] - dem_values[neighbor]
            edge_attr.append([distance, dem_difference])

    # Add self-loop attributes (distance 0 and DEM difference 0 for self-loops)
    edge_attr.extend([[0.0, 0.0]] * num_nodes)

    return torch.tensor(edge_index, dtype=torch.long).t().contiguous(), torch.tensor(edge_attr, dtype=torch.float)





def create_masks(df, var_to_predict):
    """
    Creates training, validation, test, and unmonitored masks.
    """
    assert "COUNTY_250m" in df.columns, "COUNTY_250m column not found in DataFrame"


    observations = df[var_to_predict].values

    # Mask for unmonitored nodes (observations == -999)
    unmonitored_mask = observations == -999

    # Mask for monitored nodes (i.e., nodes with values above 0)
    monitored_mask = observations > 0

    # Split the monitored nodes into training (70%), validation (20%), and test (10%)
    monitored_indices = np.where(monitored_mask)[0]
    train_indices, temp_indices = train_test_split(monitored_indices, test_size=0.3, random_state=42, stratify=df["COUNTY_250m"].values[monitored_indices]) 
    val_indices, test_indices = train_test_split(temp_indices, test_size=1/3, random_state=42, stratify=df["COUNTY_250m"].values[temp_indices])

    # Convert masks to tensors
    num_nodes = len(df)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    unmonitored_mask_tensor = torch.tensor(unmonitored_mask, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask, unmonitored_mask_tensor



def preprocess_features(df, num_co_variable, cat_co_variable):
    """
    Normalizes numerical features and one-hot encodes categorical features, preserving -999 values.
    Returns the combined feature array.
    """
    # Preserve -999 during normalization
    num_features = df[num_co_variable].values
    scaler = StandardScaler()

    # Mask -999 values in numerical features
    mask = (num_features != -999)

    # Normalize only the values that are not -999
    num_features_scaled = np.where(mask, scaler.fit_transform(num_features), -999)

    # One-hot encode categorical variables, treating -999 as a valid category and converting all to strings
    df[cat_co_variable] = df[cat_co_variable].astype(str)  # Convert all categorical values to strings
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_features = encoder.fit_transform(df[cat_co_variable].replace('-999', 'missing').values)

    return np.hstack((num_features_scaled, cat_features))


def create_heterodata(df, logger, var_to_predict):
    """
    Main function to create the HeteroData object.
    Checks for existing saved data and creates node features, edge indices, and masks.
    """
    # Check if the data is already created and load it if available
    with contextlib.suppress(Exception):
        data = torch.load('HeteroData_storage/data.pt')
        logger.info('HeteroData object loaded from data.pt')
        return data

    # Get neighbors and create the edge index and attributes
    centroid_x, centroid_y = df.geometry.x, df.geometry.y
    neighbors = get_neighbors(centroid_x, centroid_y)
    edge_index, edge_attr = create_edge_index(df, neighbors)

    # Create HeteroData object
    data = HeteroData()

    # Define categorical and numerical features
    cat_co_variable = [
        'Soil_STATSGO_250m', 
        'Glacial_Landsystems_250m',
    #    'geomorphons_250m_250Dis',
    #    'gSURRGO_swat_250m',

    ]
    
    num_co_variable = [
        "kriging_stderr_H_COND_1_250m",
        "kriging_output_H_COND_1_250m",
        'lat_250m', 
        'lon_250m',
        'DEM_250m',

    ]
    
    # Preprocess features: Normalize numerical and one-hot encode categorical features
    node_features = preprocess_features(df, num_co_variable, cat_co_variable)
    
    # Include the target variable (var_to_predict) in the node features
    node_features = np.hstack((df[[var_to_predict]].values, node_features))

    # Convert node features to PyTorch tensor
    data['centroid'].x = torch.tensor(node_features, dtype=torch.float)

    # Set edge index and edge attributes
    data['centroid', 'connected_to', 'centroid'].edge_index = edge_index
    data['centroid', 'connected_to', 'centroid'].edge_attr = edge_attr

    # Create masks for training, validation, testing, and unmonitored nodes
    train_mask, val_mask, test_mask, unmonitored_mask = create_masks(df, var_to_predict)
    data['centroid'].train_mask = train_mask
    data['centroid'].val_mask = val_mask
    data['centroid'].test_mask = test_mask
    data['centroid'].unmonitored_mask = unmonitored_mask

    # Save the HeteroData object to a file for future use
    os.makedirs('HeteroData_storage', exist_ok=True)
    torch.save(data, 'HeteroData_storage/data.pt')
    logger.info('HeteroData object saved as data.pt')

    return data
