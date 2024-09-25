import torch
import torch.nn as nn
def preprocess_static_data(numerical_data, categorical_data):
    numerical_tensors = [torch.tensor(nd, dtype=torch.float32) for nd in numerical_data]
    categorical_tensors = [torch.tensor(cd, dtype=torch.long) for cd in categorical_data]

    combined_input = torch.stack(numerical_tensors + categorical_tensors, dim=0)
    return combined_input

def preprocess_climate_data(pr, tmax, tmin):
    def standardize(tensor):
        valid_mask = tensor != -999
        mean_val = torch.mean(tensor[valid_mask])
        std_val = torch.std(tensor[valid_mask])
        tensor[valid_mask] = (tensor[valid_mask] - mean_val) / std_val
        return tensor
    ## the aim of standardize is to normalize the data
    pr = standardize(torch.tensor(pr, dtype=torch.float32))
    tmax = standardize(torch.tensor(tmax, dtype=torch.float32))
    tmin = standardize(torch.tensor(tmin, dtype=torch.float32))

    return torch.stack([pr, tmax, tmin], dim=1)
def preprocess_recharge_rainfall_data(array3d):

    ## the aim of standardize is to normalize the data
    array3d = torch.tensor(array3d, dtype=torch.float32)
    #recharge = torch.tensor(recharge, dtype=torch.float32)
    return array3d