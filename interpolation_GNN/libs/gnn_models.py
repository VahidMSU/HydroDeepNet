from libs.basic_GNN import GNNModel
from libs.GatedEdgePReLUGNN import GatedEdgePReLUGNN


def get_model(model_type):
    """
    Get the model based on the model type.
    """
    if model_type == 'GNNModel':
        return GNNModel
    elif model_type == 'GatedEdgePReLUGNN':
        return GatedEdgePReLUGNN
    else:
        raise ValueError(f"Model type {model_type} not recognized.")
    
