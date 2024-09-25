from GeoNet.DeepResidualMLP import DeepResidualMLP
from GeoNet.ResidualMLP import ResidualMLP
from GeoNet.import_data import get_huc8_ranges

def select_model(config, input_dim):
    """
    Summary:
    Selects the model to be used for training based on the configuration file.

    Args:
        config (dict): The configuration dictionary containing the model parameters.

    Returns:
        nn.Module: The model to be used for training.
    """
    print(f"### select_model: {config['model']}")
    if config['model'] == 'DeepResidualMLP':
        model = DeepResidualMLP(input_dim)
    elif config['model'] == 'ResidualMLP':
        model = ResidualMLP(input_dim)

    return model

