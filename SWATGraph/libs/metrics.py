import torch


def nse(predictions, targets) -> float:
    """
    Computes the Nash-Sutcliffe Efficiency (NSE) between predictions and targets.
    Args:
    - predictions (torch.Tensor): Predicted values.
    - targets (torch.Tensor): Observed/true values.
    Returns:
    - nse_value (float): Nash-Sutcliffe Efficiency value.
    """
    # Ensure predictions and targets are of the same shape
    assert predictions.shape == targets.shape, "Shape of predictions and targets must be the same"

    # Compute the mean of the observed data
    mean_observed = torch.mean(targets)

    # Compute the numerator and denominator for NSE
    numerator = torch.sum((targets - predictions) ** 2)
    denominator = torch.sum((targets - mean_observed) ** 2)

    # Compute NSE
    nse_value = 1 - (numerator / denominator)

    return nse_value.item()
