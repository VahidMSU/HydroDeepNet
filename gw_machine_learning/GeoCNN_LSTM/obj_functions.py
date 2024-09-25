from sympy import im
import torch
from torch import nn

def criterion(mask, preds, targets):
    # Create a mask to handle missing values (-999)
    # Ensure mask has the same dimensions as preds and targets
    mask = mask.expand_as(preds)
    mask = mask.bool()
    valid_preds = preds[mask]
    valid_targets = targets[mask]

    # Replace -999 values in targets with NaN for masking
    valid_targets[valid_targets == -999] = float('nan')
    valid_mask = ~torch.isnan(valid_targets)

    # Filter out non-positive values to avoid log issues
    valid_preds = valid_preds[valid_mask]
    valid_targets = valid_targets[valid_mask]

    # Ensure there are valid elements left after masking
    non_positive_mask = (valid_preds > 0) & (valid_targets > 0)
    valid_preds = valid_preds[non_positive_mask]
    valid_targets = valid_targets[non_positive_mask]

    # Calculate the mean squared error based on log
    mse = nn.MSELoss()
    #log_preds = torch.log(valid_preds)
    #log_targets = torch.log(valid_targets)
    return mse(valid_preds, valid_targets)

def masked_mse_loss_group(output, target, mask, groups):

    ########size of output: torch.Size([1, 15, 1, 308, 339])
    #########size of target: torch.Size([1, 15, 1, 308, 339])
    # Initialize a list to store losses for each group
    group_losses = []

    # Get the unique group identifiers
    unique_groups = groups.unique()

    for group in unique_groups:
        # Create a mask for the current group
        group_mask = (groups == group)

        # Apply the group mask to output, target, and the original mask
        group_output = torch.masked_select(output, group_mask)
        group_target = torch.masked_select(target, group_mask)
        group_valid_mask = torch.masked_select(mask, group_mask)

        # Compute the MSE loss for the current group
        diff = group_output - group_target
        group_loss = diff ** 2  # MSE loss

        # Apply the validation mask and calculate the average loss for this group
        if group_valid_mask.sum() > 0:  # Avoid division by zero
            normalized_group_loss = torch.sum(group_loss * group_valid_mask) / torch.sum(group_valid_mask)
            group_losses.append(normalized_group_loss)
        else:
            continue  # Skip this group if there are no valid elements

    # Compute the mean of the losses across all groups
    if group_losses:  # Check if there are any losses computed
        return torch.mean(torch.stack(group_losses))
    else:
        return torch.tensor(0.0, device=output.device)  # Return zero if no valid groups were processed