import torch

def masked_mse_loss(output, target, mask, groups):
    # Apply mask to zero out losses where mask is False
    loss = (output - target)** 2
    return torch.sum(loss * mask) / torch.sum(mask)

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