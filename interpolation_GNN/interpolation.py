import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from libs.gnn_models import get_model
from libs.setup_logging import setup_logger
from libs.plotting import plot_variable, plot_scatter, plot_loss_curve
from libs.Hetero_data_gen import create_heterodata
import gc
from torch.cuda.amp import GradScaler
import os 
os.environ["OPENBLAS_NUM_THREADS"] = "64"
os.environ["MKL_NUM_THREADS"] = "64"  # If using Intel MKL
os.environ["OMP_NUM_THREADS"] = "64"  # OpenMP threads




def evaluate_model(model, x_dict, edge_index_dict, edge_attr_dict, data, grid_centroids_df, logger, loss_fn, var_to_predict):
    # Evaluation on validation, test, and unmonitored nodes
    model.eval()
    with torch.no_grad():
        # Validate on the validation nodes
        val_predictions = model(x_dict, edge_index_dict, edge_attr_dict)
        val_loss = loss_fn(val_predictions[data['centroid'].val_mask], x_dict['centroid'][data['centroid'].val_mask][:, 0].unsqueeze(1))

        # Test on the test nodes
        test_predictions = model(x_dict, edge_index_dict, edge_attr_dict)
        test_loss = loss_fn(test_predictions[data['centroid'].test_mask], x_dict['centroid'][data['centroid'].test_mask][:, 0].unsqueeze(1))

        # Predict for unmonitored nodes
        unmonitored_predictions = model(x_dict, edge_index_dict, edge_attr_dict)

    # Convert predictions to NumPy arrays
    val_predictions_np = val_predictions[data['centroid'].val_mask].cpu().numpy()
    test_predictions_np = test_predictions[data['centroid'].test_mask].cpu().numpy()
    unmonitored_predictions_np = unmonitored_predictions[data['centroid'].unmonitored_mask].cpu().numpy()

    # Get indices of True values in masks
    val_indices = torch.where(data['centroid'].val_mask)[0].cpu().numpy()
    test_indices = torch.where(data['centroid'].test_mask)[0].cpu().numpy()
    unmonitored_indices = torch.where(data['centroid'].unmonitored_mask)[0].cpu().numpy()

    # Assign predictions to the correct rows in the DataFrame
    predictions_df = grid_centroids_df.copy()

    # Fill in the predictions for validation, test, and unmonitored nodes
    predictions_df.loc[val_indices, f"predicted_{var_to_predict}"] = val_predictions_np
    predictions_df.loc[test_indices,  f"predicted_{var_to_predict}"] = test_predictions_np
    predictions_df.loc[unmonitored_indices,  f"predicted_{var_to_predict}"] = unmonitored_predictions_np
    plot_scatter(predictions_df, var_to_predict, logger, stage='evaluation')
    # Plot the updated predictions in the DataFrame
    plot_variable(predictions_df,  f"predicted_{var_to_predict}", logger, stage='evaluation')



def training_loop(epochs, model, optimizer, scheduler, loss_fn, x_dict, edge_index_dict, edge_attr_dict, data, grid_centroids_df, logger, var_to_predict, patience=10, accumulation_steps=2):
    
    clip_value = 100
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0  # Counter for early stopping
    
    # Initialize the gradient scaler for mixed precision training
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass using the full heterogeneous graph
        out = model(x_dict, edge_index_dict, edge_attr_dict)

        # Compute the loss only on the training nodes (for 'centroid' node type)
        loss = loss_fn(out[data['centroid'].train_mask], x_dict['centroid'][data['centroid'].train_mask][:, 0].unsqueeze(1))

        # Backward pass and optimization with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        train_losses.append(loss.item())

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_predictions = model(x_dict, edge_index_dict, edge_attr_dict)
            val_loss = loss_fn(val_predictions[data['centroid'].val_mask], x_dict['centroid'][data['centroid'].val_mask][:, 0].unsqueeze(1))
            val_losses.append(val_loss.item())

        # Step the scheduler
        scheduler.step(val_loss)

        print(f"Epoch {epoch}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

        # Early stopping logic
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0  # Reset the counter if validation loss improves
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_model.pth')  # Save the best model
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}. No improvement in validation loss for {patience} epochs.")
            logger.info(f"Early stopping at epoch {epoch}. No improvement in validation loss for {patience} epochs.")
            break  # Stop training

        # Free up memory at the end of each epoch
        gc.collect()
        torch.cuda.empty_cache()

    # Plot the training and validation loss curves
    plot_loss_curve(train_losses, val_losses, logger)

    # Load the best model for evaluation
    model.load_state_dict(torch.load('models/best_model.pth'))
    evaluate_model(model, x_dict, edge_index_dict, edge_attr_dict, data, grid_centroids_df, logger, loss_fn, var_to_predict)


def main(args):
    # Setup
    path = "/data/MyDataBase/HuronRiverPFAS/Huron_River_Grid_250m_with_features.pkl"
    logger = setup_logger("log.txt")
    grid_centroids_df = pd.read_pickle(path)
    logger.info(f"Columns: {list(grid_centroids_df.columns)}")
    logger.info(f"Shape: {grid_centroids_df.shape}")
    from libs.plotting import plot_variable  #
    var_to_predict = args['var_to_predict']
    plot_variable(grid_centroids_df, var_to_predict, logger, stage='initial')
    data = create_heterodata(grid_centroids_df, logger, var_to_predict)

    # Ensure everything is properly set in HeteroData
    print("========================")
    print(f"HeteroData: {data}")
    print("========================")

    # Model setup
    x_dict = data.x_dict  # Node features for different node types
    edge_index_dict = data.edge_index_dict  # Edge index for different edge types
    edge_attr_dict = data.edge_attr_dict  # Edge attributes, if available

    # Move the data to the device (GPU)
    device = torch.device('cuda:0')  # Using GPU
    x_dict = {key: x.to(device) for key, x in x_dict.items()}
    edge_index_dict = {key: edge_index.to(device) for key, edge_index in edge_index_dict.items()}
    edge_attr_dict = {key: edge_attr.to(device) for key, edge_attr in edge_attr_dict.items()} if edge_attr_dict else None
    GNNModel = get_model(args['model_to_use'])
    # Create the model and move it to the same device
    model = GNNModel(in_channels=x_dict['centroid'].shape[1], hidden_channels=args['hidden_channels'], out_channels=1, edge_attr_dim=edge_attr_dict[('centroid', 'connected_to', 'centroid')].shape[1]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args['patience'], verbose=True)
    loss_fn = torch.nn.MSELoss()
    
    # Training loop
    epochs = args['epochs']
    training_loop(epochs, model, optimizer, scheduler, loss_fn, x_dict, edge_index_dict, edge_attr_dict, data, grid_centroids_df, logger, var_to_predict)

if __name__ == "__main__":
    ### empty HeteroData_storage
    files = os.listdir('HeteroData_storage')
    #for f in files:
    #    os.remove(os.path.join('HeteroData_storage', f))
    args = {
        "model_to_use": 'GatedEdgePReLUGNN', #'GatedEdgePReLUGNN',
        "var_to_predict": 'obs_H_COND_1_250m', # obs_H_COND_1_250m  #obs_AQ_THK_1_250m
        "learning_rate": 0.02,
        "weight_decay": 0.005, 
        "hidden_channels": 32, 
        "epochs": 500, 
        "patience": 10,
        "accumulation_steps": 2
        }
    main(args)
