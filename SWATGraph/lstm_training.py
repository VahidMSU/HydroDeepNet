import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import logging
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import matplotlib
from libs.lstm_prepare_data import prepare_data

matplotlib.set_loglevel('WARNING')

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

class TemporalGraphDataset(Dataset):
    def __init__(self, temporal_data, graph, target_channel_id, streamflow_data_path):
        self.data, self.targets = prepare_data(temporal_data, graph, target_channel_id, streamflow_data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)


class StreamflowPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super(StreamflowPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.relu(self.fc1(out[:, -1, :]))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class StreamflowPredictionModel:
    def __init__(self, temporal_data_path, graph_path, target_channel_id, streamflow_data_path):
        with open(temporal_data_path, 'rb') as f:
            self.temporal_data = pickle.load(f)
            logging.info(f"Temporal data with {len(self.temporal_data)} nodes loaded")
            for key, value in self.temporal_data.items():
                logging.info(f"Temporal data shape for each node: {value.shape} (time_steps, features)")
                break
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
            logging.info(f"Graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges loaded")
            logging.info(f"Graph data shape for each node: {self.graph.nodes[list(self.graph.nodes)[0]]}")

        self.target_channel_id = target_channel_id
        self.streamflow_data_path = streamflow_data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = StreamflowPredictor(input_dim=6, hidden_dim=512, output_dim=1).to(self.device)

    def train(self, epochs=100, batch_size=32, learning_rate=0.001, early_stopping_patience=50):
        dataset = TemporalGraphDataset(self.temporal_data, self.graph, self.target_channel_id, self.streamflow_data_path)
        if len(dataset) == 0:
            logging.error("The dataset is empty. Please check the target channel ID and data preparation.")
            return

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.MSELoss(reduction='mean')
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

        best_loss = float('inf')
        epochs_no_improve = 0

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs).squeeze()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            train_loss = total_loss / len(train_loader)
            val_loss = self.evaluate(test_loader, criterion)
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print('Early stopping')
                    break

    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs).squeeze()
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        self.model.train()
        return total_loss / len(dataloader)

    def predict(self, inputs):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            outputs = self.model(inputs).squeeze()
        return outputs.cpu().numpy()

    def plot_predictions(self, inputs, actuals):
        predictions = self.predict(inputs)
        plt.figure(figsize=(10, 6))
        plt.plot(actuals, label='Actual Streamflow')
        plt.plot(predictions, label='Predicted Streamflow')
        plt.xlabel('Time')
        plt.ylabel('Streamflow')
        plt.legend()
        plt.title('Predicted vs Actual Streamflow')
        nse_value = nse(torch.tensor(predictions), torch.tensor(actuals))
        plt.suptitle(f'NSE: {nse_value:.4f}')
        plt.savefig('predictions.png')


import torch

def nse(predictions, targets):
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


if __name__ == "__main__":

    name = "04115000"
    model_path = "/data/MyDataBase/SWATplus_by_VPUID/0000/huc12"
    temporal_data_path = f'{model_path}/{name}/Graphs/temporal_data.pkl'
    graph_path = f'{model_path}/{name}/Graphs/SWAT_plus_streams.gpickle'
    streamflow_data_path = f'{model_path}/{name}/streamflow_data/1_{name}.csv'

    target_channel_id = int(os.path.basename(streamflow_data_path).split('_')[0])
    logging.info(f"Target channel ID: {target_channel_id}")
    
    model = StreamflowPredictionModel(temporal_data_path, graph_path, target_channel_id, streamflow_data_path)
    model.train(epochs=500, batch_size=16, learning_rate=0.001)

    # Prepare data for prediction and plotting
    dataset = TemporalGraphDataset(model.temporal_data, model.graph, model.target_channel_id, model.streamflow_data_path)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    inputs, targets = next(iter(dataloader))
    inputs, targets = inputs.numpy(), targets.numpy()

    # Plot predictions against actual values
    model.plot_predictions(inputs, targets)
