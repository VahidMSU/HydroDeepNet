import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from read_parameter_objective_sen_cal import load_parameters_models_performance
import matplotlib.pyplot as plt
import torch.nn.functional as F

class SWATDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).cuda()
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1).cuda()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ANNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, l2_reg=0.01):
        super(ANNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.l2_reg = l2_reg
        
        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x

    def _initialize_weights(self):
        # Initialize weights using Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_l2_loss(self):
        l2_loss = sum(torch.norm(param) for param in self.parameters())
        return self.l2_reg * l2_loss

    def count_parameters(self):
        # Count total number of parameters
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
def split_data(X, y, stage, name):
    unique_names = np.unique(name)
    first_half = unique_names[:len(unique_names)//2]
    second_half = unique_names[len(unique_names)//2:]
    single_random = unique_names[np.random.choice(len(unique_names), 1)]
    print(f"single_random: {single_random}")
    #X_train = X[(stage == 'sen') & (np.isin(name, single_random))]
    #y_train = y[(stage == 'sen') & (np.isin(name, single_random))]
    #X_test = X[(stage == 'cal') & (np.isin(name, single_random))]
    #y_test = y[(stage == 'cal') & (np.isin(name, single_random))]

    X_train = X[(np.isin(name, single_random))]
    y_train = y[(np.isin(name, single_random))]
    X_test = X[(np.isin(name, single_random))]
    y_test = y[(np.isin(name, single_random))]

    return X_train, X_test, y_train, y_test

def plot_predictions(model, X, y, device='cuda'):   
    model.eval()
    with torch.no_grad():
        inputs = X.clone().detach().to(device) if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32).to(device)
        outputs = model(inputs).cpu().numpy()  # Move tensor to CPU before converting to NumPy
        y = y.clone().detach().cpu().numpy() if isinstance(y, torch.Tensor) else y  # Ensure y is also on CPU and converted to NumPy if it's a tensor
        plt.scatter(y, outputs, color='red')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Predictions vs True Values')
        plt.savefig('parameter_learning/predictions.png')

def sensitivity_analysis(data, device, epochs=100, patience=10, learning_rate=0.001, batch_size=640):
    data = data.dropna()
    enc = OneHotEncoder()
    name_vpuid_encoded = enc.fit_transform(data[['NAME', 'VPUID']]).toarray()
    X_params = data.drop(['best_score', 'NAME', 'VPUID','stage'], axis=1).values
    X = np.hstack([X_params, name_vpuid_encoded])
    y = data['best_score'].values
    stage = data['stage'].values
    name = data['NAME'].values
    
    # Apply StandardScaler and MinMaxScaler
    continuous_features = X_params
    scaler = StandardScaler()
    continuous_features = scaler.fit_transform(continuous_features)
    
    # Combine the scaled continuous features with encoded categorical features
    X = np.hstack([continuous_features, name_vpuid_encoded])
    
    X_train, X_val, y_train, y_val = split_data(X, y, stage, name)
    train_dataset = SWATDataset(X_train, y_train)
    val_dataset = SWATDataset(X_val, y_val)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = ANNModel(X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    best_val_loss = float('inf')
    patience_counter = 0
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets) + model.get_l2_loss()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_dataloader)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'parameter_learning/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break
        model.train()
    ### load the best model and plot the predictions
    model.load_state_dict(torch.load('parameter_learning/best_model.pth'))
    plot_predictions(model, val_dataset.X, val_dataset.y, device)

    return "Sensitivity analysis completed"

def remove_outliers(data, threshold=3):
    data = data[data['best_score'] < 100].reset_index(drop=True)
    return data

def main():
    epochs = 200
    patience = 25
    learning_rate = 0.001
    batch_size = 30
    df_model = pd.read_csv("/home/rafieiva/MyDataBase/codes/PostProcessing/model_characteristics/SWAT_gwflow_MODEL/df_models.csv")
    df_model['NAME'] = df_model['NAME'].astype("int64")
    print(f"NAMES: {df_model['NAME'].unique()}")
    base_path = "/data/MyDataBase/SWATplus_by_VPUID"
    vpuid = "0000"
    data = load_parameters_models_performance(base_path, vpuid)
    data['NAME'] = data['NAME'].astype("int64") 
    data = data.merge(df_model, on='NAME', how='left')
    print(f"number of data before removing outliers: {len(data)}")
    data = remove_outliers(data)
    print(f"number of data after removing outliers: {len(data)}")   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sensitivities = sensitivity_analysis(data, device, epochs, patience, learning_rate, batch_size)
    print(sensitivities)

if __name__ == "__main__":
    main()
