import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData

class LSTMHetGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_node_types, num_edge_types):
        super(LSTMHetGNN, self).__init__()
        self.lstm = nn.LSTM(in_channels, hidden_channels, batch_first=True)
        
        # Define heterogeneous graph convolution layers using SAGEConv
        self.conv1 = HeteroConv({
            ('node_type1', 'edge_type1', 'node_type2'): SAGEConv(hidden_channels, hidden_channels),
            ('node_type2', 'edge_type2', 'node_type1'): SAGEConv(hidden_channels, hidden_channels),
            # Add more types if needed
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('node_type1', 'edge_type1', 'node_type2'): SAGEConv(hidden_channels, out_channels),
            ('node_type2', 'edge_type2', 'node_type1'): SAGEConv(hidden_channels, out_channels),
            # Add more types if needed
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        # Apply LSTM on node features
        h_dict = {}
        for node_type, x in x_dict.items():
            h, _ = self.lstm(x.unsqueeze(0))
            h_dict[node_type] = h.squeeze(0)

        # Apply Heterogeneous Graph Convolution
        h_dict = self.conv1(h_dict, edge_index_dict)
        h_dict = {key: F.relu(h) for key, h in h_dict.items()}
        h_dict = self.conv2(h_dict, edge_index_dict)

        return h_dict

# Prepare your data
data = HeteroData()
data['node_type1'].x = torch.randn(10, 16)  # 10 nodes with 16 features
data['node_type2'].x = torch.randn(5, 16)   # 5 nodes with 16 features

data['node_type1', 'edge_type1', 'node_type2'].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]]) 
data['node_type2', 'edge_type2', 'node_type1'].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])

print("edge index",data.get_edge_index)

data = data.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMHetGNN(in_channels=16, hidden_channels=32, out_channels=2, num_node_types=2, num_edge_types=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    
    # Example target (random)
    target = torch.randn(10, 2).to(device)
    
    loss = criterion(out['node_type1'], target)
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
