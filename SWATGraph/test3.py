import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create synthetic temporal heterograph data
def create_temporal_heterograph(num_nodes, num_time_steps, num_features):
    # Create node features for two types of nodes: 'user' and 'item'
    user_features = torch.randn((num_nodes, num_features))
    item_features = torch.randn((num_nodes, num_features))
    
    # Create edges with timestamps
    edges = []
    for t in range(num_time_steps):
        src_nodes = torch.randint(0, num_nodes, (num_nodes,))
        dst_nodes = torch.randint(0, num_nodes, (num_nodes,))
        timestamps = torch.full((num_nodes,), t)
        edges.append((src_nodes, dst_nodes, timestamps))
    
    return user_features, item_features, edges

# Generate synthetic data
num_nodes = 100
num_time_steps = 10
num_features = 16
user_features, item_features, edges = create_temporal_heterograph(num_nodes, num_time_steps, num_features)

# Create DGL heterograph with temporal edges
data_dict = {
    ('user', 'interacts', 'item'): (edges[0][0], edges[0][1]),
    ('item', 'interacted-by', 'user'): (edges[0][1], edges[0][0])
}
g = dgl.heterograph(data_dict)
g.nodes['user'].data['feat'] = user_features
g.nodes['item'].data['feat'] = item_features
g.edata['timestamp'] = edges[0][2]

# Move the graph and data to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
g = g.to(device)
g.nodes['user'].data['feat'] = g.nodes['user'].data['feat'].to(device)
g.nodes['item'].data['feat'] = g.nodes['item'].data['feat'].to(device)
g.edata['timestamp'] = g.edata['timestamp'].to(device)

# Define a simple model for node classification
class TemporalHeteroGraphModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(TemporalHeteroGraphModel, self).__init__()
        self.layer1 = dgl.nn.GraphConv(in_feats, hidden_feats)
        self.layer2 = dgl.nn.GraphConv(hidden_feats, out_feats)

    def forward(self, g, feat):
        h = F.relu(self.layer1(g, feat))
        h = self.layer2(g, h)
        return h

# Instantiate and train the model
in_feats = num_features
hidden_feats = 32
out_feats = 2  # Example: binary classification

model = TemporalHeteroGraphModel(in_feats, hidden_feats, out_feats).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Generate dummy labels for training
labels = torch.randint(0, 2, (num_nodes,)).to(device)

# Training loop
loss_list = []
for epoch in range(100):
    model.train()
    logits = model(g, g.nodes['user'].data['feat'])
    loss = criterion(logits, labels)
    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Evaluate the model
model.eval()
with torch.no_grad():
    logits = model(g, g.nodes['user'].data['feat'])
    _, predicted = torch.max(logits, 1)
    accuracy = (predicted == labels).sum().item() / num_nodes
    print(f'Accuracy: {accuracy * 100:.2f}%')

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(range(100), loss_list, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()
