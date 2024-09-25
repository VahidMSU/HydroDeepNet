import torch
import torch.nn as nn
import torch.nn.functional as F

class Graph:
    def __init__(self, nodes, edges, node_features, edge_features=None):
        self.nodes = nodes
        self.edges = edges
        self.node_features = node_features
        self.edge_features = edge_features

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x, edge_index):
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = torch.matmul(edge_index, x)
        out = norm.view(-1, 1) * out
        out = self.linear(out)
        return F.relu(out)

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(in_features, hidden_features)
        self.layer2 = GCNLayer(hidden_features, out_features)
    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        return x

def train_model(model, graph, targets, optimizer, criterion, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(graph.node_features, graph.edges)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Example usage
if __name__ == "__main__":
    # Create a simple graph
    nodes = torch.arange(0, 4)
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    node_features = torch.randn((4, 3))  # 4 nodes with 3 features each
    graph = Graph(nodes, edges, node_features)
    # Define the model
    model = GCN(in_features=3, hidden_features=4, out_features=2)
    # Dummy targets for training
    targets = torch.randn((4, 2))
    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    # Train the model
    train_model(model, graph, targets, optimizer, criterion)