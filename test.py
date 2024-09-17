import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
import networkx as nx
import numpy as np

# Parameters
num_nodes = 10
num_edges = 15
in_channels = 4 
hidden_channels = 32
out_channels = 16

# Generate a random graph
G = nx.gnm_random_graph(num_nodes, num_edges)

# Generate node features (water levels, inflow/outflow rates, valve positions)
node_features = torch.tensor(np.random.uniform(low=0.5, high=2.0, size=(num_nodes, in_channels)), dtype=torch.float)

# Convert NetworkX graph to PyTorch Geometric Data format
data = from_networkx(G)
data.x = node_features

# GCN Model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x  # Final node representations

# Policy Network Model
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

def environment_step(state, action):
    adjusted_outflow = state[:, 2] + action.squeeze()
    adjusted_outflow = torch.clamp(adjusted_outflow, 0, 1)

    new_water_level = state[:, 0] + state[:, 1] - adjusted_outflow
    new_water_level = torch.clamp(new_water_level, 0, 2)
    
    new_state = torch.stack((new_water_level, state[:, 1], adjusted_outflow, state[:, 3]), dim=1)    
    overflow_penalty = torch.sum(torch.clamp(new_water_level - 1.5, min=0))
    reward = -overflow_penalty

    return new_state, reward

# Initialize GCN and Policy Network
gnn = GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels)
policy_network = PolicyNetwork(input_dim=out_channels, output_dim=1)

# Training loop
optimizer = torch.optim.Adam(list(gnn.parameters()) + list(policy_network.parameters()), lr=0.001)
num_episodes = 100

for episode in range(num_episodes):
    node_representations = gnn(data)  # GCN forward pass
    
    # Mean pooling of node features to create a graph-level representation
    graph_representation = torch.mean(node_representations, dim=0)
    
    action = policy_network(graph_representation)  # Policy network forward pass
    
    new_state, reward = environment_step(node_representations, action)  # Environment update
    loss = -reward.mean()  # Negative reward to maximize reward
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f'Episode {episode}: Loss = {loss.item()}')
