import numpy as np
import networkx as nx
import pandas as pd

num_nodes = 10
num_edges = 15
time_steps = 10_000

G = nx.gnm_random_graph(num_nodes, num_edges)

np.random.seed(42)
water_levels = np.random.uniform(low=0.5, high=2.0, size=(time_steps, num_nodes))
inflow_rates = np.random.uniform(low=0.1, high=0.5, size=(time_steps, num_nodes))
outflow_rates = np.random.uniform(low=0.1, high=0.5, size=(time_steps, num_nodes))
valve_positions = np.random.uniform(low=0.0, high=1.0, size=(time_steps, num_nodes))

flow_capacities = np.random.uniform(low=1.0, high=3.0, size=(num_edges,))
distances = np.random.uniform(low=0.5, high=5.0, size=(num_edges,))

data = {
    'time_step': np.repeat(np.arange(time_steps), num_nodes),
    'node': np.tile(np.arange(num_nodes), time_steps),
    'water_level': water_levels.flatten(),
    'inflow_rate': inflow_rates.flatten(),
    'outflow_rate': outflow_rates.flatten(),
    'valve_position': valve_positions.flatten(),
}
df_nodes = pd.DataFrame(data)

edge_list = np.array(G.edges())
df_edges = pd.DataFrame({
    'edge_index': np.arange(num_edges),
    'source_node': edge_list[:, 0],
    'target_node': edge_list[:, 1],
    'flow_capacity': flow_capacities,
    'distance': distances,
})

print(df_nodes.head())
print(df_edges.head())
