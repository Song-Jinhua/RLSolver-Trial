import os
import sys
import time
import torch as th


from graph_utils import load_graph_list, GraphList, load_tsp_graph_from_file
from graph_utils import build_adjacency_bool, build_adjacency_indies, obtain_num_nodes, build_adjacency_matrix
from graph_utils import update_xs_by_vs, gpu_info_str, evolutionary_replacement

TEN = th.Tensor


class SimulatorTSP:
    def __init__(self, graph_file: str, device=th.device('cpu')):
        self.device = device
        self.int_type = th.long

        '''Load the graph from file'''
        graph_list: GraphList = load_tsp_graph_from_file(graph_file)

        '''Build adjacency matrix'''
        self.adjacency_matrix = build_adjacency_matrix(graph_list=graph_list, if_bidirectional=True).to(device)

        '''Initialize nodes and edges'''
        self.num_nodes = obtain_num_nodes(graph_list)
        self.num_edges = len(graph_list)

    # def obj(self, xs: TEN) -> TEN:
    #     """
    #     Calculate the QUBO objective for TSP, which is the total tour length using vectorized operations.
    #     """
    #     num_sims = xs.shape[0]
    #     num_nodes = self.num_nodes
    #     xs = xs.view(num_sims, num_nodes, num_nodes)  # Reshape to (num_sims, num_nodes, num_nodes)
    #
    #     # Calculate distances between consecutive cities
    #     distances = th.zeros((num_sims,), dtype=th.float32, device=self.device)
    #
    #     for i in range(num_nodes - 1):
    #         distances += (xs[:, i, :] @ self.adjacency_matrix) * xs[:, i + 1, :] # ERROR
    #
    #     # Add distance from last city back to the first city
    #     distances += (xs[:, num_nodes - 1, :] @ self.adjacency_matrix) * xs[:, 0, :]
    #
    #
    #     return distances

    def obj_for_loop(self, xs: TEN) -> TEN:
        """
        Calculate the QUBO objective for TSP, which is the total tour length using explicit loops.
        """
        num_sims, num_nodes_sq = xs.shape
        num_nodes = self.num_nodes
        xs = xs.view(num_sims, num_nodes, num_nodes)  # Reshape to (num_sims, num_nodes, num_nodes)

        distances = th.zeros((num_sims,), dtype=th.float32, device=self.device)

        for sim in range(num_sims):
            for i in range(num_nodes - 1):
                for j in range(num_nodes):
                    for k in range(num_nodes):
                        distances[sim] += self.adjacency_matrix[j, k] * xs[sim, i, j] * xs[sim, i + 1, k]

            # Add the distance from the last city back to the first city
            for j in range(num_nodes):
                for k in range(num_nodes):
                    distances[sim] += self.adjacency_matrix[j, k] * xs[sim, num_nodes - 1, j] * xs[sim, 0, k]

        return distances

    def generate_xs_randomly(self, num_sims):
        """Generate random valid binary solutions for TSP QUBO"""
        xs = th.zeros((num_sims, self.num_nodes, self.num_nodes), dtype=th.float32, device=self.device)

        for i in range(num_sims):
            perm = th.randperm(self.num_nodes)  # Random permutation of cities
            xs[i, th.arange(self.num_nodes), perm] = 1.0  # Set one `1` per row and per column

        return xs.view(num_sims, -1)  # Flatten to shape (num_sims, num_nodes^2)



