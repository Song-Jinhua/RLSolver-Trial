from graph_utils import GraphList, obtain_num_nodes
import os

ModelDir = './model'  # Model directory

class ConfigGraph:
    def __init__(self, graph_list: GraphList = None, graph_type: str = 'tsp', num_nodes: int = 0):
        num_nodes = num_nodes if num_nodes > 0 else obtain_num_nodes(graph_list=graph_list)

        self.graph_type = graph_type
        self.num_nodes = num_nodes

        '''train'''
        self.num_sims = 2 ** 5  # Number of parallel simulations (similar to batch_size)
        self.num_buffers = 4  # Number of buffer datasets needed for training
        self.buffer_size = 2 ** 12  # Number of pre-generated graphs in each dataset
        self.buffer_repeats = 64  # Number of times the data is reused
        self.buffer_dir = './buffer'  # Directory to store buffer data for training

        self.learning_rate = 2 ** -14  # Learning rate for the optimizer
        self.weight_decay = 0  # Weight decay for the optimizer
        self.show_gap = 2 ** 6  # Number of steps between showing training progress

        '''model'''
        self.num_heads = 8
        self.num_layers = 4
        self.mid_dim = 256
        self.inp_dim = num_nodes  # Input is the adjacency matrix
        self.out_dim = num_nodes * num_nodes  # Output is a flattened binary QUBO matrix for TSP
        sqrt_num_nodes = int(num_nodes ** 0.5)
        self.embed_dim = max(sqrt_num_nodes - sqrt_num_nodes % self.num_heads, 32)  # Length of node embedding vector after encoding


class ConfigPolicy:
    def __init__(self, graph_list: GraphList = None, graph_type: str = 'tsp', num_nodes: int = 0):
        num_nodes = num_nodes if num_nodes > 0 else obtain_num_nodes(graph_list=graph_list)

        self.graph_type = graph_type
        self.num_nodes = num_nodes

        '''train'''
        self.num_sims = 2 ** 3  # Number of initial solutions in LocalSearch
        self.num_repeats = 2 ** 4  # Number of times each initial solution is duplicated in LocalSearch
        self.num_searches = 2 ** 2  # Number of times noise is added in LocalSearch
        self.reset_gap = 2 ** 5  # Number of iterations before resetting and starting a new search
        self.num_iters = 2 ** 7  # Total number of search iterations
        self.num_sgd_steps = 2 ** 2  # Number of gradient descent steps per supervision signal
        self.entropy_weight = 4  # Weight of the policy entropy (controlled by an annealing schedule)

        self.learning_rate = 2 ** -10  # Learning rate for the optimizer
        self.weight_decay = 0  # Weight decay for the optimizer
        self.net_path = f"{ModelDir}/policy_net_{graph_type}_Node{num_nodes}.pth"  # Path to save the policy network

        self.show_gap = 2 ** 2  # Number of steps between showing training progress

        '''model'''
        self.num_heads = 8
        self.num_layers = 4
        self.mid_dim = 256
        self.inp_dim = num_nodes * num_nodes  # Input is a flattened QUBO matrix
        self.out_dim = 1  # Output is the probability associated with each node
        sqrt_num_nodes = int(num_nodes ** 0.5)
        self.embed_dim = max(sqrt_num_nodes - sqrt_num_nodes % self.num_heads, 32) # Length of the node embedding vector

    def load_net(self, net_path: str = '', device=None, if_valid: bool = False):
        import torch as th
        net_path = net_path if net_path else self.net_path
        device = device if device else th.device('cpu')

        from network import PolicyTRS
        net = PolicyTRS(inp_dim=self.inp_dim, mid_dim=self.mid_dim, out_dim=self.out_dim,
                        embed_dim=self.embed_dim, num_heads=self.num_heads, num_layers=self.num_layers).to(device)
        if if_valid:
            if not os.path.isfile(net_path):
                raise FileNotFoundError(f"| ConfigPolicy.load_net()  net_path {net_path}")
            net.load_state_dict(th.load(net_path, map_location=device))
        else:  # if_train
            pass
        net = th.compile(net) if th.__version__ < '2.0' else net
        return net
