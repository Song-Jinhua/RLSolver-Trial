

import torch as th
from config import GraphList
from tsp_simulator import SimulatorTSP
from tsp_local_search import SolverLocalSearch

TEN = th.Tensor


def metropolis_hastings_sampling(probs: TEN, start_xs: TEN, num_repeats: int, num_iters: int = -1) -> TEN:
    """Metropolis-Hastings sampling adapted for TSP."""
    xs = start_xs.repeat(num_repeats, 1)
    ps = probs.repeat(num_repeats, 1)

    num, dim = xs.shape
    device = xs.device
    num_iters = int(dim // 4) if num_iters == -1 else num_iters

    count = 0
    for _ in range(4):
        ids = th.randperm(dim, device=device)
        for i in range(dim):
            idx = ids[i]
            chosen_p0 = ps[:, idx]
            chosen_xs = xs[:, idx]
            chosen_ps = th.where(chosen_xs, chosen_p0, 1 - chosen_p0)

            accept_rates = (1 - chosen_ps) / chosen_ps
            accept_masks = th.rand(num, device=device).lt(accept_rates)
            xs[:, idx] = th.where(accept_masks, th.logical_not(chosen_xs), chosen_xs)

            count += accept_masks.sum()
            if count >= num * num_iters:
                break
        if count >= num * num_iters:
            break
    return xs


class MCMC_TSP:
    def __init__(self, num_nodes: int, num_sims: int, num_repeats: int, num_searches: int,
                 graph_file: str, device=th.device('cpu')):
        self.num_nodes = num_nodes
        self.num_sims = num_sims
        self.num_repeats = num_repeats
        self.num_searches = num_searches
        self.device = device
        self.sim_ids = th.arange(num_sims, device=device)

        # Initialize the TSP simulator with the graph file
        self.simulator = SimulatorTSP(graph_file=graph_file, device=self.device)
        self.searcher = SolverLocalSearch(simulator=self.simulator, num_nodes=self.num_nodes)

    # 如果end to end, graph_list为空元组。如果distribution, 抽样赋值
    def reset(self, graph_file: str):
        self.simulator = SimulatorTSP(graph_file=graph_file, device=self.device)
        self.searcher = SolverLocalSearch(simulator=self.simulator, num_nodes=self.num_nodes)
        self.searcher.reset(xs=self.simulator.generate_xs_randomly(num_sims=self.num_sims))

        good_xs = self.searcher.good_xs
        good_vs = self.searcher.good_vs
        return good_xs, good_vs

    # probs: 策略网络输出值
    # start_xs： 上一轮step的输出的解中，并行环境输出的最好的解
    def step(self, start_xs: TEN, probs: TEN) -> (TEN, TEN):
        xs = metropolis_hastings_sampling(probs=probs, start_xs=start_xs, num_repeats=self.num_repeats, num_iters=-1)
        vs = self.searcher.reset(xs)
        for _ in range(self.num_searches):
            xs, vs, num_update = self.searcher.random_search(num_iters=8)
        return xs, vs

    # 好的解的数量是sim数量，一个环境出一个好的解
    def pick_good_xs(self, full_xs, full_vs) -> (TEN, TEN):
        # Update good_xs: use .view() instead of .reshape() for saving GPU memory
        xs_view = full_xs.view(self.num_repeats, self.num_sims, self.num_nodes * self.num_nodes)
        vs_view = full_vs.view(self.num_repeats, self.num_sims)
        ids = vs_view.argmin(dim=0)  # For TSP, we are minimizing the tour length

        good_xs = xs_view[ids, self.sim_ids]
        good_vs = vs_view[ids, self.sim_ids]
        return good_xs, good_vs

