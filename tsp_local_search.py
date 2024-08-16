import torch as th
from tsp_simulator import SimulatorTSP

TEN = th.Tensor

class SolverLocalSearch:
    def __init__(self, simulator: SimulatorTSP, num_nodes: int):
        self.simulator = simulator
        self.num_nodes = num_nodes

        self.num_sims = 0
        self.good_xs = th.tensor([])  # solution x
        self.good_vs = th.tensor([])  # objective value

    def reset(self, xs: TEN):
        vs = self.simulator.obj_for_loop(xs=xs)

        self.good_xs = xs
        self.good_vs = vs
        self.num_sims = xs.shape[0]
        return vs

    def random_search(self, num_iters: int = 8, num_spin: int = 8, noise_std: float = 0.3):
        sim = self.simulator
        kth = self.num_nodes - num_spin

        prev_xs = self.good_xs.clone()
        prev_vs = sim.obj(prev_xs)

        for _ in range(num_iters):
            '''Flip bits randomly in the binary solution matrix'''
            flip_mask = th.randint(0, 2, prev_xs.shape, dtype=th.bool, device=self.simulator.device)
            xs = prev_xs.clone()
            xs[flip_mask] = th.logical_not(xs[flip_mask])

            # Ensure that xs remains a valid permutation matrix
            xs = self._correct_solution(xs)

            vs = sim.obj(xs)
            update_xs_by_vs(prev_xs, prev_vs, xs, vs)

        '''Addition of deterministic adjustments'''
        for i in range(sim.num_nodes):
            xs1 = prev_xs.clone()

            # Perform pairwise swaps in the permutation matrix
            for j in range(self.num_nodes):
                for k in range(j + 1, self.num_nodes):
                    xs1[:, j, :], xs1[:, k, :] = xs1[:, k, :].clone(), xs1[:, j, :].clone()
                    vs1 = sim.obj(xs1)
                    update_xs_by_vs(prev_xs, prev_vs, xs1, vs1)
                    xs1[:, j, :], xs1[:, k, :] = xs1[:, k, :].clone(), xs1[:, j, :].clone()  # Swap back to original

        num_update = update_xs_by_vs(self.good_xs, self.good_vs, prev_xs, prev_vs)
        return self.good_xs, self.good_vs, num_update

    def _correct_solution(self, xs):
        """
        Ensure the binary solution matrix remains a valid TSP solution.
        Each row and each column should have exactly one `1`.
        """
        num_sims, num_nodes_sq = xs.size()
        num_nodes = self.num_nodes
        xs = xs.view(num_sims, num_nodes, num_nodes)

        # Correct rows
        for i in range(num_nodes):
            row_sums = xs[:, i, :].sum(dim=1)
            while (row_sums != 1).any():
                for sim in range(num_sims):
                    if row_sums[sim] != 1:
                        one_indices = xs[sim, i, :].nonzero(as_tuple=True)[0]
                        if row_sums[sim] > 1:
                            xs[sim, i, one_indices[1:]] = 0
                        elif row_sums[sim] < 1:
                            available_columns = (xs[sim, :, one_indices[0]] == 0).nonzero(as_tuple=True)[0]
                            xs[sim, i, available_columns[0]] = 1
                row_sums = xs[:, i, :].sum(dim=1)

        # Correct columns
        for j in range(num_nodes):
            col_sums = xs[:, :, j].sum(dim=1)
            while (col_sums != 1).any():
                for sim in range(num_sims):
                    if col_sums[sim] != 1:
                        one_indices = xs[sim, :, j].nonzero(as_tuple=True)[0]
                        if col_sums[sim] > 1:
                            xs[sim, one_indices[1:], j] = 0
                        elif col_sums[sim] < 1:
                            available_rows = (xs[sim, one_indices[0], :] == 0).nonzero(as_tuple=True)[0]
                            xs[sim, available_rows[0], j] = 1
                col_sums = xs[:, :, j].sum(dim=1)

        return xs.view(num_sims, -1)  # Flatten back to original shape


def update_xs_by_vs(xs0, vs0, xs1, vs1):
    """
    Update the solutions in xs0 and vs0 with those in xs1 and vs1 if they are better.
    This function is tailored for a minimization problem (TSP).
    """
    # Since TSP is a minimization problem, we update when vs1 <= vs0
    good_is = vs1.le(vs0)
    xs0[good_is] = xs1[good_is]
    vs0[good_is] = vs1[good_is]
    return good_is.sum().item()  # Return the number of updated solutions


def show_gpu_memory(device):
    if not th.cuda.is_available():
        return 'not th.cuda.is_available()'

    all_memory = th.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # GB
    max_memory = th.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
    now_memory = th.cuda.memory_allocated(device) / (1024 ** 3)  # GB

    show_str = (
        f"AllRAM {all_memory:.2f} GB, "
        f"MaxRAM {max_memory:.2f} GB, "
        f"NowRAM {now_memory:.2f} GB, "
        f"Rate {(max_memory / all_memory) * 100:.2f}%"
    )
    return show_str
