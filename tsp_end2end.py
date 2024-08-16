import os
import sys
import torch as th
from torch.nn.utils import clip_grad_norm_

from config import ConfigPolicy, GraphList
from network import reset_parameters_of_model
from evaluator import Evaluator, read_info_from_recorder
from tsp_local_search import show_gpu_memory
from graph_utils import load_tsp_graph_from_file, obtain_num_nodes

TEN = th.Tensor

from env_l2a_tsp import MCMC_TSP

'''run'''


def valid_in_single_graph(
        args0: ConfigPolicy = None,
        graph_file: str = None,
        if_valid: bool = True,
):
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    '''dummy value'''
    _graph_type, _num_nodes, _graph_id = 'TSP', 0, 0  # Adjusted for TSP
    args0 = args0 if args0 else ConfigPolicy(graph_type=_graph_type, num_nodes=_num_nodes)

    '''Load TSP graph from file'''
    graph_file = graph_file if graph_file else 'path/to/tsp/graph_file.tsp'
    graph_list = load_tsp_graph_from_file(graph_file)

    '''config: train'''
    num_nodes = obtain_num_nodes(graph_list=graph_list)
    args0.num_nodes = num_nodes
    num_sims = args0.num_sims
    num_repeats = args0.num_repeats
    num_searches = args0.num_searches
    reset_gap = args0.reset_gap
    num_iters = args0.num_iters
    num_sgd_steps = args0.num_sgd_steps
    entropy_weight = args0.entropy_weight

    weight_decay = args0.weight_decay
    learning_rate = args0.learning_rate

    show_gap = args0.show_gap

    '''model'''
    from network import PolicyMLP
    policy_net = PolicyMLP(num_bits=num_nodes * num_nodes, mid_dim=256).to(device)
    policy_net = th.compile(policy_net) if th.__version__ < '2.0' else policy_net

    net_params = list(policy_net.parameters())
    optimizer = th.optim.AdamW(net_params, lr=learning_rate, maximize=False, weight_decay=weight_decay)

    '''iterator'''
    th.set_grad_enabled(False)
    mcmc = MCMC_TSP(num_nodes=num_nodes, num_sims=num_sims, num_repeats=num_repeats, num_searches=num_searches,
                    graph_file=graph_file, device=device)
    if_maximize = False  # TSP is a minimization problem

    '''evaluator'''
    save_dir = f"./TSP_{num_nodes}"
    os.makedirs(save_dir, exist_ok=True)
    good_xs, good_vs = mcmc.reset(graph_file=graph_file)
    evaluator = Evaluator(save_dir=save_dir, num_bits=num_nodes, x=good_xs[0], v=good_vs[0].item(),
                          if_maximize=if_maximize)
    evaluators = []

    '''loop'''
    lamb_entropy = (th.cos(th.arange(reset_gap, device=device) / (reset_gap - 1) * th.pi) + 1) / 2 * entropy_weight
    for i in range(num_iters):
        probs = policy_net.auto_regressive(xs_flt=good_xs[good_vs.argmin(), None, :].float())
        probs = probs.repeat(num_sims, 1)

        full_xs, full_vs = mcmc.step(start_xs=good_xs, probs=probs)
        good_xs, good_vs = mcmc.pick_good_xs(full_xs=full_xs, full_vs=full_vs)

        advantages = -full_vs.float()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        del full_vs

        th.set_grad_enabled(True)
        for j in range(num_sgd_steps):
            probs = policy_net.auto_regressive(xs_flt=good_xs[good_vs.argmin(), None, :].float())
            probs = probs.repeat(num_sims, 1)

            full_ps = probs.repeat(num_repeats, 1)
            logprobs = th.log(th.where(full_xs, full_ps, 1 - full_ps)).sum(dim=1)

            _probs = 1 - probs
            entropy = (probs * probs.log2() + _probs * _probs.log2()).mean(dim=1)
            obj_entropy = entropy.mean()
            obj_values = (th.softmax(logprobs, dim=0) * advantages).sum()

            objective = obj_values + obj_entropy * lamb_entropy[i % reset_gap]
            optimizer.zero_grad()
            objective.backward()
            clip_grad_norm_(net_params, 3)
            optimizer.step()
        th.set_grad_enabled(False)

        '''update good_xs'''
        good_i = good_vs.argmin()
        good_x = good_xs[good_i]
        good_v = good_vs[good_i]
        if_show_x = evaluator.record2(i=i, vs=good_v.item(), xs=good_x)

        if (i + 1) % show_gap == 0 or if_show_x:
            _probs = 1 - probs
            entropy = (probs * probs.log2() + _probs * _probs.log2()).mean(dim=1)
            obj_entropy = -entropy.mean().item()

            show_str = f"| entropy {obj_entropy:9.4f} tsp_value {good_vs.float().mean().item():6.2f} < {good_vs.min():6.2f}"
            evaluator.logging_print(show_str=show_str, if_show_x=False)
            sys.stdout.flush()

        if (i + 1) % reset_gap == 0:
            print(f"| reset {show_gpu_memory(device=device)} "
                  f"| improvement_rate {1. - evaluator.best_v / evaluator.first_v:8.5f}")
            sys.stdout.flush()

            '''method1: reset (keep old graph)'''
            # reset_parameters_of_model(model=policy_net)
            # good_xs, good_vs = iterator.reset(graph_file=graph_file)

            '''method2: reload (load new graph)'''
            if if_valid:
                reset_parameters_of_model(model=policy_net)
                net_params = list(policy_net.parameters())
                optimizer = th.optim.AdamW(net_params, lr=learning_rate, maximize=False, weight_decay=weight_decay)

            good_xs, good_vs = mcmc.reset(graph_file=graph_file)

            evaluators.append(evaluator)
            evaluator = Evaluator(save_dir=save_dir, num_bits=num_nodes, x=good_xs[0], v=good_vs[0].item(),
                                  if_maximize=False)

    if if_valid:
        evaluator.save_record_draw_plot(fig_dpi=300)
    else:
        th.save(policy_net.state_dict(), args0.net_path)
    return evaluators


def tsp_end2end_mlp():
    graph_file = "att48.tsp"  # Take att48.tsp as example

    '''input'''
    graph_list: GraphList = load_tsp_graph_from_file(graph_file)
    num_nodes = obtain_num_nodes(graph_list)

    """MLP"""
    args0 = ConfigPolicy(graph_type='TSP', num_nodes=num_nodes)
    # args0.net_path = f"{model_dir}/policy_net_TSP_Node{len(graph_list.nodes)}.pth"

    evaluator = valid_in_single_graph(args0=args0, graph_file=graph_file, if_valid=True)[0]

    '''MLP output'''
    x = evaluator.best_x
    x_str = evaluator.best_x_str()
    valid_str = read_info_from_recorder(evaluator.recorder2, per_second=10)

    info_str = f"TSP {x_str} {valid_str}"
    print(f"|INFO: {info_str}")
    print(f"|Solution X {x}")


if __name__ == '__main__':
    use_mlp = True
    if use_mlp:
        tsp_end2end_mlp()  # an end-to-end tutorial for TSP
