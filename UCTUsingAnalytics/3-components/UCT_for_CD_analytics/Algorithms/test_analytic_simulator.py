from SimulatorAnalysis.UCT_data_collection import *
from ucts import uct
from ucts import TopoPlanner

from utils.util import init_position, generate_depth_list, del_all_files, mkdir, get_args, get_sim_configs, \
    read_reward_hash, save_reward_hash, get_steps_traj, read_reward_hash_list


def anay_read_test(configs):
    sim_configs = get_sim_configs(configs)
    sim = TopoPlanner.TopoGenSimulator(sim_configs, configs['num_component'])
    sim.key_expression = read_analytics_result()
    init_nodes = []
    init_nodes = [0, 3, 1]
    for e in init_nodes:
        action = TopoPlanner.TopoGenAction('node', e)
        sim.act(action)
    edges = []
    edges = [[0, 3], [1, 8], [2, 5], [4, 7], [6, 7]]
    edges = [[0, 6], [1, 4], [2, 8], [3, 7], [5, 7]]

    for edge in edges:
        action = TopoPlanner.TopoGenAction('edge', edge)
        sim.act(action, False)

    list_of_node, list_of_edge, netlist, joint_list = convert_to_netlist(sim.current.graph,
                                                                         sim.current.component_pool,
                                                                         sim.current.port_pool,
                                                                         sim.current.parent,
                                                                         sim.current.comp2port_mapping)
    key = key_circuit_from_lists(list_of_edge, list_of_node, netlist)

    # result = find_one_analytics_result(key, sim.key_expression, sim.current.graph, sim.current.comp2port_mapping,
    #                                    sim.current.port2comp_mapping, sim.current.idx_2_port, sim.current.port_2_idx,
    #                                    sim.current.parent,
    #                                    sim.current.component_pool, sim.current.same_device_mapping,
    #                                    sim.current.port_pool)
    result = get_one_analytics_result(sim.key_expression, sim.current.graph, sim.current.comp2port_mapping,
                                      sim.current.port2comp_mapping, sim.current.idx_2_port, sim.current.port_2_idx,
                                      sim.current.parent, sim.current.component_pool, sim.current.same_device_mapping,
                                      sim.current.port_pool)

    print(result)
