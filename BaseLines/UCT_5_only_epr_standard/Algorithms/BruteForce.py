import _thread
import gc
import json
import threading
import multiprocessing
import os
import time
import sys
import math
import random
from ucts import uct
from ucts import TopoPlanner
from simulator3component.build_topology import nets_to_ngspice_files
from simulator3component.simulation import simulate_topologies
from simulator3component.simulation_analysis import analysis_topologies
from Viz.uctViz import delAllFiles, TreeGraph
import numpy as np
from SimulatorAnalysis.UCT_data_collection import simulate_one_analytics_result
import datetime

from utils.util import mkdir, get_sim_configs, save_reward_hash, get_steps_traj, read_approve_path, \
    read_joint_component_prob
from SimulatorAnalysis import UCT_data_collection

import copy
from SimulatorAnalysis.UCT_data_collection import key_expression_dict, simulate_one_analytics_result, get_analytics_file


def brute_force(depth_list, trajectories, configs, date_str):
    # key_expression = key_expression_dict()
    result_set = []
    out_file_name = "Results/mutitest" + "-" + date_str + "-" + str(os.getpid()) + ".txt"
    figure_folder = "figures/" + date_str + "/"

    simulation_folder = "simu_analy/" + date_str + '-' + str(os.getpid()) + "/" + str(trajectories[0])
    mkdir(figure_folder)
    mkdir(simulation_folder)
    want_tree_viz = True
    sim_configs = get_sim_configs(configs)

    fo = open(out_file_name, "w")
    fo.write("max_depth,num_runs,avg_step\n")
    init_nums = 1
    anal_results = [['efficiency', 'vout', 'reward', 'DC_para', 'query']]
    simu_results = [['efficiency', 'vout', 'reward', 'DC_para', 'query']]
    avg_reward = 0

    approved_path_freq = read_approve_path(0.0, '5comp_buck_path_freqs.json')
    # approved_path_freq = read_approve_path(0.0)

    print(approved_path_freq)
    component_condition_prob = read_joint_component_prob(configs['num_component'] - 3)
    key_expression = UCT_data_collection.read_analytics_result()
    key_sim_effi = UCT_data_collection.read_sim_result()

    t = 0
    for Traj in trajectories:
        result_sim = TopoPlanner.TopoGenSimulator(sim_configs, approved_path_freq, component_condition_prob,
                                                  key_expression, key_sim_effi, configs['fix_paras'],
                                                  configs['num_component'])
        max_result = -1
        sim = TopoPlanner.TopoGenSimulator(sim_configs, approved_path_freq, component_condition_prob,
                                           key_expression, key_sim_effi, configs['fix_paras'], configs['num_component'])
        init_state = TopoPlanner.TopoGenState(init=True)
        for _ in range(Traj):
            avg_steps = 0
            print()
            avg_cumulate_reward = 0
            fo.write("----------------------------------------------------------------------" + "\n")
            uct_simulators = []
            tree_size = 0
            sim.set_state(init_state)
            # For fixed commponent type
            init_nodes = []
            edges = []
            # init_nodes = [random.choice([0, 1])]
            # init_nodes = [0, 1, 3, 2, 0]
            #
            # init_nodes = [0,1,3,0,3]
            # edges = [[0,10],[1,6],[2,7],[3,10],[4,12],[5,9],[8,9],[6,11]]
            # init_nodes = [0,3,1,3,1]
            # edges = [[0,7],[1,4],[2,6],[3,10],[5,9],[8,10],[9,11],[10,12]]
            # init_nodes = [0, 3, 1, 1, 0]
            # edges = [[0, 8], [1, 4], [2, 5], [5, 10], [3, 9], [4, 11], [6, 12], [7, 12]]

            # init_nodes = [3, 0, 0, 3, 1]
            # edges = [[0, 7], [1, 3], [2, 12], [4, 6], [4, 10], [4, 11], [5, 7], [8, 9]]

            for node in init_nodes:
                action = TopoPlanner.TopoGenAction('node', node)
                sim.act(action, configs['reward_method'])

            # edges = {2: {5}, 5: {2}, 1: {8}, 8: {1}, 0: {3}, 3: {0}, 4: {7}, 7: {4, 6}, 6: {7}})
            # edges = [[0, 3], [1, 8], [2, 5], [-1, -1], [4, 6]]
            for edge in edges:
                action = TopoPlanner.TopoGenAction('edge', edge)
                r = sim.act(action, configs['reward_method'])
            # UCT_data_collection.save_sim_result(sim.key_sim_effi_)
            # return

            mc_return = 0
            reward_list = []
            discount = 1
            final_return = None
            final_result = sim.default_policy(mc_return, configs["gamma"], discount, final_return, reward_list,
                                              configs["component_default_policy"], configs["path_default_policy"],
                                              configs["reward_method"])
            if final_result >= max_result:
                print(final_result, max_result)
                result_sim.set_state(copy.deepcopy(sim.get_state()))
                max_result = final_result
        max_sim_reward_result = sim.get_tops_sim_info()
        for i in range(len(sim.topk)):
            current = sim.topk[i][0]
            if current.graph_is_valid():
                print(sim.topk[i][2])
                results_tmp = get_analytics_file(key_expression, current.graph,
                                                 current.comp2port_mapping,
                                                 current.port2comp_mapping, current.idx_2_port,
                                                 current.port_2_idx,
                                                 current.parent, current.component_pool,
                                                 current.same_device_mapping, sim.current.port_pool,
                                                 None, sim.configs_['min_vout'])
            else:
                results_tmp = None
            date_now_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", results_tmp)
            name = str(t) + '-' + date_now_str
            with open(simulation_folder + '/' + name + '-analytic.json', 'w') as f:
                json.dump(results_tmp, f)
            f.close()
            t += 1

        sim.set_state(result_sim.get_state())
        final_para_str = sim.current.parameters
        sim.get_state().visualize(
            "result with parameter:" + str(str(final_para_str)) + " ", figure_folder)

        effis = [sim.get_effi_info()]
        fo.write("Final topology of game " + ":\n")
        fo.write("component_pool:" + str(sim.current.component_pool) + "\n")
        fo.write(str(sim.current.parameters) + "\n")
        fo.write("port_pool:" + str(sim.current.port_pool) + "\n")
        fo.write("graph:" + str(sim.current.graph) + "\n")
        fo.write("efficiency:" + str(effis) + "\n")
        fo.write("final reward:" + str(max_result) + "\n")
        fo.write("step:" + str(avg_steps) + "\n")
        total_query = sim.query_counter
        total_hash_query = sim.hash_counter
        for simulator in uct_simulators:
            total_query += simulator.query_counter
            total_hash_query += simulator.hash_counter
        fo.write("query time:" + str(total_query) + " total tree size:" + str(tree_size) + "\n")
        fo.write("hash query time:" + str(total_hash_query) + "\n")
        end_time = datetime.datetime.now()
        final_para_str = sim.current.parameters
        sim.get_state().visualize(
            "result with parameter:" + str(str(final_para_str)) + " ", figure_folder)
        fo.write("end at:" + str(end_time) + "\n")
        fo.write("result with parameter:" + str(str(final_para_str)) + "\n")
        fo.write("----------------------------------------------------------------------" + "\n")
        fo.write("configs:" + str(configs) + "\n")
        fo.write("final rewards:" + str(avg_cumulate_reward) + "\n")

        print(effis)
        anal_result = [effis[0]['efficiency'], effis[0]['output_voltage'],
                       max_result, final_para_str, total_query]
        anal_results.append(anal_result)
        simu_result = [max_sim_reward_result['max_sim_effi'], max_sim_reward_result['max_sim_vout'],
                       max_sim_reward_result['max_sim_reward'], max_sim_reward_result['max_sim_para'],
                       total_query]
        simu_results.append(simu_result)
        # UCT_data_collection.save_analytics_result(sim.key_expression)
        UCT_data_collection.save_sim_result(sim.key_sim_effi_)

        del sim
        del result_sim
        gc.collect()

    print("figures are saved in:" + str(figure_folder) + "\n")
    print("outputs are saved in:" + out_file_name + "\n")

    for result in anal_results:
        fo.write(str(result) + "\n")
    fo.close()
    return anal_results, simu_results
