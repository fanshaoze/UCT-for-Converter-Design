import _thread
import threading
import multiprocessing
import os
import time
import sys
import math
import random
from ucts import uct
from ucts import TopoPlanner
from SimulatorAnalysis import gen_topo
from simulator3component.build_topology import nets_to_ngspice_files
from simulator3component.simulation import simulate_topologies
from simulator3component.simulation_analysis import analysis_topologies
from Viz.uctViz import delAllFiles, TreeGraph
import numpy as np
import datetime

from utils.util import mkdir, get_sim_configs, save_reward_hash, get_steps_traj, read_approve_path, \
    read_joint_component_prob
from SimulatorAnalysis import UCT_data_collection

import copy
from SimulatorAnalysis.UCT_data_collection import key_expression_dict


def merge_act_nodes(dest_act_node, act_node):
    dest_act_node.avg_return_ = dest_act_node.avg_return_ * dest_act_node.num_visits_ + \
                                act_node.avg_return_ * act_node.num_visits_
    dest_act_node.num_visits_ += act_node.num_visits_
    dest_act_node.avg_return_ = dest_act_node.avg_return_ / dest_act_node.num_visits_


def get_action_from_trees(uct_tree_list, uct_tree, tree_num=4):
    contain_action_flag = 0
    uct_tree.root_.act_vect_ = []
    for i in range(tree_num):
        for j in range(len(uct_tree_list[i].node_vect_)):
            contain_action_flag = 0
            for x in range(len(uct_tree.root_.act_vect_)):
                if uct_tree.root_.act_vect_[x].equal(uct_tree_list[i].act_vect_[j]):
                    contain_action_flag = 1
                    break
            if contain_action_flag == 1:
                if j < len(uct_tree_list[i].node_vect_) and uct_tree_list[i].node_vect_[j] is not None:
                    merge_act_nodes(uct_tree.root_.node_vect_[x], uct_tree_list[i].node_vect_[j])
            else:
                if j < len(uct_tree_list[i].node_vect_) and uct_tree_list[i].node_vect_[j] is not None:
                    uct_tree.root_.act_vect_.append(uct_tree_list[i].act_vect_[j].duplicate())
                    uct_tree.root_.node_vect_.append(uct_tree_list[i].node_vect_[j])
    act_node = uct_tree.get_action()
    return act_node


def get_action_from_planners(uct_planner_list, uct_tree, tree_num=4):
    contain_action_flag = 0
    uct_tree.root_.act_vect_ = []
    uct_tree.root_.node_vect_ = []
    for i in range(tree_num):
        for j in range(len(uct_planner_list[i].root_.node_vect_)):
            contain_action_flag = 0
            for x in range(len(uct_tree.root_.act_vect_)):
                if uct_tree.root_.act_vect_[x].equal(uct_planner_list[i].root_.act_vect_[j]):
                    contain_action_flag = 1
                    break
            if contain_action_flag == 1:
                if j < len(uct_planner_list[i].root_.node_vect_) and uct_planner_list[i].root_.node_vect_[
                    j] is not None:
                    merge_act_nodes(uct_tree.root_.node_vect_[x], uct_planner_list[i].root_.node_vect_[j])
            else:
                if j < len(uct_planner_list[i].root_.node_vect_) and uct_planner_list[i].root_.node_vect_[
                    j] is not None:
                    uct_tree.root_.act_vect_.append(uct_planner_list[i].root_.act_vect_[j].duplicate())
                    uct_tree.root_.node_vect_.append(uct_planner_list[i].root_.node_vect_[j])
    act_node = uct_tree.get_action()
    # act_node = uct_tree.get_action_rave()
    return act_node


def get_action_from_trees_vote(uct_planner_list, uct_tree, tree_num=4):
    action_nodes = []
    counts = {}
    for i in range(tree_num):
        action_nodes.append(uct_planner_list[i].get_action())
    for i in range(len(action_nodes)):
        tmp_count = 0
        if counts.get(action_nodes[i]) is None:
            for j in range(len(action_nodes)):
                if action_nodes[j].equal(action_nodes[i]):
                    tmp_count += 1
            counts[action_nodes[i]] = tmp_count
    for action, tmp_count in counts.items():
        if tmp_count == max(counts.values()):
            selected_action = action
    return selected_action


def Planner_test(depth_list, trajectory, configs, date_str):
    # key_expression = key_expression_dict()
    result_set = []
    out_file_name = "Results/mutitest" + "-" + date_str + "-" + str(os.getpid()) + ".txt"
    figure_folder = "figures/" + date_str + "/"
    mkdir(figure_folder)
    want_tree_viz = True
    sim_configs = get_sim_configs(configs)

    fo = open(out_file_name, "w")
    fo.write("max_depth,num_runs,avg_step\n")
    init_nums = 1
    results = []
    avg_reward = 0
    for max_depth in depth_list:
        for num_runs in trajectory:
            print("max depth is", max_depth, ",trajectory is", num_runs, "every thread has ",
                  int(num_runs / configs["tree_num"]), " trajectories")
            avg_steps = 0

            for j in range(0, init_nums):
                print()
                avg_cumulate_reward = 0
                cumulate_reward_list = []
                fo.write("----------------------------------------------------------------------" + "\n")
                uct_simulators = []
                approved_path_freq = read_approve_path(0.0, '5comp_buck_path_freqs.json')
                # approved_path_freq = read_approve_path(0.0)

                print(approved_path_freq)
                component_condition_prob = read_joint_component_prob(configs['num_component'] - 3)
                key_expression = UCT_data_collection.read_analytics_result()

                for i in range(0, int(configs["game_num"] / init_nums)):
                    steps = 0
                    cumulate_plan_time = 0
                    r = 0
                    tree_size = 0

                    sim = TopoPlanner.TopoGenSimulator(sim_configs, approved_path_freq, component_condition_prob,
                                                       key_expression, configs['num_component'])

                    # For fixed commponent type
                    init_nodes = []
                    # init_nodes = [0, 1, 3, 2, 0]
                    # init_nodes = [0, 0, 1, 2, 3]
                    # init_nodes = [0, 0, 3, 3, 1]
                    # init_nodes = [1, 1, 2, 3, 3]

                    # init_nodes = [1,3,0]
                    for e in init_nodes:
                        action = TopoPlanner.TopoGenAction('node', e)
                        sim.act(action)
                    edges = []
                    # edges = [[0, 6], [1, 3], [2, 12], [4, 5], [6, 7], [8, 9], [10, 11]]
                    # edges = [[0, 8], [1, 3], [2, 12], [4, 11], [4, 5], [6, 7], [8, 9], [10, 11]]
                    # edges = [[0, 8], [1, 3], [-1, -1], [4, 11], [4, 5], [6, 7], [8, 9], [10, 11]]


                    # adj = sim.get_adjust_parameter_set()
                    # print(sim.get_adjust_parameter_set())
                    # {2: {5}, 5: {2}, 1: {8}, 8: {1}, 0: {3}, 3: {0}, 4: {7}, 7: {4, 6}, 6: {7}})
                    # edges = [[0, 3], [1, 8], [2, 5], [-1, -1], [4, 6]]
                    # edges = [[0, 3], [1, 8], [2, 5], [-1, -1]]
                    # edges = []
                    # edges = [[0, 8], [1, 3], [2, 6], [4, 6], [5, 7]]
                    # edges = [[0, 4], [1, 7], [2, 6], [8, 6], [5, 3]]
                    # edges = [[0, 8], [1, 3], [2, 5], [4, 6], [6, 7]]
                    # edges = []
                    # #
                    for edge in edges:
                        action = TopoPlanner.TopoGenAction('edge', edge)
                        r = sim.act(action)
                    act_str = "test"
                    # sim.current.visualize(act_str, figure_folder)
                    # tmp_state = sim.get_state()
                    # print(tmp_state.check_have_no_GND_path())
                    # # ['VIN - Sa - Sa - Sb - C - L - VOUT', 'VIN - Sa - Sa - GND', 'VOUT - L - C - Sb - GND']
                    # return

                    # print(r)

                    # return
                    # topologies = [sim.get_state()]
                    # nets_to_ngspice_files(topologies, configs, configs['num_component'])
                    # simulate_topologies(len(topologies), configs['num_component'], configs["sys_os"])
                    # effis = analysis_topologies(configs, len(topologies), configs['num_component'])
                    # print("effis of topo:", effis)
                    # return
                    mc_return = 0
                    reward_list = []
                    discount = 1
                    final_return = None
                    # final_result = sim.random_policy(mc_return, configs["gamma"], discount, final_return, reward_list)

                    final_result = sim.default_policy(mc_return, configs["gamma"], discount, final_return, reward_list,
                                                      True, True)
                    print(sim.current.check_have_no_GND_path())
                    print(sim.current.graph)
                    print('game number: ', i)
                    result_set.append(final_result)
                    if final_result > 0:
                        print("final_result ", final_result)

    pos_count = 0
    large_count = 0
    for i in range(len(result_set)):
        if result_set[i] > 0:
            pos_count += 1
            if result_set[i] > 0.5:
                print(result_set[i])
                large_count += 1
    print("positive count: ", pos_count)
    print("large reward count: ", large_count)
    print("avg of result", sum(result_set) / 1000)
    return final_result
