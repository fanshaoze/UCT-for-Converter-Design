import _thread
import copy
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
import datetime
from utils.util import mkdir, get_sim_configs, save_reward_hash, get_steps_traj, read_approve_path, \
    read_joint_component_prob
from utils.eliminate_isomorphism import unblc_comp_set_mapping, get_component_priorities
from SimulatorAnalysis import UCT_data_collection
from SimulatorAnalysis.simulate_with_topology import *
# from GNN_gendata.GenerateTrainData import update_dataset
import gc

from multiprocessing import Process, Manager, Pipe, Value


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
                if j < len(uct_planner_list[i].root_.node_vect_) and \
                        uct_planner_list[i].root_.node_vect_[j] is not None:
                    merge_act_nodes(uct_tree.root_.node_vect_[x], uct_planner_list[i].root_.node_vect_[j])
            else:
                if j < len(uct_planner_list[i].root_.node_vect_) and \
                        uct_planner_list[i].root_.node_vect_[j] is not None:
                    uct_tree.root_.act_vect_.append(uct_planner_list[i].root_.act_vect_[j].duplicate())
                    uct_tree.root_.node_vect_.append(uct_planner_list[i].root_.node_vect_[j])
    act_node = uct_tree.get_action()
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


def read_DP_files(configs):
    target_min_vout = -500
    target_max_vout = 500
    if target_min_vout < configs['target_vout'] < 0:
        approved_path_freq = read_approve_path(0.0,
                                               './3comp_buck_boost_sim_path_freqs.json')
        component_condition_prob = read_joint_component_prob(configs['num_component'] - 3,
                                                             None,
                                                             "./3comp_buck_boost_sim_node_joint_probs.json")

        print(approved_path_freq)
        print(component_condition_prob)

    elif 0 < configs['target_vout'] < 100:
        approved_path_freq = read_approve_path(0.0, './3comp_buck_sim_path_freqs.json')
        component_condition_prob = read_joint_component_prob(configs['num_component'] - 3,
                                                             None,
                                                             "./3comp_buck_sim_node_joint_probs.json")

        print(approved_path_freq)
        print(component_condition_prob)


    elif 100 < configs['target_vout'] < target_max_vout:
        approved_path_freq = read_approve_path(0.0, './3comp_boost_sim_path_freqs.json')
        component_condition_prob = read_joint_component_prob(configs['num_component'] - 3,
                                                             None,
                                                             "./3comp_boost_sim_node_joint_probs.json")
        print(approved_path_freq)
        print(component_condition_prob)
    else:
        return None
    return approved_path_freq, component_condition_prob


def write_info_to_file(fo, sim, effis, avg_cumulate_reward, avg_steps, total_query, total_hash_query, start_time,
                       tree_size, configs):
    fo.write("Final topology of game " + ":\n")
    fo.write("component_pool:" + str(sim.current.component_pool) + "\n")
    fo.write(str(sim.current.parameters) + "\n")
    fo.write("port_pool:" + str(sim.current.port_pool) + "\n")
    fo.write("graph:" + str(sim.current.graph) + "\n")
    fo.write("efficiency:" + str(effis) + "\n")
    fo.write("final reward:" + str(avg_cumulate_reward) + "\n")
    fo.write("step:" + str(avg_steps) + "\n")
    fo.write("query time:" + str(total_query) + " total tree size:" + str(tree_size) + "\n")
    fo.write("hash query time:" + str(total_hash_query) + "\n")
    end_time = datetime.datetime.now()
    fo.write("end at:" + str(end_time) + "\n")
    fo.write("start at:" + str(start_time) + "\n")
    fo.write("execute time:" + str((end_time - start_time).seconds) + " seconds\n")
    fo.write("result with parameter:" + str(sim.current.parameters) + "\n")
    fo.write("----------------------------------------------------------------------" + "\n")
    fo.write("configs:" + str(configs) + "\n")
    fo.write("final rewards:" + str(avg_cumulate_reward) + "\n")
    return fo


def print_and_write_child_info(fo, child_idx, _root, child, child_node, child_state):
    print("action ", child_idx, " :", _root.act_vect_[child_idx].type, "on",
          _root.act_vect_[child_idx].value)
    print("action child ", child_idx, " avg_return:", child.avg_return_)
    print("state child ", child_idx, " reward:", child_node.reward_)
    print("state ", child_idx, "ports:", child_state.port_pool)
    print("state child", child_idx, "graph:", child_state.graph)

    fo.write("action " + str(child_idx) + " :" + str(_root.act_vect_[child_idx].type) + "on" +
             str(_root.act_vect_[child_idx].value) + "\n")
    fo.write("action child " + str(child_idx) + " avg_return:" + str(child.avg_return_) + "\n")
    fo.write("action child " + str(child_idx) + " num_visits_:" + str(child.num_visits_) + "\n")
    fo.write(
        "state child " + str(child_idx) + "child node reward:" + str(child_node.reward_) + "\n")
    fo.write("state child " + str(child_idx) + "ports:" + str(child_state.port_pool) + "\n")
    fo.write("state child " + str(child_idx) + "graph:" + str(child_state.graph) + "\n")
    return fo


def parallel_get_multi_k_sim_results(sim, configs, total_query, start_time, total_planning_time,
                                     anal_results, simu_results, save_tops, filted_save_tops, filter_sim):
    effis = []
    max_sim_reward_results = {}

    for k in configs['topk_list']:
        max_sim_reward_results[k] = {'max_sim_state': 'empty', 'max_sim_reward': 0, 'max_sim_para': -1,
                                     'max_sim_effi': -1, 'max_sim_vout': -500}

    sim.current, sim.reward, sim.current.parameters = sim.get_max_seen()
    if sim.current is None:
        return max_sim_reward_results
    max_result = sim.reward

    max_topk = copy.deepcopy(sim.topk)
    sim.topk = copy.deepcopy(max_topk[-max(configs['topk_list']):])
    print('lenth of topk for parallel simulation: ', len(sim.topk))
    shared_key_sim_effi_ = parallel_topk_simulation(thread_num=configs['tree_num'], topks=sim.topk,
                                                    target_vout=configs['target_vout'],
                                                    candidate_params=sim.candidate_params)
    for k, v in shared_key_sim_effi_.items():
        print(k, 'with simu info: ', v)
    sim.key_sim_effi_ = shared_key_sim_effi_

    if configs['reward_method'] == 'analytics':
        max_topk = copy.deepcopy(sim.topk)
        for k in configs['topk_list']:
            sim.topk = copy.deepcopy(max_topk[-k:])
            # effis: [reward, effi, vout, para], get the max's reward
            effis = sim.get_reward_using_anal()

            if len(sim.topk) == 0:
                max_sim_reward_result = {'max_sim_state': 'empty', 'max_sim_reward': 0, 'max_sim_para': [],
                                         'max_sim_effi': -1, 'max_sim_vout': -500}
            else:
                # Using the reward_threshold, if need gnn filter.
                # max_sim_reward_result = get_simulator_tops_sim_info(sim=sim, filter_sim=filter_sim,
                #                                                     thershold=configs['reward_threshold'])

                max_sim_reward_result = get_simulator_tops_sim_info(sim=sim, filter_sim=filter_sim,
                                                                    thershold=1)
                # [state, anal_reward, anal_para, key, sim_reward, sim_effi, sim_vout, sim_para]
                top_simus = []
                filted_top_simus = []
                for top in sim.topk:
                    top_simus.append(top[4])
                save_tops.append(top_simus)
                for top in sim.topk:
                    filted_top_simus.append(top[-1])
                filted_save_tops.append(filted_top_simus)

            max_sim_reward_results[k] = max_sim_reward_result
            # simu_results[k] = [['efficiency', 'vout', 'reward', 'DC_para', 'query', 'avg_time', 'new_query']]
            total_running_time = (datetime.datetime.now() - start_time).seconds
            if effis is not None:
                # anal_result [effi, vout, max result, para, total query]
                anal_result = [effis[1], effis[2], max_result, str(sim.current.parameters), total_query,
                               total_running_time, total_planning_time]
                anal_results[k].append(anal_result)
            simu_results[k].append(
                [max_sim_reward_result['max_sim_effi'], max_sim_reward_result['max_sim_vout'],
                 max_sim_reward_result['max_sim_reward'],
                 max_sim_reward_result['max_sim_para'],
                 total_query, total_running_time, total_planning_time])

    elif configs['reward_method'] == 'simulator':
        effis = sim.get_reward_using_sim()
        max_sim_reward_result = {}
        if len(sim.topk) == 0:
            max_sim_reward_result = {'max_sim_state': 'empty', 'max_sim_reward': 0, 'max_sim_para': -1,
                                     'max_sim_effi': -1, 'max_sim_vout': -500}
        else:
            max_sim_reward_result = get_simulator_tops_sim_info(sim=sim)
        total_running_time = (datetime.datetime.now() - start_time).seconds
        for k in configs['topk_list']:
            if effis is not None:
                # anal_result [effi, vout, max result, para, total query]
                anal_results[k].append([effis[1], effis[2], max_result, str(sim.current.parameters),
                                        total_query, total_running_time, total_planning_time])
            simu_results[k].append(
                [max_sim_reward_result['max_sim_effi'], max_sim_reward_result['max_sim_vout'],
                 max_sim_reward_result['max_sim_reward'],
                 max_sim_reward_result['max_sim_para'],
                 total_query, total_running_time, total_planning_time])
    else:
        effis = None
    print("effis of topo:", effis)

    return anal_results, simu_results, effis, save_tops, filted_save_tops


def get_multi_k_sim_results(sim, configs, total_query, start_time, total_planning_time,
                            anal_results, simu_results, save_tops, filted_save_tops, filter_sim):
    effis = []
    max_sim_reward_results = {}
    for k in configs['topk_list']:
        max_sim_reward_results[k] = {'max_sim_state': 'empty', 'max_sim_reward': 0, 'max_sim_para': -1,
                                     'max_sim_effi': -1, 'max_sim_vout': -500}

    sim.current, sim.reward, sim.current.parameters = sim.get_max_seen()
    if sim.current is None:
        return max_sim_reward_results
    max_result = sim.reward
    if configs['reward_method'] == 'analytics':
        max_topk = copy.deepcopy(sim.topk)
        for k in configs['topk_list']:
            sim.topk = copy.deepcopy(max_topk[-k:])
            # effis: [reward, effi, vout, para], get the max's reward
            effis = sim.get_reward_using_anal()

            if len(sim.topk) == 0:
                max_sim_reward_result = {'max_sim_state': 'empty', 'max_sim_reward': 0, 'max_sim_para': [],
                                         'max_sim_effi': -1, 'max_sim_vout': -500}
            else:
                # Using the reward_threshold, if need gnn filter.
                # max_sim_reward_result = get_simulator_tops_sim_info(sim=sim, filter_sim=filter_sim,
                #                                                     thershold=configs['reward_threshold'])
                max_sim_reward_result = get_simulator_tops_sim_info(sim=sim, filter_sim=filter_sim,
                                                                    thershold=1)
                # [state, anal_reward, anal_para, key, sim_reward, sim_effi, sim_vout, sim_para]
                top_simus = []
                filted_top_simus = []
                for top in sim.topk:
                    top_simus.append(top[4])
                save_tops.append(top_simus)
                for top in sim.topk:
                    filted_top_simus.append(top[-1])
                filted_save_tops.append(filted_top_simus)

            max_sim_reward_results[k] = max_sim_reward_result
            # simu_results[k] = [['efficiency', 'vout', 'reward', 'DC_para', 'query', 'avg_time', 'new_query']]
            total_running_time = (datetime.datetime.now() - start_time).seconds
            if effis is not None:
                # anal_result [effi, vout, max result, para, total query]
                anal_result = [effis[1], effis[2], max_result, str(sim.current.parameters), total_query,
                               total_running_time, total_planning_time]
                anal_results[k].append(anal_result)
            simu_results[k].append(
                [max_sim_reward_result['max_sim_effi'], max_sim_reward_result['max_sim_vout'],
                 max_sim_reward_result['max_sim_reward'],
                 max_sim_reward_result['max_sim_para'],
                 total_query, total_running_time, total_planning_time])

    elif configs['reward_method'] == 'simulator':
        effis = sim.get_reward_using_sim()
        max_sim_reward_result = {}
        if len(sim.topk) == 0:
            max_sim_reward_result = {'max_sim_state': 'empty', 'max_sim_reward': 0, 'max_sim_para': -1,
                                     'max_sim_effi': -1, 'max_sim_vout': -500}
        else:
            max_sim_reward_result = get_simulator_tops_sim_info(sim=sim)
        total_running_time = (datetime.datetime.now() - start_time).seconds
        for k in configs['topk_list']:
            if effis is not None:
                # anal_result [effi, vout, max result, para, total query]
                anal_results[k].append([effis[1], effis[2], max_result, str(sim.current.parameters),
                                        total_query, total_running_time, total_planning_time])
            simu_results[k].append(
                [max_sim_reward_result['max_sim_effi'], max_sim_reward_result['max_sim_vout'],
                 max_sim_reward_result['max_sim_reward'],
                 max_sim_reward_result['max_sim_para'],
                 total_query, total_running_time, total_planning_time])
    else:
        effis = None
    print("effis of topo:", effis)

    return anal_results, simu_results, effis, save_tops, filted_save_tops


def trajs_all_in_first_step(total_step, num_runs):
    steps_traj = []
    for i in range(total_step):
        if i == 0:
            steps_traj.append(total_step * num_runs - 2 * (total_step - 1))
        else:
            steps_traj.append(2)
    if steps_traj[0] == 2:
        steps_traj[0] = 4
    return steps_traj


def copy_simulators_info(sim, uct_simulators):
    def keyFunc(element):
        return element[1]

    sim.no_isom_seen_state_list.clear()
    sim.topk.clear()
    sim.new_query_time, sim.new_query_counter = 0, 0
    no_isom_seen_state_dict = {}
    for i in range(len(uct_simulators)):
        for no_isom_state in uct_simulators[i].no_isom_seen_state_list:
            if no_isom_state[1] not in no_isom_seen_state_dict:
                no_isom_seen_state_dict[no_isom_state[1]] = no_isom_state[0]
    for k, v in no_isom_seen_state_dict.items():
        sim.no_isom_seen_state_list.append([v, k])
    for uct_simu in uct_simulators:
        if uct_simu.current_max['reward'] > sim.current_max['reward']:
            sim.current_max = uct_simu.current_max
        sim.new_query_time += uct_simu.new_query_time
        sim.new_query_counter += uct_simu.new_query_counter
        # self.topk.append([self.current.duplicate(), self.reward, self.current.parameters, key_info])
        for top in uct_simu.topk:
            if not sim.topk_include_current(top[3]):
                sim.topk.append(top)

        if hasattr(sim, 'surrogate_hash_table') and hasattr(uct_simu, 'surrogate_hash_table'):
            sim.surrogate_hash_table.update(uct_simu.surrogate_hash_table)
    sim.topk.sort(key=keyFunc)
    for top in sim.topk:
        print(top[1], ' ', top[3])
    return sim


def get_total_querys(uct_simulators):
    total_query, total_hash_query = 0, 0
    for simulator in uct_simulators:
        total_query += simulator.query_counter
        total_hash_query += simulator.hash_counter
    return total_query, total_hash_query


def pre_fix_topo(sim):
    # For fixed commponent type
    init_nodes = []
    init_edges = []
    # init_nodes = [0,1,2,1,2]
    # init_edges = [[0,9],[1,11],[2,3],[3,7],[4,12],[5,12],[6,8],[10,12]]
    print(init_nodes)
    # init_nodes = [1, 0, 1, 3, 0]
    # init_nodes = [0, 0, 3, 3, 1]
    # init_nodes = [0, 0, 1, 2, 3]
    for node in init_nodes:
        action = TopoPlanner.TopoGenAction('node', node)
        sim.act(action)

    # edges = [[0, 7], [1,10], [2,6], [3,9], [4, 8], [5,9], [-1,-1], [-1,-1]]
    # edges = [[0, 6], [1,3], [2,10], [3,7], [4, 11], [5,8], [6,12], [-1,-1], [8,11],[9,11]
    #     , [-1,-1],[-1,-1],[-1,-1]]

    # edges = [[0, 3], [1, 8], [2, 5], [4, 7], [6, 7]]
    # edges = [[0, 8], [1, 3], [2, 4], [4, 6], [5, 7]]
    for edge in init_edges:
        action = TopoPlanner.TopoGenAction('edge', edge)
        sim.act(action)

    # for action_tmp in sim.act_vect:
    #     print(action_tmp.value)
    total_step = sim.configs_['num_component'] - 3 - len(init_nodes) + 3 + \
                 2 * (sim.configs_['num_component'] - 3) - len(init_edges) + int(not sim.configs_['sweep'])
    return init_nodes, init_edges, sim, total_step


def init_pipes(tree_num):
    """
    create the parallel pipes
    @param tree_num:
    @return: main and sub pipes
    """
    conn_main = []
    conn_sub = []
    for i in range(tree_num):
        conn_0, conn_1 = Pipe()
        conn_main.append(conn_0)
        conn_sub.append(conn_1)
    return conn_main, conn_sub


def close_pipes(tree_num, conn_main, conn_sub):
    """

    @param tree_num:
    @param conn_main:
    @param conn_sub:
    @return:
    """
    for i in range(tree_num):
        conn_main[i].close()
        conn_sub[i].close()
    return conn_main, conn_sub


def parallel_UCF_test(trajectory, test_number, configs, result_folder, Sim=None, uct_tree_list=None,
                      keep_uct_tree=False):
    global component_condition_prob
    if Sim is None:
        Sim = TopoPlanner.TopoGenSimulator
        inside_sim = True
    else:
        inside_sim = False
    # path = './SimulatorAnalysis/database/analytic-expression.json'
    # is_exits = os.path.exists(path)
    # if not is_exits:
    #     UCT_data_collection.key_expression_dict()
    # print("finish reading key-expression")
    out_file_folder = 'Results/' + result_folder + '/'
    mkdir(out_file_folder)
    out_file_name = out_file_folder + str(trajectory) + '-result.txt'
    out_round_folder = 'Results/' + result_folder + '/' + str(trajectory)
    mkdir(out_round_folder)
    figure_folder = "figures/" + result_folder + "/"
    mkdir(figure_folder)

    # out_file_name = "Results/mutitest_" + str(configs['target_vout']) + "-" + date_str + "-" + str(os.getpid()) + ".txt"
    # figure_folder = "figures/" + result_folder + "/"
    # mkdir(figure_folder)

    sim_configs = get_sim_configs(configs)
    start_time = datetime.datetime.now()

    simu_results = {}
    anal_results = {}
    save_tops = []
    filted_save_tops = []
    for k in configs['topk_list']:
        simu_results[k] = [['efficiency', 'vout', 'reward', 'DC_para', 'query', 'avg_time', 'avg_planning_time']]
        anal_results[k] = [['efficiency', 'vout', 'reward', 'DC_para', 'query', 'avg_time', 'avg_planning_time']]

    fo = open(out_file_name, "w")
    fo.write("max_depth,num_runs,avg_step\n")
    avg_step_list = []

    for test_idx in range(test_number):
        fo.write("----------------------------------------------------------------------" + "\n")
        num_runs = trajectory
        avg_steps, avg_cumulate_reward, steps, cumulate_plan_time, r, tree_size, preset_component_num = \
            0, 0, 0, 0, 0, 0, 0

        cumulate_reward_list, uct_simulators, uct_tree_list = [], [], []

        # parallel inits
        conn_main, conn_sub = init_pipes(tree_num=configs['tree_num'])
        reward_hash, threads, total_traj = Manager().dict(), [], Value('i', num_runs)

        approved_path_freq, component_condition_prob = read_DP_files(configs)
        key_expression = UCT_data_collection.read_no_sweep_analytics_result()
        key_sim_effi = UCT_data_collection.read_no_sweep_sim_result()

        component_priorities = get_component_priorities()
        # TODO must be careful, if we delete the random adding of sa,sb,
        #  we also need to change the preset comp number
        _unblc_comp_set_mapping, _ = unblc_comp_set_mapping(['Sa', 'Sb', 'L', 'C'],
                                                            configs['num_component'] - 3 - preset_component_num)
        # for k, v in _unblc_comp_set_mapping.items():
        #     print(k, '\t', v)

        # init outer simulator and tree

        filter_sim = None
        # from topo_envs.GNNRewardSim import GNNRewardSim
        # def sim_init(*a):
        #     return GNNRewardSim(configs['eff_model'], configs['vout_model'], configs['reward_model'],
        #                         configs['debug'], *a)
        #
        # filter_sim = sim_init(sim_configs, approved_path_freq,
        #                       component_condition_prob,
        #                       key_expression, _unblc_comp_set_mapping, component_priorities,
        #                       key_sim_effi,
        #                       None, configs['num_component'], None)
        current_round_start_time = datetime.datetime.now()

        sim = Sim(sim_configs, approved_path_freq,
                  component_condition_prob,
                  key_expression, _unblc_comp_set_mapping, component_priorities,
                  key_sim_effi,
                  None, configs['num_component'], filter_sim)
        sim.graph_2_reward = reward_hash

        isom_topo_dict = {}

        uct_tree = uct.UCTPlanner(sim, -1, num_runs, configs["ucb_scalar"], configs["gamma"],
                                  configs["leaf_value"], configs["end_episode_value"],
                                  configs["deterministic"], configs["rave_scalar"], configs["rave_k"],
                                  configs['component_default_policy'], configs['path_default_policy'])

        uct_simulators.clear()
        uct_tree_list.clear()
        # init inner simulators and trees
        # replace the key_sim_effi_for_anal with the key_sim_effi to enable the simu hash table for trees
        key_sim_effi_for_anal = {}
        for n in range(configs["tree_num"]):
            uct_simulators.append(Sim(sim_configs, approved_path_freq,
                                      component_condition_prob,
                                      key_expression, _unblc_comp_set_mapping, component_priorities,
                                      key_sim_effi_for_anal,
                                      None, configs['num_component'], filter_sim))

            uct_tree_list.append(
                uct.UCTPlanner(uct_simulators[n], -1, int(num_runs / configs["tree_num"]),
                               configs["ucb_scalar"], configs["gamma"], configs["leaf_value"],
                               configs["end_episode_value"], configs["deterministic"],
                               configs["rave_scalar"], configs["rave_k"], configs['component_default_policy'],
                               configs['path_default_policy']))

        # For fixed commponent type
        init_nodes, init_edges, sim, total_step = pre_fix_topo(sim)

        for n in range(configs["tree_num"]):
            t = Process(target=uct_tree_list[n].set_and_plan, args=(total_traj, reward_hash, conn_sub[n],))
            threads.append(t)
            t.start()
        print(len(threads))

        # set roots
        uct_tree.set_root_node(sim.get_state(), sim.get_actions(), sim.get_next_candidate_components(),
                               sim.get_current_candidate_components(), sim.get_weights(), r, sim.is_terminal())

        for n in range(configs["tree_num"]):
            uct_tree_list[n].set_root_node(sim.get_state(), sim.get_actions(),
                                           sim.get_next_candidate_components(),
                                           sim.get_current_candidate_components(),
                                           sim.get_weights(), r, sim.is_terminal())

        #  configs['num_component'] - 3 - len(init_nodes) for computing how many component to add
        # + 3 + 2 * (configs['num_component'] - 3) for computing how many ports to consider
        # Here, if not sweep, int(not configs['sweep']) is 1, which means we add one step to chose duty cycle

        # True: for using fixed trajectory on every step(False decreasing trajectory)
        steps_traj = get_steps_traj(total_step * num_runs, total_step,
                                    int((num_runs / 10) ** 1.8), 2.7, False)

        # steps_traj = trajs_all_in_first_step(total_step, num_runs)
        # steps_traj = [num_runs for _ in range(total_step)]
        print(steps_traj)
        traj_idx = len(init_nodes) + len(init_edges)
        # return
        # for _ in range(1):
        while not sim.is_terminal():
            plan_start = datetime.datetime.now()

            step_traj = steps_traj[traj_idx]
            if step_traj == 0:
                step_traj = 1

            total_traj.value = step_traj
            print("current step is: ", steps)
            if steps == 0:
                for n in range(configs["tree_num"]):
                    conn_main[n].send(-1)
                    conn_main[n].send(uct_tree.root_)
                # conn_main[n].send(uct_tree_list[n].sim_.graph_2_reward)
            else:
                for n in range(configs["tree_num"]):
                    conn_main[n].send(action)
                    conn_main[n].send(sim.get_state())
                # conn_main[n].send(uct_tree_list[n].sim_.graph_2_reward)

            for n in range(configs["tree_num"]):
                n_tree_size = conn_main[n].recv()
                uct_tree_list[n] = conn_main[n].recv()
                depth = conn_main[n].recv()
                tree_size += n_tree_size

            plan_end_1 = datetime.datetime.now()
            instance_plan_time = (plan_end_1 - plan_start).seconds
            cumulate_plan_time += instance_plan_time

            # action = uct_tree_list[0].get_action()
            action = get_action_from_planners(uct_tree_list, uct_tree, configs["tree_num"])

            r = sim.act(action)

            avg_cumulate_reward += r
            steps += 1

            traj_idx += 1
            # Here means finishing component adding reset traj idx as num_component-3 to guarantee
            # the total number of edge selection is fixed. The graph == {} means graph is {} before
            # edge adding and not {} after the first step of edge adding
            if len(sim.current.component_pool) == sim.num_component_ and sim.current.graph == {}:
                traj_idx = 5

            print("instant reward:", uct_tree.root_.reward_, "cumulate reward: ", avg_cumulate_reward,
                  "planning time:", instance_plan_time, "cumulate planning time:", cumulate_plan_time)
            fo.write("instant reward:" + str(uct_tree.root_.reward_) + "cumulate reward: " + str(avg_cumulate_reward) +
                     "planning time:" + str(instance_plan_time) + "cumulate planning time:" + str(cumulate_plan_time))

        # get max
        for i in range(configs['tree_num']):
            uct_simulators[i] = uct_tree_list[i].sim_
        total_query, total_hash_query = get_total_querys(uct_simulators)
        sim = copy_simulators_info(sim, uct_simulators)

        sim.get_state().visualize(
            "result with parameter:" + str(sim.current.parameters) + " ", figure_folder)

        effis = []
        if inside_sim:

            # anal_results, simu_results, effis, save_tops, filted_save_tops = \
            #     get_multi_k_sim_results(sim=sim, configs=configs,
            #                             total_query=total_query, start_time=current_round_start_time,
            #                             total_planning_time=cumulate_plan_time, save_tops=save_tops,
            #                             anal_results=anal_results, simu_results=simu_results, filter_sim=filter_sim,
            #                             filted_save_tops=filted_save_tops)
            anal_results, simu_results, effis, save_tops, filted_save_tops = \
                parallel_get_multi_k_sim_results(sim=sim, configs=configs,
                                                 total_query=total_query, start_time=current_round_start_time,
                                                 total_planning_time=cumulate_plan_time, save_tops=save_tops,
                                                 anal_results=anal_results, simu_results=simu_results,
                                                 filter_sim=filter_sim,
                                                 filted_save_tops=filted_save_tops)


        print("##################### finish current")
        cumulate_reward_list.append(avg_cumulate_reward)

        avg_steps += steps
        avg_steps = avg_steps / configs["game_num"]
        fo = write_info_to_file(fo, sim, effis, avg_cumulate_reward, avg_steps, total_query, total_hash_query,
                                start_time, tree_size, configs)

        end_time = datetime.datetime.now()
        avg_step_list.append(avg_steps)

        # UCT_data_collection.save_sta_result(sim.key_sta, 'sta_only_epr.json')

        # UCT_data_collection.save_no_sweep_analytics_result(sim.key_expression)
        # TODO save simulation rewards
        # if not configs['skip_sim']:
        #     UCT_data_collection.save_no_sweep_sim_result(uct_simulators[0].key_sim_effi_)

        # if configs['get_traindata']:
        #     for isom_topo_and_key in sim.no_isom_seen_state_list:
        #         isom_topo, key = isom_topo_and_key[0], isom_topo_and_key[1]
        #         if key + '$' + str(isom_topo.parameters) not in sim.key_sim_effi_:
        #             continue
        #         else:
        #             eff = sim.key_sim_effi_[key + '$' + str(isom_topo.parameters)][0]
        #             vout = sim.key_sim_effi_[key + '$' + str(isom_topo.parameters)][1]
        #             if vout == -500:
        #                 continue
        #         isom_topo_dict[key + '$' + str(isom_topo.parameters)] = [isom_topo, eff, vout, 0, 0]
        #     update_dataset(isom_topo_dict)
        for n in range(configs["tree_num"]):
            conn_main[n].send(-2)
            conn_main[n].send(None)
        close_pipes(tree_num=configs['tree_num'], conn_main=conn_main, conn_sub=conn_sub)
        conn_main.clear()
        conn_sub.clear()

    print("figures are saved in:" + str(figure_folder) + "\n")
    print("outputs are saved in:" + out_file_name + "\n")

    for result in anal_results:
        fo.write(str(result) + "\n")
    fo.close()

    # save_reward_hash(sim)
    del result

    gc.collect()

    return {'sim': sim,
            'time': (end_time - start_time).seconds,
            'query_num': total_query,
            'state_list': uct_tree_list[0].get_all_states(),
            'uct_tree': uct_tree,
            'uct_tree_list': uct_tree_list
            }, anal_results, simu_results, save_tops, filted_save_tops
