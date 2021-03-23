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
from SimulatorSetPara.build_topology import nets_to_ngspice_files
from SimulatorSetPara.simulation import simulate_topologies
from SimulatorSetPara.simulation_analysis import analysis_topologies

from Viz.uctViz import delAllFiles, TreeGraph
import numpy as np
import datetime

from utils.util import init_position, generate_depth_list, del_all_files, mkdir, get_sim_configs, \
    read_reward_hash, save_reward_hash, get_steps_traj, read_reward_hash_list, read_approve_path,\
    read_joint_component_prob
import copy


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


def serial_UCF_test(depth_list, trajectory, configs, date_str):
    high_count = 0
    out_file_name = "Results/mutitest" + "-" + date_str + "-" + str(os.getpid()) + ".txt"
    figure_folder = "figures/" + date_str + "/"
    mkdir(figure_folder)
    want_tree_viz = True
    if want_tree_viz:
        viz_folder = "Viz/TreeStructures" + date_str + "-" + str(os.getpid()) + "/"
        mkdir(viz_folder)
        # viz_step = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        viz_step = []
    sim_configs = get_sim_configs(configs)
    uct_tree_list = []
    cumulate_reward_list = []
    start_time = datetime.datetime.now()

    fo = open(out_file_name, "w")
    fo.write("max_depth,num_runs,avg_step\n")
    avg_step_list = []
    init_nums = 1
    results = []
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
                approved_path_freq = read_approve_path()
                component_condition_prob = read_joint_component_prob(configs['num_component']-3)

                for i in range(0, int(configs["game_num"] / init_nums)):
                    steps = 0
                    cumulate_plan_time = 0
                    r = 0
                    tree_size = 0

                    sim = TopoPlanner.TopoGenSimulator(sim_configs, approved_path_freq, component_condition_prob,
                                                       configs['num_component'])

                    sim.no_isom_graph_2_reward = read_reward_hash("topo_info_hash.txt")
                    sim.encode_no_isom_graph_2_reward = read_reward_hash("encode_topo_info_hash.txt")

                    for k,v in sim.no_isom_graph_2_reward.items():
                        tmp_reward = v[0]
                        tmp_efficiency = v[1]
                        tmp_out_voltage = v[2]
                        if tmp_reward > 0.1:
                            print(k)
                            print(tmp_reward,tmp_efficiency,tmp_out_voltage)
                    # return
                    uct_tree = uct.UCTPlanner(sim, max_depth, num_runs, configs["ucb_scalar"], configs["gamma"],
                                              configs["leaf_value"], configs["end_episode_value"],
                                              configs["deterministic"], configs["rave_scalar"], configs["rave_k"],
                                              configs['component_default_policy'], configs['path_default_policy'])
                    uct_simulators.clear()
                    uct_tree_list.clear()
                    for n in range(configs["tree_num"]):
                        uct_simulators.append(TopoPlanner.TopoGenSimulator(sim_configs, approved_path_freq,
                                                                           component_condition_prob,
                                                                           configs['num_component']))
                        uct_simulators[n].no_isom_graph_2_reward = read_reward_hash("topo_info_hash.txt")
                        uct_simulators[n].encode_no_isom_graph_2_reward = read_reward_hash("encode_topo_info_hash.txt")

                        uct_tree_list.append(
                            uct.UCTPlanner(uct_simulators[n], max_depth, int(num_runs / configs["tree_num"]),
                                           configs["ucb_scalar"], configs["gamma"], configs["leaf_value"],
                                           configs["end_episode_value"], configs["deterministic"],
                                           configs["rave_scalar"], configs["rave_k"],
                                           configs['component_default_policy'], configs['path_default_policy']))

                    # For fixed commponent type
                    init_nodes = []
                    # init_nodes = [0, 3, 1]
                    # for e in init_nodes:
                    #     action = TopoPlanner.TopoGenAction('node', e)
                    #     sim.act(action)
                    edges = []
                    # adj = sim.get_adjust_parameter_set()
                    # print(sim.get_adjust_parameter_set())
                    # {2: {5}, 5: {2}, 1: {8}, 8: {1}, 0: {3}, 3: {0}, 4: {7}, 7: {4, 6}, 6: {7}})
                    # edges = [[0, 3], [1, 8], [2, 5], [4, 7], [6, 7]]
                    # edges = [[0, 8], [1, 3], [2, 4], [4, 6], [5, 7]]
                    # # # edges = [[0, 4], [1, 8], [2, 5]]
                    # #
                    # for edge in edges:
                    #     action = TopoPlanner.TopoGenAction('edge', edge)
                    #     sim.act(action)
                    # return
                    # topologies = [sim.get_state()]
                    # nets_to_ngspice_files(topologies, configs, configs['num_component'])
                    # simulate_topologies(len(topologies), configs['num_component'], configs["sys_os"])
                    # effis = analysis_topologies(configs, len(topologies), configs['num_component'])
                    # print("effis of topo:", effis)
                    # return
                    # sim.default_policy(mc_return, self.gamma_, discount, final_return, reward_list)

                    uct_tree.set_root_node(sim.get_state(), sim.get_actions(), r, sim.is_terminal())
                    for n in range(configs["tree_num"]):
                        uct_tree_list[n].set_root_node(sim.get_state(), sim.get_actions(), r, sim.is_terminal())
                    total_step = configs['num_component'] - 3 - len(init_nodes) + \
                                 3 + 2 * (configs['num_component'] - 3)
                    steps_traj = get_steps_traj(total_step * num_runs, total_step, 6, 2, True)
                    print(steps_traj)
                    while not sim.is_terminal():
                        # fo.write(str(steps)+"------step ---------------------------------------------" + "\n")
                        plan_start = datetime.datetime.now()
                        step_traj = steps_traj[sim.current.step - len(init_nodes)]
                        step_traj = int(step_traj / configs["tree_num"])

                        all_nodes = {}

                        for n in range(configs["tree_num"]):
                            print("sim.current.step", sim.current.step)
                            tree_size_tmp, tree_tmp, depth, node_list = uct_tree_list[n].plan(step_traj, True,
                                                                                              sim.current.step - (
                                                                                                      len(edges) + len(
                                                                                                  init_nodes)))

                            if steps in viz_step:
                                folder = viz_folder + "tree" + str(n) + "step" + str(steps) + "/"
                                is_exists = os.path.exists(folder)
                                if not is_exists:
                                    os.makedirs(folder)
                                else:
                                    delAllFiles(folder)
                                treeviz = TreeGraph(node_list)

                                treeviz.drawAll(uct_tree_list[n], folder, True)
                                tree_size += tree_size_tmp
                                if viz_step.index(steps) == len(viz_step) - 1:
                                    return

                            tree_size += tree_size_tmp
                            fo.write("increased tree size:" + str(tree_size_tmp) + "\n")

                        _root = uct_tree_list[0].root_
                        fo.write("root rave_num_visits:" + str(_root.rave_num_visits_))
                        for child_idx in range(len(_root.node_vect_)):
                            child = _root.node_vect_[child_idx]
                            print("action ", child_idx, " :", _root.act_vect_[child_idx].type, "on",
                                  _root.act_vect_[child_idx].value)
                            fo.write("action " + str(child_idx) + " :" + str(_root.act_vect_[child_idx].type) + "on" +
                                     str(_root.act_vect_[child_idx].value) + "\n")
                            print("action child ", child_idx, " avg_return:", child.avg_return_)
                            print("action child ", child_idx, " num_visits:", child.num_visits_)
                            print("action child ", child_idx, " rave_avg_return:", child.rave_avg_return_)
                            print("action child ", child_idx, " rave_num_visits:", child.rave_num_visits_)
                            fo.write("action child " + str(child_idx) + " avg_return:" + str(child.avg_return_) + "\n")
                            fo.write("action child " + str(child_idx) + " num_visits_:" + str(child.num_visits_) + "\n")
                            fo.write("action child " + str(child_idx) + " rave_avg_return:" + str(
                                child.rave_avg_return_) + "\n")
                            fo.write("action child " + str(child_idx) + " rave_num_visits_:" + str(
                                child.rave_num_visits_) + "\n")

                            child_node = child.state_vect_[0]
                            child_state = child_node.state_
                            print("state child ", child_idx, " reward:", child_node.reward_)
                            fo.write(
                                "state child " + str(child_idx) + "child node reward:" + str(child_node.reward_) + "\n")
                            print("state ", child_idx, "ports:", child_state.port_pool)
                            fo.write("state child " + str(child_idx) + "ports:" + str(child_state.port_pool) + "\n")
                            print("state child", child_idx, "graph:", child_state.graph)
                            fo.write("state child " + str(child_idx) + "graph:" + str(child_state.graph) + "\n")
                        plan_end_1 = datetime.datetime.now()
                        instance_plan_time = (plan_end_1 - plan_start).seconds
                        cumulate_plan_time += instance_plan_time

                        if configs["act_selection"] == "Pmerge":
                            action = get_action_from_planners(uct_tree_list, uct_tree, configs["tree_num"])
                        elif configs["act_selection"] == "Tmerge":
                            action = get_action_from_trees(uct_tree_list, uct_tree, configs["tree_num"])
                        elif configs["act_selection"] == "Vote":
                            action = get_action_from_trees_vote(uct_tree_list, uct_tree, configs["tree_num"])

                        if configs["output"]:
                            print("{}-action:".format(steps), end='')
                            action.print()
                            fo.write("take the action: type:" + str(action.type) +
                                     " value: " + str(action.value) + "\n")
                            print("{}-state:".format(steps), end='')

                        r = sim.act(action)

                        if sim.get_state().parent:
                            if action.type == 'node':
                                act_str = 'adding node {}'.format(sim.basic_components[action.value])
                            elif action.type == 'edge':
                                if action.value[1] < 0 or action.value[0] < 0:
                                    act_str = 'skip connecting'
                                else:
                                    act_str = 'connecting {} and {}'.format(sim.current.idx_2_port[action.value[0]],
                                                                            sim.current.idx_2_port[action.value[1]])
                            else:
                                act_str = 'terminal'
                            sim.get_state().visualize(act_str, figure_folder)
                        for n in range(configs["tree_num"]):
                            uct_tree_list[n].update_root_node(action, sim.get_state())
                        avg_cumulate_reward += r
                        steps += 1
                        print("instant reward:", uct_tree.root_.reward_, "cumulate reward: ", avg_cumulate_reward,
                              "planning time:", instance_plan_time, "cumulate planning time:", cumulate_plan_time)
                        fo.write("instant reward:" + str(uct_tree.root_.reward_) +
                                 "cumulate reward: " + str(avg_cumulate_reward) +
                                 "planning time:" + str(instance_plan_time) +
                                 "cumulate planning time:" + str(cumulate_plan_time))

                    topologies = [sim.get_state()]
                    nets_to_ngspice_files(topologies, configs, configs['num_component'])
                    simulate_topologies(len(topologies), configs['num_component'], configs["sys_os"])
                    effis = analysis_topologies(configs, len(topologies), configs['num_component'])


                    print("effis of topo:", effis)
                    print("#####################Game:", i, "  steps: ", steps, "  average cumulate reward: ",
                          avg_cumulate_reward)
                    cumulate_reward_list.append(avg_cumulate_reward)
                    avg_steps += steps

                    avg_steps = avg_steps / configs["game_num"]
                    fo.write("Final topology of game " + str(i) + ":\n")
                    fo.write("component_pool:" + str(sim.current.component_pool) + "\n")
                    fo.write(str(sim.current.parameters) + "\n")
                    fo.write("port_pool:" + str(sim.current.port_pool) + "\n")
                    fo.write("graph:" + str(sim.current.graph) + "\n")
                    fo.write("efficiency:" + str(effis) + "\n")
                    fo.write("final reward:" + str(avg_cumulate_reward) + "\n")
                    fo.write("step:" + str(avg_steps) + "\n")
                    total_query = sim.query_counter
                    total_hash_query = sim.hash_counter
                    for simulator in uct_simulators:
                        total_query += simulator.query_counter
                        total_hash_query += simulator.hash_counter
                    fo.write("query time:" + str(total_query) + " total tree size:" + str(tree_size) + "\n")
                    fo.write("hash query time:" + str(total_hash_query) + "\n")
                    end_time = datetime.datetime.now()
                    sim.get_state().visualize(
                        "final of" + str(max_depth) + " " + str(num_runs) + " " + str(i) + " " + out_file_name,
                        figure_folder)
                    fo.write("end at:" + str(end_time) + "\n")
                    fo.write("start at:" + str(start_time) + "\n")
                    fo.write("execute time:" + str((end_time - start_time).seconds) + " seconds\n")
                    fo.write("----------------------------------------------------------------------" + "\n")
                    print(max_depth, ",", num_runs, ":", avg_steps)
                    avg_step_list.append(avg_steps)
                fo.write("configs:" + str(configs) + "\n")
                fo.write("final rewards:" + str(avg_cumulate_reward) + "\n")
            result = "Traj: " + str(num_runs)
            if effis[0]['efficiency'] > 0.8:
                high_count +=1
            result = result + "#Efficiency:" + str(effis[0]['efficiency']) + \
                     "#FinalRewards:" + str(avg_cumulate_reward) + \
                     "#ExecuteTime:" + str((end_time - start_time).seconds) + \
                     "#QueryTime:" + str(total_query) + "#TreeSize:" + str(tree_size)
            results.append(result)

    print("figures are saved in:" + str(figure_folder) + "\n")
    print("outputs are saved in:" + out_file_name + "\n")

    del_all_files(str(configs['num_component']) + "component_data_random")
    del_all_files("sim_analysis")

    for tmp_uct_tree in uct_tree_list:
        sim.graph_2_reward.update(tmp_uct_tree.sim_.graph_2_reward)

    for result in results:
        fo.write(result + "\n")
    fo.write("time of high efficiency: " + str(high_count) + "\n")

    save_reward_hash(sim)
    # print(read_reward_hash())

    fo.close()
    return
