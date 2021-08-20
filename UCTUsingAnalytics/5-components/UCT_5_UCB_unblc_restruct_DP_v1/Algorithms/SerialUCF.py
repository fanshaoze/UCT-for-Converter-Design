import _thread
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

import gc


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
        approved_path_freq = read_approve_path(0.0, '3comp_buck_boost_sim_path_freqs.json')
        component_condition_prob = read_joint_component_prob(configs['num_component'] - 3,
                                                             None, "3comp_buck_boost_sim_node_joint_probs.json")

        print(approved_path_freq)
        print(component_condition_prob)

    elif 0 < configs['target_vout'] < 100:
        approved_path_freq = read_approve_path(0.0, '3comp_buck_sim_path_freqs.json')
        component_condition_prob = read_joint_component_prob(configs['num_component'] - 3,
                                                             None, "3comp_buck_sim_node_joint_probs.json")

        print(approved_path_freq)
        print(component_condition_prob)


    elif 100 < configs['target_vout'] < target_max_vout:
        approved_path_freq = read_approve_path(0.0, '3comp_boost_sim_path_freqs.json')
        component_condition_prob = read_joint_component_prob(configs['num_component'] - 3,
                                                             None, "3comp_boost_sim_node_joint_probs.json")
        print(approved_path_freq)
        print(component_condition_prob)
    else:
        return None
    return approved_path_freq, component_condition_prob


def serial_UCF_test(trajectory, test_number, configs, result_folder):
    path = './SimulatorAnalysis/database/analytic-expression.json'
    is_exits = os.path.exists(path)
    if not is_exits:
        UCT_data_collection.key_expression_dict()
    print("finish reading key-expression")
    out_file_folder = 'Results/'+result_folder+'/'
    mkdir(out_file_folder)
    out_file_name = out_file_folder+str(trajectory)+'-result.txt'
    out_round_folder = 'Results/'+result_folder+'/'+str(trajectory)
    mkdir(out_round_folder)

    # out_file_name = "Results/mutitest_" + str(configs['target_vout']) + "-" + date_str + "-" + str(os.getpid()) + ".txt"
    figure_folder = "figures/" + result_folder + "/"
    mkdir(figure_folder)

    sim_configs = get_sim_configs(configs)
    start_time = datetime.datetime.now()

    fo = open(out_file_name, "w")
    fo.write("max_depth,num_runs,avg_step\n")
    avg_step_list = []
    anal_results = [['efficiency', 'vout', 'reward', 'DC_para', 'query']]
    simu_results = [['efficiency', 'vout', 'reward', 'DC_para', 'query']]

    for test_idx in range(test_number):
            num_runs = trajectory
            avg_steps = 0
            print()
            avg_cumulate_reward = 0
            cumulate_reward_list = []
            fo.write("----------------------------------------------------------------------" + "\n")
            uct_simulators = []
            uct_tree_list = []

            approved_path_freq, component_condition_prob = read_DP_files(configs)
            key_expression = UCT_data_collection.read_analytics_result()
            key_sim_effi = UCT_data_collection.read_sim_result()

            basic_components = ['Sa', 'Sb', 'L', 'C']
            preset_component_num = 0
            # preset_component_idx = random.choice([0, 1])
            # init_nodes = [preset_component_idx]
            init_nodes = []
            # preset_component = basic_components[preset_component_idx]
            component_priorities = get_component_priorities()
            # component_priorities = get_component_priorities(preset_component)
            # TODO must be careful, if we delete the random adding of sa,sb,
            #  we also need to change the preset comp number
            _unblc_comp_set_mapping, _ = unblc_comp_set_mapping(['Sa', 'Sb', 'L', 'C'],
                                                                configs['num_component'] - 3 - preset_component_num)
            for k, v in _unblc_comp_set_mapping.items():
                print(k, '\t', v)
            # TODO: just for test, remember to remove
            # for k, v in _unblc_comp_set_mapping.items():
            #     _unblc_comp_set_mapping[k] = 1
            # return

            steps = 0
            cumulate_plan_time = 0
            r = 0
            tree_size = 0

            sim = TopoPlanner.TopoGenSimulator(sim_configs, approved_path_freq, component_condition_prob,
                                               key_expression, _unblc_comp_set_mapping, component_priorities,
                                               key_sim_effi,
                                               None,
                                               configs['num_component'])

            uct_tree = uct.UCTPlanner(sim, -1, num_runs, configs["ucb_scalar"], configs["gamma"],
                                      configs["leaf_value"], configs["end_episode_value"],
                                      configs["deterministic"], configs["rave_scalar"], configs["rave_k"],
                                      configs['component_default_policy'], configs['path_default_policy'])

            uct_simulators.clear()
            uct_tree_list.clear()
            for n in range(configs["tree_num"]):
                uct_simulators.append(
                    TopoPlanner.TopoGenSimulator(sim_configs, approved_path_freq, component_condition_prob,
                                                 key_expression, _unblc_comp_set_mapping, component_priorities,
                                                 key_sim_effi,
                                                 None,
                                                 configs['num_component']))

                uct_tree_list.append(
                    uct.UCTPlanner(uct_simulators[n], -1, int(num_runs / configs["tree_num"]),
                                   configs["ucb_scalar"], configs["gamma"], configs["leaf_value"],
                                   configs["end_episode_value"], configs["deterministic"],
                                   configs["rave_scalar"], configs["rave_k"], configs['component_default_policy'],
                                   configs['path_default_policy']))

            # For fixed commponent type
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
            # return

            uct_tree.set_root_node(sim.get_state(), sim.get_actions(), sim.get_next_candidate_components(),
                                   sim.get_current_candidate_components(), sim.get_weights(), r, sim.is_terminal())

            for n in range(configs["tree_num"]):
                uct_tree_list[n].set_root_node(sim.get_state(), sim.get_actions(),
                                               sim.get_next_candidate_components(),
                                               sim.get_current_candidate_components(),
                                               sim.get_weights(), r, sim.is_terminal())

            #  configs['num_component'] - 3 - len(init_nodes) for computing how many component to add
            # + 3 + 2 * (configs['num_component'] - 3) for computing how many ports to consider
            total_step = configs['num_component'] - 3 - len(init_nodes) + \
                         3 + 2 * (configs['num_component'] - 3) - len(init_edges)
            # True: for using fixed trajectory on every step(False decreasing trajectory)
            steps_traj = get_steps_traj(total_step * num_runs, total_step,
                                        int((num_runs / 10) ** 1.8), 2.7, False)
            # TODO just for test
            # steps_traj = [33, 17, 7, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            print(steps_traj)
            traj_idx = len(init_nodes) + len(init_edges)
            traj_idx = 0
            # return

            while not sim.is_terminal():
            # for _ in range(1):
                plan_start = datetime.datetime.now()
                # step_traj = steps_traj[sim.current.step]
                step_traj = steps_traj[traj_idx]
                step_traj = int(step_traj / configs["tree_num"])
                if step_traj == 0:
                    step_traj = 1
                for n in range(configs["tree_num"]):
                    print("sim.current.step", sim.current.step)
                    tree_size_tmp, tree_tmp, depth = \
                        uct_tree_list[n].plan(step_traj, False)
                    tree_size += tree_size_tmp
                    fo.write("increased tree size:" + str(tree_size_tmp) + "\n")

                _root = uct_tree_list[0].root_
                for child_idx in range(len(_root.node_vect_)):
                    child = _root.node_vect_[child_idx]
                    child_node = child.state_vect_[0]
                    child_state = child_node.state_
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
                plan_end_1 = datetime.datetime.now()
                instance_plan_time = (plan_end_1 - plan_start).seconds
                cumulate_plan_time += instance_plan_time

                action = uct_tree_list[0].get_action()

                if configs["output"]:
                    print("{}-action:".format(steps), end='')
                    action.print()
                    fo.write("take the action: type:" + str(action.type) +
                             " value: " + str(action.value) + "\n")
                    print("{}-state:".format(steps), end='')

                r = sim.act(action)

                for n in range(configs["tree_num"]):
                    uct_tree_list[n].update_root_node(action, sim.get_state())
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
                fo.write("instant reward:" + str(uct_tree.root_.reward_) +
                         "cumulate reward: " + str(avg_cumulate_reward) +
                         "planning time:" + str(instance_plan_time) +
                         "cumulate planning time:" + str(cumulate_plan_time))


            # get max
            sim.graph_2_reward = uct_simulators[0].graph_2_reward
            sim.current_max = uct_simulators[0].current_max
            sim.current, sim.reward, sim.current.parameters = sim.get_max_seen()
            max_result = sim.reward
            effis = [sim.get_effi_info()]

            max_sim_reward_result = uct_simulators[0].get_tops_sim_info()
            # max_sim_reward_result = {'max_sim_effi': 0, 'max_sim_vout': 0, 'max_sim_reward': 0, 'max_sim_para': 0}

            for_simulation_file = 'round'+'_'+str(test_idx)
            results_tmp = uct_simulators[0].generate_topk_simulation_base_dict()
            with open(out_round_folder + '/' + for_simulation_file + '.json', 'w') as f:
                json.dump(results_tmp, f)
            f.close()


            print("effis of topo:", effis)
            print("#####################Game:", "  steps: ", steps, "  average cumulate reward: ",
                  avg_cumulate_reward)
            cumulate_reward_list.append(avg_cumulate_reward)
            avg_steps += steps

            avg_steps = avg_steps / configs["game_num"]
            fo.write("Final topology of game " + ":\n")
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
            final_para_str = sim.current.parameters
            sim.get_state().visualize(
                "result with parameter:" + str(str(final_para_str)) + " ", figure_folder)
            fo.write("end at:" + str(end_time) + "\n")
            fo.write("start at:" + str(start_time) + "\n")
            fo.write("execute time:" + str((end_time - start_time).seconds) + " seconds\n")
            fo.write("result with parameter:" + str(str(final_para_str)) + "\n")
            fo.write("----------------------------------------------------------------------" + "\n")

            avg_step_list.append(avg_steps)

            fo.write("configs:" + str(configs) + "\n")
            fo.write("final rewards:" + str(avg_cumulate_reward) + "\n")

            result = "Traj: " + str(num_runs)
            print(effis)
            anal_result = [effis[0]['efficiency'], effis[0]['output_voltage'],
                       max_result, final_para_str, total_query]
            anal_results.append(anal_result)
            simu_result = [max_sim_reward_result['max_sim_effi'], max_sim_reward_result['max_sim_vout'],
                           max_sim_reward_result['max_sim_reward'], max_sim_reward_result['max_sim_para'],
                           total_query]
            simu_results.append(simu_result)
            # UCT_data_collection.save_analytics_result(uct_simulators[0].key_expression)
            UCT_data_collection.save_sta_result(uct_simulators[0].key_sta, 'sta_only_epr.json')
            print('hash counter', uct_simulators[0].hash_counter)
            print('hash length', len(uct_simulators[0].graph_2_reward))
            UCT_data_collection.save_sim_result(uct_simulators[0].key_sim_effi_)

            del sim
            del uct_simulators
            del uct_tree
            del uct_tree_list
            # del key_expression
            gc.collect()

    print("figures are saved in:" + str(figure_folder) + "\n")
    print("outputs are saved in:" + out_file_name + "\n")

    for result in anal_results:
        fo.write(str(result) + "\n")
    fo.close()

    # save_reward_hash(sim)
    del result

    gc.collect()
    return anal_results, simu_results

