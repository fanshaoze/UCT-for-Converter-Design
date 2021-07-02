import os
import random
from copy import deepcopy
import datetime
from ucts import TopoPlanner
import datetime
from utils.util import mkdir, get_sim_configs, save_reward_hash, get_steps_traj, read_approve_path, \
    read_joint_component_prob, read_DP_files
from SimulatorAnalysis import UCT_data_collection

import gc

from utils.util import get_sim_configs, mkdir, del_all_files


def genetic_search(trajectory, configs, date_str):
    out_file_name = "Results/mutitest_" + str(configs['target_vout']) + "-" + date_str + "-" + str(os.getpid()) +\
                    ".txtGS"

    figure_folder = "figures/" + date_str + "/"
    mkdir(figure_folder)
    fo = open(out_file_name, "w")

    start_time = datetime.datetime.now()
    approved_path_freq, component_condition_prob = read_DP_files(configs)
    key_expression = UCT_data_collection.read_analytics_result()

    sim_configs = get_sim_configs(configs)
    results = []
# [600,600,600,600]
    for Traj in trajectory:
        sim = TopoPlanner.TopoGenSimulator(sim_configs, approved_path_freq, component_condition_prob,
                                           key_expression, configs['num_component'])
        function_sim = TopoPlanner.TopoGenSimulator(sim_configs, approved_path_freq, component_condition_prob,
                                                    key_expression, configs['num_component'])

        sim.current.print()
        origin_query = sim.query_counter
        # number of mutation in each round
        select_num = configs["mutate_num"]
        # number of individuals(last remained tops + newly generated) in every generation
        individual_num = configs["individual_num"]
        individuals = []

        def keyFunc(element):
            return element[1]

        init_state = TopoPlanner.TopoGenState(init=True)
        origin_state = deepcopy(init_state)
        for _ in range(individual_num):
            sim.set_state(deepcopy(init_state))
            # mc_return = 0, reward_list = [], discount = 1, final_return = None
            final_result = sim.default_policy(0, configs["gamma"], 1, None, [],
                                              configs["component_default_policy"], configs["path_default_policy"])
            individuals.append([sim.get_state(), final_result])
        individuals.sort(key=keyFunc)  # actually we can use Select(A, n)
        top_topos = individuals[-select_num:]
        current_total_traj = individual_num
        generation_num = 0
        while current_total_traj < Traj:

            fo.write("generation state with" + str(generation_num) + "\n")
            # max_reward = sim.get_reward()
            individuals.clear()
            for top_topo in top_topos:
                individuals.append(top_topo)
            i = 0
            while i < individual_num:
                top_topo = random.choice(top_topos)
                sim.set_state(top_topo[0])
                # >0.1 for crossover
                if random.random() < 0.8:
                # if random.random() < 1:
                # if random.random() < 0:
                    mutation_probs = [0.1, 0.3, 0.2, 0.2]
                    # mutation_probs = [1, 0.1, 0.1, 0.1]
                    tmp_comp_pool = deepcopy(sim.current.component_pool)
                    tmp_port_pool = deepcopy(sim.current.port_pool)
                    tmp_idx_2_port = deepcopy(sim.current.idx_2_port)
                    tmp_parent = deepcopy(sim.current.parent)

                    mutated_state, reward, change = sim.mutate(mutation_probs)

                    if reward >= 0.8:
                        print('pre comp pool:',tmp_comp_pool)
                        print('pre port pool:',tmp_port_pool)
                        print('pre idx to port:',tmp_idx_2_port)
                        print('pre parent:',tmp_parent)
                        print(change)
                        print('current comp pool:',mutated_state.component_pool)
                        print('current port pool:',mutated_state.port_pool)
                        print('current idx to port:',mutated_state.idx_2_port)
                        print('current parent:',mutated_state.parent)
                    if mutated_state is None and reward == -1:
                        continue
                    else:
                        i += 1

                        if mutated_state.graph_is_valid():
                            current_total_traj += 1
                        individuals.append([mutated_state, reward])
                        fo.write(str(i) + " " + str(generation_num) + " " + change + " reward " + str(reward) + "\n")
                else:
                    crossover_topos = random.choices(top_topos,k=2)
                    sim.set_state(crossover_topos[0][0])
                    function_sim.set_state(crossover_topos[1][0])
                    function_sim.graph_2_reward = sim.graph_2_reward
                    mutated_state_0, reward_0, change_0, mutated_state_1, reward_1, change_1 \
                        = TopoPlanner.crossover(sim, function_sim)
                    if (mutated_state_0 is None) or (mutated_state_1 is None) or \
                            (reward_0 == -1) or (reward_1 == -1):
                        print(mutated_state_0, reward_0, mutated_state_1, reward_1)
                        continue
                    else:
                        i += 2
                        if mutated_state_0.graph_is_valid():
                            current_total_traj += 1
                        if mutated_state_1.graph_is_valid():
                            current_total_traj += 1
                        individuals.append([mutated_state_0, reward_0])
                        individuals.append([mutated_state_1, reward_1])
                        fo.write(str(i) + " " + str(generation_num) + " " + change_0 + ' ' +
                                 change_1 + " rewards " + str(reward_0) + ' ' +
                                 str(reward_1) + "\n")

                # add a random topo
            for _ in range(int(individual_num/5)):
                sim.set_state(deepcopy(init_state))
                print(configs["component_default_policy"], configs["path_default_policy"])
                final_result = sim.default_policy(0, configs["gamma"], 1, None, [],
                                                  configs["component_default_policy"], configs["path_default_policy"])

                fo.write("random" + " " + str(generation_num) + " " + " reward " + str(final_result) + "\n")
                individuals.append([sim.get_state(), final_result])
                current_total_traj += 1
            individuals.sort(key=keyFunc)  # actually we can use Select(A, n)
            top_topos = individuals[-select_num:]
            step_best = top_topos[-1]

            fo.write(str(generation_num) + " step GS best: -----------------------------------------" + "\n")
            fo.write(str(step_best[0].port_pool) + "\n")
            fo.write(str(step_best[0].graph) + "\n")
            fo.write("max reward of " + str(generation_num) + " " + str(step_best[1]) + "\n")

            generation_num += 1

        sim.current, sim.reward, sim.current.parameters = sim.get_max_seen()
        effis = [sim.get_effi_info()]

        print("effis of topo:", effis)

        fo.write("Final topology of game " + ":\n")
        fo.write("component_pool:" + str(sim.current.component_pool) + "\n")
        fo.write(str(sim.current.parameters) + "\n")
        fo.write("port_pool:" + str(sim.current.port_pool) + "\n")
        fo.write("graph:" + str(sim.current.graph) + "\n")
        fo.write("efficiency:" + str(effis) + "\n")
        fo.write("final reward:" + str(sim.reward) + "\n")
        total_query = sim.query_counter + function_sim.query_counter
        total_hash_query = sim.hash_counter + function_sim.hash_counter
        fo.write("query time:" + str(total_query) + "\n")
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

        fo.write("configs:" + str(configs) + "\n")
        fo.write("final rewards:" + str(sim.reward) + "\n")

        result = "#Efficiency:" + str(effis[0]['efficiency']) + "#FinalRewards:" + str(
            sim.reward) + "#ExecuteTime:" + str((end_time - start_time).seconds) + "#QueryTime:" + str(
            total_query) + "#HashSize:" + str(len(sim.graph_2_reward))
        results.append(result)
        UCT_data_collection.save_analytics_result(sim.key_expression)
        print('hash counter', total_hash_query)
        print('hash length', len(sim.graph_2_reward))
        print("Finished genetic search")
        sim.set_state(origin_state)
        fo.write("--------------------Finished genetic search-------------------\n")
    print("figures are saved in:" + str(figure_folder) + "\n")
    print("outputs are saved in:" + out_file_name + "\n")
    for result in results:
        fo.write(result + "\n")
    fo.close()
    # save_reward_hash(sim)
    del result
    gc.collect()
    return
