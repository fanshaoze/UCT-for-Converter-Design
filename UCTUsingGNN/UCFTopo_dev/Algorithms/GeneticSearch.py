from copy import deepcopy
import datetime
from ucts import TopoPlanner
from simulator3component.build_topology import nets_to_ngspice_files
from simulator3component.simulation import simulate_topologies
from simulator3component.simulation_analysis import analysis_topologies
from utils.util import get_sim_configs, mkdir, del_all_files


def genetic_search(configs, date_str):
    out_file_name = "Results/mutitest" + "-" + date_str + ".txt"
    figure_folder = "figures/" + date_str + "/"
    mkdir(figure_folder)
    fo = open(out_file_name + "GS", "w")

    num_games = configs["game_num"]
    output = True
    num_component = configs["num_component"]
    sys_os = configs["sys_os"]
    start_time = datetime.datetime.now()

    sim_configs = get_sim_configs(configs)
    for _ in range(0, num_games):
        sim = TopoPlanner.TopoGenSimulator(sim_configs, num_component)
        sim.random_generate_graph()
        print("num_component:", sim.current.num_component)
        print("component_pool:", sim.current.component_pool)
        print("port_pool:", sim.current.port_pool)
        print("port2comp_mapping:", sim.current.port2comp_mapping)
        print("comp2port_mapping:", sim.current.comp2port_mapping)
        print("same_device_mapping:", sim.current.same_device_mapping)
        print("idx_2_port:", sim.current.idx_2_port)
        print("port_2_idx:", sim.current.port_2_idx)
        print("graph:", sim.current.graph)
        print("parent:", sim.current.parent)
        print("count_map:", sim.current.count_map)
        print("step:", sim.current.step)
        sim.get_state().visualize("GS init" + out_file_name + "GS", figure_folder)


        return
        print("--------------------------------init state is valid-----------------------------")
        fo.write("-------------------------------init state is valid----------\n")

        topologies = [sim.get_state()]
        nets_to_ngspice_files(topologies, configs, num_component)
        simulate_topologies(len(topologies), num_component, sys_os)
        effis = analysis_topologies(configs, len(topologies), num_component)
        fo.write("init topology:" + "\n")
        print("effis of topo:", effis)
        fo.write(str(sim.current.port_pool) + "\n")
        fo.write(str(sim.current.graph) + "\n")
        fo.write(str(effis) + "\n")
        print("port pool:", str(sim.current.port_pool) + "\n")
        print("graph:", str(sim.current.graph) + "\n")
        print("effi:", str(effis) + "\n")
        init_state = sim.get_state()
        origin_state = deepcopy(init_state)
        origin_query = sim.query_counter
        sim.get_state().visualize("GS init" + out_file_name + "GS", figure_folder)

        mutate_num = configs["mutate_num"]
        generation_num = configs["mutate_generation"]
        step = 50
        results = []
        for k in range(generation_num, generation_num + step, step):
            fo.write("generation state with" + str(k) + "\n")
            sim.set_state(origin_state)
            max_reward = sim.get_reward()
            for j in range(k):
                mutations = []
                mutation_rewards = []
                i = 0
                while i < mutate_num:
                    sim.set_state(init_state)
                    mutated_state, reward, change = sim.mutate()
                    if mutated_state is None and reward == -1:
                        continue
                    if not mutated_state.graph_is_valid():
                        print("not valid graph:", mutated_state.graph)
                    else:
                        mutations.append(mutated_state)
                        mutation_rewards.append(reward)
                    fo.write(str(i) + " " + str(j) + " " + change + " reward " + str(reward) + "\n")
                    if output:
                        print(i, j, "------------------------------------")
                        print("mutated is same as inited", sim.get_state().equal(init_state))
                    print(init_state.graph)
                    print(i, " ", j, " ", change, " ", sim.get_state().graph)
                    i += 1

                sim.random_generate_graph()
                mutations.append(sim.get_state())
                reward = sim.get_reward()
                mutation_rewards.append(reward)
                print("mutation_rewards:", mutation_rewards)
                fo.write("random" + " " + str(j) + " " + change + " reward " + str(reward) + "\n")

                mutations.append(init_state)
                mutation_rewards.append(max_reward)

                max_reward = max(mutation_rewards)
                init_state = mutations[mutation_rewards.index(max_reward)]
                sim.set_state(init_state)
                fo.write(str(j) + " step GS best: -----------------------------------------" + "\n")
                fo.write(str(sim.current.port_pool) + "\n")
                fo.write(str(sim.current.graph) + "\n")
                fo.write("max reward of " + str(j) + " " + str(max_reward) + "\n")
                topologies = [sim.get_state()]
                nets_to_ngspice_files(topologies, configs, num_component)
                simulate_topologies(len(topologies), num_component, sys_os)
                effis = analysis_topologies(configs, len(topologies), num_component)
                print("***************graph of mutate", j, ":", init_state.graph)
                print("***************reward of mutate", j, ":", max_reward)
                fo.write(str(effis) + "\n")
            sim.current.visualize("GS result of:" + str(k) + " in " + out_file_name, figure_folder)
            results.append((k, sim.current.port_pool, str(sim.current.graph), str(effis), str(max_reward)))
            fo.write("finish GS of " + str(k) + "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" + "\n")
            fo.write("query of " + str(k) + " = " + str(sim.query_counter) + "\n")
            fo.write("hash of " + str(k) + " = " + str(sim.hash_counter) + "\n")
        print("Finished genetic search")
        sim.set_state(origin_state)
        fo.write("--------------------Finished genetic search-------------------\n")
        for k in range(len(results)):
            fo.write("result of " + str(k) + "+++++++++++++++++++++++++++++++" + "\n")
            fo.write(str(results[k][0]) + "\n")
            fo.write(str(results[k][1]) + "\n")
            fo.write(str(results[k][2]) + "\n")
            fo.write(str(results[k][3]) + "\n")
            fo.write(str(results[k][4]) + "\n")
        end_time = datetime.datetime.now()
        fo.write("origin query:" + str(origin_query) + "\n")
        fo.write("end at:" + str(end_time) + "\n")
        fo.write("start at:" + str(start_time) + "\n")
        fo.write("execute time:" + str((end_time - start_time).seconds) + " seconds\n")
    print("figures are saved in:" + str(figure_folder) + "\n")
    print("outputs are saved in:" + out_file_name + "\n")
    del_all_files(str(configs['num_component']) + "component_data_random")
    del_all_files("sim_analysis")
    fo.close()
    return
