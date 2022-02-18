from copy import deepcopy

from SimulatorAnalysis.gen_topo import *
from SimulatorAnalysis.UCT_data_collection import *
from ucts.GetReward import *
from multiprocessing import Process, Manager, Pipe, Value, Queue


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


def simulate_one_analytics_result(analytics_info, file_token):
    """
        input a topology, return the simulation information of different duty cycles
        """
    result_dict = {}
    if analytics_info is None:
        return {'[]': {'efficiency': 0, 'Vout': -500}}
    analytic = analytics_info
    cki_folder = './SimulatorAnalysis/database/cki/'
    for fn in analytic:
        print(fn)
        count = 0
        for param in analytic[fn]:
            param_value = [float(value) for value in param[1:-1].split(', ')]
            device_name = analytic[fn][param][0]
            netlist = analytic[fn][param][1]
            file_name = file_token + '_' + fn + '-' + str(count) + '.cki'
            convert_cki_full_path(file_name, param_value, device_name, netlist)
            print(file_name)
            simulate(file_name)
            count = count + 1
    # only one fn in analytic if only simulate one
    for fn in analytic:
        count = 0
        for param in analytic[fn]:
            param_value = [float(value) for value in param[1:-1].split(', ')]
            device_name = analytic[fn][param][0]
            netlist = analytic[fn][param][1]
            file_name = file_token + '_' + fn + '-' + str(count) + '.simu'
            path = file_name
            vin = param_value[device_name['Vin']]
            freq = param_value[device_name['Frequency']] * 1000000
            rin = param_value[device_name['Rin']]
            rout = param_value[device_name['Rout']]
            print(file_name)
            result = calculate_efficiency(path, vin, freq, rin, rout)
            # print(result)
            param = str(param)
            param_spl = param.split(',')
            para_val = param_spl[0].replace('(', '')
            result_dict[para_val] = result
            count = count + 1
        return result_dict


def get_single_topo_sim_result(current, sweep, candidate_params, key_sim_effi_, skip_sim, key_expression_mapping,
                               target_vout, min_vout, file_token='1'):
    effi_info = {}
    topk_max_reward, topk_max_para, topk_max_effi, topk_max_vout = -1, [], 0, 500

    list_of_node, list_of_edge, netlist, joint_list = convert_to_netlist(current.graph,
                                                                         current.component_pool,
                                                                         current.port_pool,
                                                                         current.parent,
                                                                         current.comp2port_mapping)
    current_key = key_circuit_from_lists(list_of_edge, list_of_node, netlist)
    if not sweep:
        candidate_paras = [current.parameters]
    else:
        candidate_paras = candidate_params
    for para in candidate_paras:
        current.parameters = para
        if key_sim_effi_.__contains__(current_key + '$' + str(current.parameters)):
            effi_info_from_hash = key_sim_effi_[current_key + '$' + str(current.parameters)]
            print('find pre simulated &&&&&&&&&&&&&&&&&&&&&&')
            effi_info[str(current.parameters)] = {'para': current.parameters, 'efficiency': effi_info_from_hash[0],
                                                  'Vout': effi_info_from_hash[1]}

        else:
            if skip_sim:
                effi_info[str(current.parameters)] = {'para': current.parameters, 'efficiency': 0, 'Vout': 500}
            else:
                if current.graph_is_valid():

                    results_tmp = get_analytics_file(key_expression_mapping, current.graph,
                                                     current.comp2port_mapping,
                                                     current.port2comp_mapping, current.idx_2_port,
                                                     current.port_2_idx,
                                                     current.parent, current.component_pool,
                                                     current.same_device_mapping, current.port_pool,
                                                     {'Duty_Cycle': [current.parameters[0]],
                                                      'C': [current.parameters[1]],
                                                      'L': [current.parameters[2]]}, min_vout)
                    # [float(current.parameters)], min_vout)
                else:
                    results_tmp = None
                list_of_node, list_of_edge, netlist, joint_list = convert_to_netlist(current.graph,
                                                                                     current.component_pool,
                                                                                     current.port_pool,
                                                                                     current.parent,
                                                                                     current.comp2port_mapping)
                current_key = key_circuit_from_lists(list_of_edge, list_of_node, netlist)
                addtional_effi_info = simulate_one_analytics_result(analytics_info=results_tmp, file_token=file_token)
                # effi_info: '[duty cycle, C, L]': [[duty cycle, C, L], efficiency, vout]
                # TODO update the additinal effi info
                assert len(addtional_effi_info) == 1
                for k, effi_result in addtional_effi_info.items():
                    effi_info[str(current.parameters)] = effi_result
                    effi_info[str(current.parameters)]['para'] = current.parameters
    for para_str, simu_info in effi_info.items():
        if current_key + '$' + para_str not in key_sim_effi_:
            key_sim_effi_[current_key + '$' + para_str] = [simu_info['efficiency'], simu_info['Vout']]
    print(effi_info)
    # get the max among the candidate paras and return
    for para, simu_info in effi_info.items():
        tmp_reward = calculate_reward({'efficiency': simu_info['efficiency'], 'output_voltage': simu_info['Vout']},
                                      target_vout)
        if tmp_reward >= topk_max_reward:
            topk_max_reward, topk_max_para, topk_max_effi, topk_max_vout = \
                tmp_reward, simu_info['para'], simu_info['efficiency'], simu_info['Vout']
    return topk_max_reward, topk_max_effi, topk_max_vout, topk_max_para


def find_simu_max_reward_para(para_effi_info, target_vout=50):
    # return the simulation information about the topology in topk that has the highest simulation reward
    max_reward = -1
    if 'result_valid' in para_effi_info and para_effi_info['result_valid'] == False:
        return 0, {'efficiency': 0, 'Vout': -500}, -1
    for para in para_effi_info:
        effi_info = para_effi_info[para]
        effi = {'efficiency': effi_info['efficiency'], 'output_voltage': effi_info['Vout']}
        tmp_reward = calculate_reward(effi, target_vout=target_vout)
        if tmp_reward > max_reward:
            max_reward = tmp_reward
            max_effi_info = deepcopy(effi_info)
            max_para = para
    return max_reward, max_effi_info, max_para


def get_simulator_tops_sim_info(sim, filter_sim=None, thershold=None, shared_key_sim_effi_=None):
    """
    get the topks' simulation reward, return the simulation information of the topo
    that has the highest simulation reward
    """
    max_sim_reward_result = {'max_sim_para': [], 'max_sim_reward': -1, 'max_sim_effi': 0, 'max_sim_vout': -500,
                             'max_sim_state': None}
    if shared_key_sim_effi_:
        sim.key_sim_effi_ = shared_key_sim_effi_
    for i in range(len(sim.topk)):
        simulation_flag = True
        current = sim.topk[i][0]
        gnn_reward = -100
        if filter_sim is not None:
            filter_sim.current = current
            gnn_reward = filter_sim.get_reward()
            if gnn_reward < thershold:
                simulation_flag = False

        topk_max_reward, topk_max_effi, topk_max_vout, topk_max_para = \
            get_single_topo_sim_result(current=current, sweep=sim.configs_['sweep'],
                                       candidate_params=sim.candidate_params,
                                       key_sim_effi_=sim.key_sim_effi_,
                                       skip_sim=sim.configs_['skip_sim'],
                                       key_expression_mapping=sim.key_expression,
                                       target_vout=sim.configs_['target_vout'],
                                       min_vout=sim.configs_['min_vout'])
        # [state, anal_reward, anal_para, sim_reward, sim_effi, sim_vout, sim_para]
        sim.topk[i].append(topk_max_reward)
        sim.topk[i].append(topk_max_effi)
        sim.topk[i].append(topk_max_vout)
        sim.topk[i].append(topk_max_para)
        if not simulation_flag:
            sim.topk[i].append(-100)
            print('simulation is not done for topk!!!!!!!!!!! gnn:', gnn_reward,
                  ' real:', topk_max_reward)

        else:
            sim.topk[i].append(topk_max_reward)
        if max_sim_reward_result['max_sim_reward'] < topk_max_reward:
            max_sim_reward_result['max_sim_state'] = sim.topk[i][0]
            max_sim_reward_result['max_sim_reward'] = topk_max_reward
            max_sim_reward_result['max_sim_effi'] = topk_max_effi
            max_sim_reward_result['max_sim_vout'] = topk_max_vout
            max_sim_reward_result['max_sim_para'] = topk_max_para
    return max_sim_reward_result


def parallel_topk_simulation(thread_num, topks, target_vout, candidate_params):
    """

    @param thread_num: number of thread
    @param topks: the topks
    @param target_vout: target output voltage
    @return:
    """

    conn_main, conn_sub = init_pipes(tree_num=thread_num)
    topk_circuit_queue, shared_key_sim_effi_, threads = Manager().Queue(), Manager().dict(), []

    # put circuits in the queue
    for top in topks:
        current = top[0]
        for para in candidate_params:
            current.parameters = para
            print('queueing', current.get_key(), '$', para)
            topk_circuit_queue.put(current)

    for i in range(thread_num):
        t = Process(target=parallel_get_simulator_tops_sim_info,
                    args=(conn_sub[i], topk_circuit_queue, shared_key_sim_effi_))
        threads.append(t)
        t.start()

    for n in range(thread_num):
        conn_main[n].send(n)
        conn_main[n].send(target_vout)
    for n in range(thread_num):
        print(n, conn_main[n].recv())

    close_pipes(tree_num=thread_num, conn_main=conn_main, conn_sub=conn_sub)
    conn_main.clear()
    conn_sub.clear()
    return shared_key_sim_effi_


def parallel_get_simulator_tops_sim_info(conn, topk_circuit_queue, shared_key_sim_effi_):
    # create pipes
    # generate the queue
    #
    """
    get the topks' simulation reward, return the simulation information of the topo
    that has the highest simulation reward
    """
    thread_id = conn.recv()
    print(thread_id, ' start simulation')
    target_vout = conn.recv()
    while not topk_circuit_queue.empty():
        current = topk_circuit_queue.get()
        top_reward, top_effi, top_vout, top_para = \
            get_single_topo_sim_result(current=current, sweep=False,
                                       candidate_params=None,
                                       key_sim_effi_=shared_key_sim_effi_,
                                       skip_sim=False,
                                       key_expression_mapping={},
                                       target_vout=target_vout,
                                       min_vout=-500, file_token=str(thread_id))
        current_key = current.get_key()
        shared_key_sim_effi_[current_key + "$" + str(current.parameters)] = [top_effi, top_vout]
    conn.send(' finished topk simulation')
    return
