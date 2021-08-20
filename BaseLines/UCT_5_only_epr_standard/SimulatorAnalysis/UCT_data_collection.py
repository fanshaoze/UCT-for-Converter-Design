import json
import pandas as pd
import csv
from SimulatorAnalysis.gen_topo import *
from SimulatorAnalysis.random_topo import generate_one_data_json
from SimulatorAnalysis.data import generate_key_data
from SimulatorAnalysis.circuit_val import get_one_circuit
from SimulatorAnalysis.expr_val import get_one_expression
from SimulatorAnalysis.analytics import get_analytics_result, get_analytics_information


def key_expression_dict(target_vout_min=-500, target_vout_max=500):
    data_json_file = json.load(open("./SimulatorAnalysis/database/data.json"))
    # data_json_file = json.load(open("./database/data.json"))
    # print(len(data_json_file))
    expression_json_file = json.load(open("./SimulatorAnalysis/database/expression.json"))
    # expression_json_file = json.load(open("./database/expression.json"))

    # print("length of expression_json_file:", len(expression_json_file))
    simu_results = []
    with open("./SimulatorAnalysis/database/analytic.csv", "r") as csv_file:
        # with open("./database/analytic.csv", "r") as csv_file:

        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            simu_results.append(row)

    print(len(simu_results))
    print(len(data_json_file))
    print(len(expression_json_file))

    data = {}
    key_expression = {}
    x = 0
    for data_fn in data_json_file:
        # print(x)
        if data_fn in expression_json_file:
            expression = expression_json_file[data_fn]
        else:
            expression = "Invalid"
            duty_cycle_para = "None"
            key_expression[data_json_file[data_fn]['key'] + '$' + duty_cycle_para] = {}
            key_expression[data_json_file[data_fn]['key'] + '$' + duty_cycle_para]['Expression'] = expression
            key_expression[data_json_file[data_fn]['key'] + '$' + duty_cycle_para]['Efficiency'] = 0
            key_expression[data_json_file[data_fn]['key'] + '$' + duty_cycle_para]['Vout'] = target_vout_min

        for i in range(len(simu_results)):
            if simu_results[i][0] == data_fn:

                while i < (len(simu_results)) and simu_results[i][0] == data_fn:
                    duty_cycle_para = float(simu_results[i][1])
                    # print(duty_cycle_para)
                    assert not (data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para) in key_expression)

                    # print(data_json_file[data_fn]['key'] + '$' + paras_str)
                    key_expression[data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para)] = {}
                    key_expression[data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para)][
                        'Expression'] = expression
                    if simu_results[i][3] != 'False' and simu_results[i][4] != 'False':
                        # effi = {'efficiency': float(int(simu_results[i][4]) / 100),
                        #         'output_voltage': float(simu_results[i][3])}
                        # tmp_reward = calculate_reward(effi, 50, -500, 500)
                        # if tmp_reward > 0:
                        #     print(tmp_reward,data_fn)
                        key_expression[data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para)]['Efficiency'] = \
                            simu_results[i][4]
                        key_expression[data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para)]['Vout'] = \
                            simu_results[i][3]
                    else:
                        key_expression[data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para)]['Efficiency'] = 0
                        key_expression[data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para)]['Vout'] = \
                            target_vout_min
                    i += 1
                break
        x += 1
    # with open('./database/analytic-expression.json', 'w') as f:
    with open('./SimulatorAnalysis/database/analytic-expression.json', 'w') as f:
        json.dump(key_expression, f)
    f.close()

    return key_expression


def save_one_analytics_result(key_expression):
    pass


def generate_para_strs(device_list):
    para_strs = []
    parameters = json.load(open("./SimulatorAnalysis/param.json"))
    param2sweep, paramname = gen_param(device_list, parameters)
    # print(param2sweep)
    # print(paramname)
    paramall = list(it.product(*(param2sweep[Name] for Name in paramname)))
    # print(paramall)
    name_list = {}
    for index, name in enumerate(paramname):
        name_list[name] = index

    for vect in paramall:
        para_strs.append(str(vect))

    return para_strs


def find_one_analytics_result(key, key_expression, graph, comp2port_mapping, port2comp_mapping, idx_2_port, port_2_idx,
                              parent,
                              component_pool, same_device_mapping, port_pool):
    """
    We use the key information to get one analytics' result
    :param key: uniquely representing a topology
    :param key_expression: key expression mapping
    :param graph:
    :param comp2port_mapping:
    :param port2comp_mapping:
    :param idx_2_port:
    :param port_2_idx:
    :param parent:
    :param component_pool:
    :param same_device_mapping:
    :param port_pool:
    :return: analytics result
    """
    para_result = {}
    if key + 'Invalid' in key_expression:
        return {'None': [key_expression[key + '$' + 'Invalid']['Effiency'],
                         key_expression[key + '$' + 'Invalid']['Vout']]}
    else:
        find_flag = 0
        device_list = get_one_device_list(graph, comp2port_mapping, port2comp_mapping, idx_2_port, port_2_idx, parent,
                                          component_pool, same_device_mapping, port_pool)
        if not device_list:
            return None
        paras_str = generate_para_strs(device_list)

        for para_str in paras_str:
            paras = para_str[1:-2].split(',')
            duty_cycle_para = float(paras[0])
            # duty_cycle_para = float(simu_results[i][1])

            if key + '$' + str(duty_cycle_para) in key_expression:
                find_flag = 1
                para_result[str(duty_cycle_para)] = [key_expression[key + '$' + str(duty_cycle_para)]['Efficiency'],
                                                     key_expression[key + '$' + str(duty_cycle_para)]['Vout']]

    if find_flag == 1:
        return para_result
    return None


def get_one_device_list(graph, comp2port_mapping, port2comp_mapping, idx_2_port, port_2_idx, parent, component_pool,
                        same_device_mapping, port_pool):
    name = "PCC-device-list"
    directory_path = './SimulatorAnalysis/components_data_random'
    target_folder = './SimulatorAnalysis/database'
    return_info_data_json = generate_one_data_json(name, directory_path, graph, comp2port_mapping, port2comp_mapping,
                                                   idx_2_port, port_2_idx, parent,
                                                   component_pool, same_device_mapping, port_pool)
    if return_info_data_json == "Has prohibit path" or \
            return_info_data_json == "Not has switch" or \
            return_info_data_json == "Has redundant loop":
        return None
    key = generate_key_data(directory_path, name, target_folder)
    circuit = get_one_circuit(target_folder, name)
    return circuit['device_list']


def get_analytics_file(key_expression, graph, comp2port_mapping, port2comp_mapping, idx_2_port, port_2_idx,
                       parent, component_pool,
                       same_device_mapping, port_pool, fix_paras, target_vout_min=-500):
    para_result = {}
    pid_label = str(os.getpid())
    name = "PCC-" + pid_label
    directory_path = './SimulatorAnalysis/components_data_random'
    target_folder = './SimulatorAnalysis/database'
    return_info_data_json = generate_one_data_json(name, directory_path, graph, comp2port_mapping, port2comp_mapping,
                                                   idx_2_port, port_2_idx, parent,
                                                   component_pool, same_device_mapping, port_pool)
    if return_info_data_json == "Has prohibit path" or \
            return_info_data_json == "Not has switch" or \
            return_info_data_json == "Has redundant loop":
        return None
    key = generate_key_data(directory_path, name, target_folder)
    circuit = get_one_circuit(target_folder, name)
    expression = get_one_expression(target_folder, name)
    simu_results = None
    if expression == "invalid":
        duty_cycle_para = "None"
        key_expression[key + '$' + str(duty_cycle_para)] = {}
        key_expression[key + '$' + str(duty_cycle_para)]['Expression'] = expression
        key_expression[key + '$' + str(duty_cycle_para)]['Efficiency'] = 0
        key_expression[key + '$' + str(duty_cycle_para)]['Vout'] = target_vout_min

    else:
        simu_results = get_analytics_information(target_folder, name, fix_paras)
    print(simu_results)
    return simu_results


def simulate_one_analytics_result(analytics_info):
    if analytics_info == None:
        return {'result_valid': False, 'efficiency': 0, 'Vout': 0}
    analytic = analytics_info
    cki_folder = 'database/cki/'
    for fn in analytic:
        print(int(fn[6:9]))
        count = 'tmp_simu'
        for param in analytic[fn]:
            param_value = [float(value) for value in param[1:-1].split(', ')]
            device_name = analytic[fn][param][0]
            netlist = analytic[fn][param][1]
            file_name = fn + '-' + str(count) + '.cki'
            convert_cki_full_path(file_name, param_value, device_name, netlist)
            print(file_name)
            simulate(file_name)
    data = {}
    data_csv = []
    for fn in analytic:
        for param in analytic[fn]:
            param_value = [float(value) for value in param[1:-1].split(', ')]
            device_name = analytic[fn][param][0]
            netlist = analytic[fn][param][1]
            file_name = fn + '-' + str(count) + '.simu'
            path = file_name
            vin = param_value[device_name['Vin']]
            freq = param_value[device_name['Frequency']] * 1000000
            rin = param_value[device_name['Rin']]
            rout = param_value[device_name['Rout']]
            print(file_name)
            result = calculate_efficiency(path, vin, freq, rin, rout)
            print(result)
            return result


def get_one_analytics_result(key_expression, graph, comp2port_mapping, port2comp_mapping, idx_2_port, port_2_idx,
                             parent, component_pool,
                             same_device_mapping, port_pool, target_vout_min=-500):
    para_result = {}
    pid_label = str(os.getpid())
    name = "PCC-" + pid_label
    directory_path = './SimulatorAnalysis/components_data_random'
    target_folder = './SimulatorAnalysis/database'
    return_info_data_json = generate_one_data_json(name, directory_path, graph, comp2port_mapping, port2comp_mapping,
                                                   idx_2_port, port_2_idx, parent,
                                                   component_pool, same_device_mapping, port_pool)
    if return_info_data_json == "Has prohibit path" or \
            return_info_data_json == "Not has switch" or \
            return_info_data_json == "Has redundant loop":
        return None
    key = generate_key_data(directory_path, name, target_folder)
    circuit = get_one_circuit(target_folder, name)
    expression = get_one_expression(target_folder, name)
    if expression == "invalid":
        duty_cycle_para = "None"
        key_expression[key + '$' + str(duty_cycle_para)] = {}
        key_expression[key + '$' + str(duty_cycle_para)]['Expression'] = expression
        key_expression[key + '$' + str(duty_cycle_para)]['Efficiency'] = 0
        key_expression[key + '$' + str(duty_cycle_para)]['Vout'] = target_vout_min

    else:
        simu_results = get_analytics_result(target_folder, name)
        # print(simu_results)
        for i in range(len(simu_results)):
            # print("simu_results[i]: ", simu_results[i])
            # para_str = simu_results[i][1]
            # paras = para_str[1:-2].split(',')
            duty_cycle_para = float(simu_results[i][1])
            str(duty_cycle_para)
            key_expression[key + '$' + str(duty_cycle_para)] = {}
            key_expression[key + '$' + str(duty_cycle_para)]['Expression'] = expression
            if simu_results[i][3] != 'False' and simu_results[i][4] != 'False':
                key_expression[key + '$' + str(duty_cycle_para)]['Efficiency'] = simu_results[i][4]
                key_expression[key + '$' + str(duty_cycle_para)]['Vout'] = simu_results[i][3]
            else:
                key_expression[key + '$' + str(duty_cycle_para)]['Efficiency'] = 0
                key_expression[key + '$' + str(duty_cycle_para)]['Vout'] = target_vout_min

            para_result[str(duty_cycle_para)] = [key_expression[key + '$' + str(duty_cycle_para)]['Efficiency'],
                                                 key_expression[key + '$' + str(duty_cycle_para)]['Vout']]

        return para_result


def read_analytics_result():
    data_json_file = json.load(open("./SimulatorAnalysis/database/analytic-expression.json"))

    return data_json_file


def read_sim_result():
    data_json_file = json.load(open("./SimulatorAnalysis/database/key_sim_result.json"))
    return data_json_file


def save_analytics_result(simu_results):
    with open('./SimulatorAnalysis/database/analytic-expression.json', 'w') as f:
        json.dump(simu_results, f)
    f.close()


def save_sim_result(simu_results):
    with open('./SimulatorAnalysis/database/key_sim_result.json', 'w') as f:
        json.dump(simu_results, f)
    f.close()


def merge_and_save_analytics_result(files):
    merge_simu_results = {}
    for file_name in files:
        data_json_file = json.load(open(file_name))
        merge_simu_results.update(data_json_file)
    with open('analytic-expression.json', 'w') as f:
        json.dump(merge_simu_results, f)
    f.close()
    return


if __name__ == '__main__':
    key_expression_dict()
