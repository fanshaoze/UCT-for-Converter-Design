import json
import pandas as pd
import csv
from SimulatorAnalysis.gen_topo import *
from SimulatorAnalysis.random_topo import generate_one_data_json
from SimulatorAnalysis.data import generate_key_data
from SimulatorAnalysis.circuit import get_one_circuit
from SimulatorAnalysis.expression import get_one_expression
from SimulatorAnalysis.analytics import get_analytics_result


def key_expression_dict(target_vout_min=-500, target_vout_max=500):
    data_json_file = json.load(open("./SimulatorAnalysis/database/data.json"))
    print(len(data_json_file))
    expression_json_file = json.load(open("./SimulatorAnalysis/database/expression.json"))
    print(len(expression_json_file))
    simu_results = []
    with open("./SimulatorAnalysis/database/analytic.csv", "r") as csv_file:
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
        if x == 41:
            print("find")
            print(data_json_file[data_fn]['key'])
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
                    paras_str = simu_results[i][1]
                    paras = paras_str[1:-2].split(',')
                    duty_cycle_para = float(paras[0])
                    # print(duty_cycle_para)
                    assert not (data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para) in key_expression)
                    if simu_results[i][-1] == 'False':
                        key_expression[data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para)] = {}
                        key_expression[data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para)][
                            'Expression'] = expression
                        key_expression[data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para)]['Efficiency'] = 0
                        key_expression[data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para)][
                            'Vout'] = target_vout_min
                    else:
                        # print(data_json_file[data_fn]['key'] + '$' + paras_str)
                        key_expression[data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para)] = {}
                        key_expression[data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para)][
                            'Expression'] = expression
                        key_expression[data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para)]['Efficiency'] = \
                            simu_results[i][3]
                        key_expression[data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para)]['Vout'] = \
                            simu_results[i][2]
                    i += 1
                break
        x += 1
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
    if return_info_data_json == "Has prohibit path" or return_info_data_json == "Not has switch":
        return None
    key = generate_key_data(directory_path, name, target_folder)
    circuit = get_one_circuit(target_folder, name)
    return circuit['device_list']


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
    if return_info_data_json == "Has prohibit path" or return_info_data_json == "Has switch":
        return None
    key = generate_key_data(directory_path, name, target_folder)
    circuit = get_one_circuit(target_folder, name)
    expression = get_one_expression(target_folder, name)
    if expression == "invalid":
        duty_cycle_para = "None"
        key_expression[key + '$' + str(duty_cycle_para)]['Expression'] = expression
        key_expression[key + '$' + str(duty_cycle_para)]['Efficiency'] = 0
        key_expression[key + '$' + str(duty_cycle_para)]['Vout'] = target_vout_min

    else:
        simu_results = get_analytics_result(target_folder, name)
        # print(simu_results)
        for i in range(len(simu_results)):
            para_str = simu_results[i][1]
            paras = para_str[1:-2].split(',')
            duty_cycle_para = float(paras[0])
            str(duty_cycle_para)

            if simu_results[i][-1] == 'False':
                key_expression[key + '$' + str(duty_cycle_para)] = {}
                key_expression[key + '$' + str(duty_cycle_para)]['Expression'] = expression
                key_expression[key + '$' + str(duty_cycle_para)]['Efficiency'] = 0
                key_expression[key + '$' + str(duty_cycle_para)]['Vout'] = target_vout_min
            else:
                key_expression[key + '$' + str(duty_cycle_para)] = {}
                key_expression[key + '$' + str(duty_cycle_para)]['Expression'] = expression
                key_expression[key + '$' + str(duty_cycle_para)]['Efficiency'] = simu_results[i][3]
                key_expression[key + '$' + str(duty_cycle_para)]['Vout'] = simu_results[i][2]
            para_result[str(duty_cycle_para)] = [key_expression[key + '$' + str(duty_cycle_para)]['Efficiency'],
                                                 key_expression[key + '$' + str(duty_cycle_para)]['Vout']]

        return para_result


def read_analytics_result():
    data_json_file = json.load(open("./SimulatorAnalysis/database/analytic-expression.json"))
    return data_json_file


def save_analytics_result(simu_results):
    with open('./SimulatorAnalysis/database/analytic-expression.json', 'w') as f:
        json.dump(simu_results, f)
    f.close()


if __name__ == '__main__':
    key_expression_dict()
