import json

from PM_GNN.code.gen_topo_for_dateset import *
from PM_GNN.code.data_for_dateset import generate_topo_info
from PM_GNN.code.circuit_for_dateset import generate_circuit_info
from PM_GNN.code.topo_lists_for_dataset import generate_data_for_topo


def generate_topo_for_GNN_model(circuit_topo):
    parameters = json.load(open("PM_GNN/code/param.json"))
    assert circuit_topo.parameters != -1
    parameters['Duty_Cycle'] = [circuit_topo.parameters]
    topo_data = generate_data_for_topo(circuit_topo)
    data_info = generate_topo_info(topo_data)
    circuit_info = generate_circuit_info(data_info)  # cki

    # with open('./database/analytic.csv', newline='') as f:
    # reader = csv.reader(f)
    # result_analytic = list(reader)

    dataset = {}

    # n_os = 100

    device_list = circuit_info["device_list"]
    num_dev = len(device_list) - 3
    param2sweep, paramname = gen_param(device_list, parameters)
    paramall = list(it.product(*(param2sweep[Name] for Name in paramname)))

    name_list = {}
    for index, name in enumerate(paramname):
        name_list[name] = index

    count = 0
    tmp_device_name = ["Duty_Cycle", "Frequency", "Vin", "Rout", "Cout", "Rin"] + device_list[-num_dev:]

    device_name = {}

    for i, item in enumerate(tmp_device_name):
        device_name[item] = i

    count = 0
    for vect in paramall:
        edge_attr = {}
        edge_attr0 = {}
        node_attr = {}
        node_attr["VIN"] = [1, 0, 0, 0]
        node_attr["VOUT"] = [0, 1, 0, 0]
        node_attr["GND"] = [0, 0, 1, 0]

        for val, key in enumerate(device_name):
            if key in ["Duty_Cycle", "Frequency", "Vin", "Rout", "Cout", "Rin"]:
                continue
            duty_cycle = vect[device_name["Duty_Cycle"]]
            if key[:2] == 'Ra':
                edge_attr['Sa' + key[2]] = [1 / float(vect[val]) * duty_cycle, 0, 0]
                edge_attr0['Sa' + key[2]] = [float(vect[val]), 0, 0, 0, 0, duty_cycle]
            elif key[:2] == 'Rb':
                edge_attr['Sb' + key[2]] = [1 / float(vect[val]) * (1 - duty_cycle), 0, 0]
                edge_attr0['Sb' + key[2]] = [0, float(vect[val]), 0, 0, 0, duty_cycle]
            elif key[0] == 'C':
                edge_attr[key] = [0, float(vect[val]), 0]
                edge_attr0[key] = [0, 0, vect[val], 0, 0, 0]
            elif key[0] == 'L':
                edge_attr[key] = [0, 0, 1 / float(vect[val])]
                edge_attr0[key] = [0, 0, 0, vect[val], 0, 0]
            else:
                edge_attr[key] = [0, 0, 0, 0, 0, 0]

        for item in data_info["list_of_node"]:
            if str(item).isnumeric():
                node_attr[str(item)] = [0, 0, 0, 1]
        dataset[str(count)] = {"list_of_edge": data_info["list_of_edge"],
                               "list_of_node": data_info["list_of_node"],
                               "netlist": data_info["netlist"],
                               "edge_attr": edge_attr,
                               "edge_attr0": edge_attr0,
                               "node_attr": node_attr,
                               "duty_cycle": vect[device_name["Duty_Cycle"]],
                               "rout": vect[device_name["Rout"]],
                               "cout": vect[device_name["Cout"]],
                               "freq": vect[device_name["Frequency"]]
                               }
        count = count + 1
    print('dataset: ', dataset)

    return dataset


if __name__ == '__main__':

    cki = json.load(open("database/circuit.json"))
    parameters = json.load(open("param.json"))

    data = json.load(open("database/data.json"))
    result = json.load(open("database/sim.json"))

    with open('./database/analytic.csv', newline='') as f:
        reader = csv.reader(f)
        result_analytic = list(reader)

    dataset = {}

    n_os = 100

    for fn in cki:

        device_list = cki[fn]["device_list"]
        num_dev = len(device_list) - 3
        param2sweep, paramname = gen_param(device_list, parameters)
        paramall = list(it.product(*(param2sweep[Name] for Name in paramname)))

        name_list = {}
        for index, name in enumerate(paramname):
            name_list[name] = index

        count = 0
        tmp_device_name = ["Duty_Cycle", "Frequency", "Vin", "Rout", "Cout", "Rin"] + device_list[-num_dev:]

        device_name = {}

        for i, item in enumerate(tmp_device_name):
            device_name[item] = i

        count = 0

        tmp_list_analytic = []

        for vect in paramall:
            edge_attr = {}
            edge_attr0 = {}
            node_attr = {}
            node_attr["VIN"] = [1, 0, 0, 0]
            node_attr["VOUT"] = [0, 1, 0, 0]
            node_attr["GND"] = [0, 0, 1, 0]

            for val, key in enumerate(device_name):
                if key in ["Duty_Cycle", "Frequency", "Vin", "Rout", "Cout", "Rin"]:
                    continue
                duty_cycle = vect[device_name["Duty_Cycle"]]
                if key[:2] == 'Ra':
                    edge_attr['Sa' + key[2]] = [1 / float(vect[val]) * duty_cycle, 0, 0]
                    edge_attr0['Sa' + key[2]] = [float(vect[val]), 0, 0, 0, 0, duty_cycle]
                elif key[:2] == 'Rb':
                    edge_attr['Sb' + key[2]] = [1 / float(vect[val]) * (1 - duty_cycle), 0, 0]
                    edge_attr0['Sb' + key[2]] = [0, float(vect[val]), 0, 0, 0, duty_cycle]
                elif key[0] == 'C':
                    edge_attr[key] = [0, float(vect[val]), 0]
                    edge_attr0[key] = [0, 0, vect[val], 0, 0, 0]
                elif key[0] == 'L':
                    edge_attr[key] = [0, 0, 1 / float(vect[val])]
                    edge_attr0[key] = [0, 0, 0, vect[val], 0, 0]
                else:
                    edge_attr[key] = [0, 0, 0, 0, 0, 0]

            for item in data[fn]["list_of_node"]:
                if str(item).isnumeric():
                    node_attr[str(item)] = [0, 0, 0, 1]

            vout = result[fn][str(vect)]["vout"]
            eff = result[fn][str(vect)]["eff"]

            flag = 0
            for item in result_analytic:

                if fn == item[0] and str(vect):
                    flag = 1
                    vout_analytic = item[2]
                    eff_analytic = item[3]

            if flag == 0:
                vout_analytic = 0
                eff_analytic = 0

            dataset[fn + '-' + str(count)] = {"list_of_edge": data[fn]["list_of_edge"],
                                              "list_of_node": data[fn]["list_of_node"],
                                              "netlist": data[fn]["netlist"],
                                              "edge_attr": edge_attr,
                                              "edge_attr0": edge_attr0,
                                              "node_attr": node_attr,
                                              "vout": vout,
                                              "eff": eff,
                                              "vout_analytic": vout_analytic,
                                              "eff_analytic": eff_analytic,
                                              "duty_cycle": vect[device_name["Duty_Cycle"]],
                                              "rout": vect[device_name["Rout"]],
                                              "cout": vect[device_name["Cout"]],
                                              "freq": vect[device_name["Frequency"]]

                                              }

            count = count + 1

    with open('./database/dataset.json', 'w') as f:
        json.dump(dataset, f)
    f.close()
