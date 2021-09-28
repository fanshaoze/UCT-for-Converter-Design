from PM_GNN.code.gen_topo_for_dateset import *


def generate_topo_info(topo_data):
    list_of_node = topo_data['list_of_node']
    list_of_edge = topo_data['list_of_edge']
    netlist = topo_data['netlist']
    # TODO has tuple problem
    # key_list = key_circuit_for_single_topo(list_of_edge, list_of_node, netlist)
    # key = key_list[0]
    key = key_circuit_from_lists(list_of_edge, list_of_node, netlist)

    data = {
        "key": key,
        "list_of_node": list_of_node,
        "list_of_edge": list_of_edge,
        "netlist": netlist
    }
    return data


if __name__ == '__main__':

    json_file = json.load(open("./components_data_random/data.json"))

    data_folder = 'component_data_random'
    target_folder = 'database'

    circuit_dic = {}

    data = {}

    # generate the key for the topo in data.json
    for fn in json_file:

        tmp = json_file

        key_list = []

        key_list = key_circuit(fn, tmp)

        for key in key_list:
            if key in circuit_dic:
                circuit_dic[key].append(fn)
            else:
                circuit_dic[key] = []
                circuit_dic[key].append(fn)

    count = 0

    with open(target_folder + '/key.json', 'w') as outfile:
        json.dump(circuit_dic, outfile)
    outfile.close()

    filename_list = []

    json_file = json.load(open("./components_data_random/data.json"))

    for key in circuit_dic:

        filename = circuit_dic[key][0]

        if filename not in filename_list:
            filename_list.append(filename)
        else:
            continue

        list_of_node = json_file[filename]['list_of_node']
        list_of_edge = json_file[filename]['list_of_edge']
        netlist = json_file[filename]['netlist']

        #            print(netlist)

        name = 'Topo-' + format(count, '04d')
        topo_file = target_folder + '/topo/' + name + '.png'

        save_topo(list_of_node, list_of_edge, topo_file)

        count = count + 1

        data[name] = {
            "key": key,
            "list_of_node": list_of_node,
            "list_of_edge": list_of_edge,
            "netlist": netlist
        }

    with open(target_folder + '/data.json', 'w') as outfile:
        json.dump(data, outfile)
    outfile.close()

    print(len(data))
