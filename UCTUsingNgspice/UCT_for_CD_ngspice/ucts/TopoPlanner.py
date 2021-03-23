import random
from ucts import uct
from utils.util import hash_str_encoding, save_one_in_hash_file, Merge, sample_component
from utils import gen_topo
from utils.eliminate_isomorphism import instance_to_isom_str, maintain_reward_hash_with_edge, \
    init_candidate_parameters, get_component_priorities
import collections
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import product
from SimulatorSetPara.build_topology import nets_to_ngspice_files
from SimulatorSetPara.simulation import simulate_topologies
from SimulatorSetPara.simulation_analysis import analysis_topologies

import datetime
import warnings


def union(x, y, parent):
    f_x = find(x, parent)
    f_y = find(y, parent)
    if f_x == f_y:
        return False
    parent[f_x] = f_y
    return True


def find(x, parent):
    if parent[x] != x:
        parent[x] = find(parent[x], parent)
    return parent[x]


def find_connected_set(x, parent):
    net_list = []
    for i in range(len(parent)):
        if already_connected(x, i, parent):
            net_list.append(i)
    return net_list


def already_connected(x, y, parent):
    return find(x, parent) == find(y, parent)


def get_component_type(component):
    if component.startswith('inductor'):
        ret = 'inductor'
    elif component.startswith('capacitor'):
        ret = 'capacitor'
    elif component.startswith('FET-A'):
        ret = 'FET-A'
    elif component.startswith('FET-B'):
        ret = 'FET-B'
    else:
        ret = component
    return ret


def sort_dict_string(graph_dict):
    """
    Change the graph dict to an string, with sorted keys and sorted set value
    :graph_dict: the dict representing the graph
    """
    graph_dict_str = ""
    keys = graph_dict.keys()
    keys = list(keys)
    keys.sort()
    for key in keys:
        act_list = list(graph_dict[key])
        act_list.sort()
        graph_dict_str += str(key) + ":" + str(act_list) + ","
    return graph_dict_str


def convert_graph_without_parents(graph):
    tmp_graph = deepcopy(graph)
    list_of_edge = set()

    for node in tmp_graph.keys():
        nei_set = tmp_graph[node]
        length_of_nei = len(nei_set)
        for _ in range(length_of_nei):
            list_of_edge.add((node, nei_set.pop()))

    return list(list_of_edge)


def graph_has_short_cut(graph, parent, same_device_mapping):
    for node in graph:
        root_node = find(node, parent)
        if node in same_device_mapping:
            cur_node_the_other_port = same_device_mapping[node]
            cur_node_the_other_port_root = find(cur_node_the_other_port, parent)
            if cur_node_the_other_port_root == root_node:
                return True
    return False


def graph_connect_roots(parent):
    gnd_root = find(0, parent)
    vin_root = find(1, parent)
    vout_root = find(2, parent)
    if gnd_root == vin_root or vin_root == vout_root or vout_root == gnd_root:
        return True
    return False


def convert_graph(graph, comp2port_mapping, parent, same_device_mapping, port_pool):
    list_of_node = set()
    list_of_edge = set()
    has_short_cut = False

    for node in comp2port_mapping:
        if len(comp2port_mapping[node]) == 2:
            list_of_node.add(port_pool[comp2port_mapping[node][0]])
            list_of_node.add(port_pool[comp2port_mapping[node][1]])
            list_of_edge.add((port_pool[comp2port_mapping[node][1]], port_pool[comp2port_mapping[node][0]]))

    for node in graph:
        root_node = find(node, parent)
        list_of_node.add(port_pool[node])
        list_of_node.add(port_pool[root_node])
        # TODO only for one device's short cut, but global case may not be achieved
        if node in same_device_mapping:
            cur_node_the_other_port = same_device_mapping[node]
            cur_node_the_other_port_root = find(cur_node_the_other_port, parent)
            if cur_node_the_other_port_root == root_node:
                has_short_cut = True

        if root_node != node:
            list_of_edge.add((port_pool[node], port_pool[root_node]))

        for nei in graph[node]:
            list_of_node.add(port_pool[nei])
            if nei != root_node:
                list_of_edge.add((port_pool[nei], port_pool[root_node]))

    return list(list_of_node), list(list_of_edge), has_short_cut


def convert_to_netlist(component_pool, port_pool, parent, comp2port_mapping):
    # for one component, find the two port
    # if one port is GND/VIN/VOUT, leave it don't need to find the root
    # if one port is normal port, then find the root, if the port equal to root then leave it,
    # if the port is not same as root, change the port to root
    list_of_node = set()
    list_of_edge = set()

    for idx, comp in enumerate(component_pool):
        list_of_node.add(comp)
        for port in comp2port_mapping[idx]:
            port_root = find(port, parent)
            if port_root in [0, 1, 2]:
                list_of_node.add(port_pool[port_root])
                list_of_node.add(port_root)
                list_of_edge.add((comp, port_root))
                list_of_edge.add((port_pool[port_root], port_root))
            else:
                list_of_node.add(port_root)
                list_of_edge.add((comp, port_root))

    return list(list_of_node), list(list_of_edge)


def calculate_reward(effi):
    if effi['output_voltage'] <= 0 or effi['output_voltage'] >= 50:
        return 0
    elif effi['output_voltage'] <= 0.25 * 50:
        # print(effi['efficiency'] * 0.01 * (1 - (0.25*50 - effi['output_voltage']) / (0.25*50)))
        return effi['efficiency'] * 0.01 * (1 - (0.25 * 50 - effi['output_voltage']) / (0.25 * 50))
    elif effi['output_voltage'] >= 0.75 * 50:
        # print(effi['efficiency'] * 0.01 * (1 - (effi['output_voltage'] - 0.75*50) / (0.25*50)))
        return effi['efficiency'] * 0.01 * (1 - (effi['output_voltage'] - 0.75 * 50) / (0.25 * 50))
    elif effi['output_voltage'] >= 25:
        # print(effi['efficiency'] * (1 - (1 - 0.01) * (effi['output_voltage'] - 25) / (0.25*50)))
        return effi['efficiency'] * (1 - (1 - 0.01) * (effi['output_voltage'] - 25) / (0.25 * 50))
    elif effi['output_voltage'] < 25:
        # print(effi['efficiency'] * (1 - (1 - 0.01) * (25 - effi['output_voltage']) / (0.25*50)))
        return effi['efficiency'] * (1 - (1 - 0.01) * (25 - effi['output_voltage']) / (0.25 * 50))
    return 0


def remove_roots(allowed_root_pair, current_root_pair):
    for root_pair in allowed_root_pair:
        if (current_root_pair[0] in root_pair) and (current_root_pair[1] in root_pair):
            allowed_root_pair.remove(root_pair)
            break
    return allowed_root_pair


def find_roots(allowed_root_pair, current_root_pair):
    for root_pair in allowed_root_pair:
        if (current_root_pair[0] in root_pair) and (current_root_pair[1] in root_pair):
            return True
    return False


#
class TopoGenState(uct.State):
    def __init__(self, init=False):
        if init:
            self.num_component = 0
            # self.component_pool = ['VIN', 'VOUT', "GND"]
            # self.port_pool = ['VIN', 'VOUT', "GND"]
            self.component_pool = ['GND', 'VIN', "VOUT"]
            self.port_pool = ['GND', 'VIN', "VOUT"]
            self.count_map = {"FET-A": 0, "FET-B": 0, "capacitor": 0, "inductor": 0}
            self.comp2port_mapping = {0: [0], 1: [1],
                                      2: [2]}  # key is the idx in component pool, value is idx in port pool
            self.port2comp_mapping = {0: 0, 1: 1, 2: 2}

            # self.port_2_idx = {'VIN': 0, 'VOUT': 1, "GND": 2}
            # self.idx_2_port = {0: 'VIN', 1: 'VOUT', 2: "GND"}
            self.port_2_idx = {'GND': 0, 'VIN': 1, "VOUT": 2}
            self.idx_2_port = {0: 'GND', 1: 'VIN', 2: "VOUT"}
            self.same_device_mapping = {}
            self.graph = collections.defaultdict(set)
            self.parent = None
            self.step = 0
            self.parameters = {"VIN": [1, 0, 0, 0, 0, 0, 0, 0],
                               "VOUT": [0, 1, 0, 0, 0, 0, 0, 0],
                               "GND": [0, 0, 1, 0, 0, 0, 0, 0],
                               "FET-A-0": [0, 0, 0, 0, 0, 0.1, 0, 0],
                               "FET-B-0": [0, 0, 0, 0, 0, 0, 0.25, 0],
                               "inductor-0-left": [0, 0, 0, 0, 0, 0, 0, 1],
                               "inductor-0-right": [0, 0, 0, 0, 0, 0, 0, 1],
                               "FET-A-0-left": [0, 0, 0, 0, 0, 0, 0, 1],
                               "FET-A-0-right": [0, 0, 0, 0, 0, 0, 0, 1],
                               "FET-B-0-left": [0, 0, 0, 0, 0, 0, 0, 1],
                               "FET-B-0-right": [0, 0, 0, 0, 0, 0, 0, 1],
                               "inductor-0": [0, 0, 0, 0, 0.75, 0, 0, 0]}
            self.candidate_para = init_candidate_parameters()

    # self.act_vect = []

    def init_disjoint_set(self):
        """
        The topology also use a union-find set to store the merged points, this is the init
        of union-find set printing the set and having a look(together with the graph dict)
        would be helpful
        """
        self.parent = list(range(len(self.port_pool)))

    def equal(self, state):
        if isinstance(state, TopoGenState):
            return self.component_pool == state.component_pool and \
                   self.port_pool == state.port_pool and \
                   self.num_component == state.num_component and \
                   self.count_map == state.count_map and \
                   self.comp2port_mapping == state.comp2port_mapping and \
                   self.port2comp_mapping == state.port2comp_mapping and \
                   self.port_2_idx == state.port_2_idx and \
                   self.idx_2_port == state.idx_2_port and \
                   self.same_device_mapping == state.same_device_mapping and \
                   self.graph == state.graph and \
                   self.step == state.step and \
                   self.parent == state.parent
        return False

    def get_edges(self):
        edges = []
        for key, vals in self.graph.items():
            for v in vals:
                edges.append((self.idx_2_port[key], self.idx_2_port[v]))
        return edges

    def duplicate(self):
        return deepcopy(self)

    def print(self):
        print('component_pool: {} \nport_pool: {}\nstep: {}'.format(self.component_pool, self.port_pool, self.step))

    def get_node_num(self):
        return len(self.component_pool) - 3

    def get_edge_num(self):
        edge_num = 0
        for key, val in self.graph.items():
            edge_num += len(val)
        return edge_num / 2

    def graph_is_connected(self):
        if self.graph:
            list_of_node, list_of_edge, has_short_cut = convert_graph(self.graph, self.comp2port_mapping, self.parent,
                                                                      self.same_device_mapping, self.port_pool)
            G = nx.Graph()
            G.add_nodes_from(list_of_node)
            G.add_edges_from(list_of_edge)
            return nx.is_connected(G)
        else:
            return False

    def has_all_ports(self):
        for i in range(3, len(self.port_pool)):
            if i not in self.graph:
                return False
        return True

    def has_in_out_gnd(self):
        return (1 in self.graph) and (2 in self.graph) and (0 in self.graph)

    def graph_is_valid(self):
        if self.graph:
            list_of_node, list_of_edge, has_short_cut = convert_graph(self.graph, self.comp2port_mapping, self.parent,
                                                                      self.same_device_mapping, self.port_pool)
            G = nx.Graph()
            G.add_nodes_from(list_of_node)
            G.add_edges_from(list_of_edge)
            return (not has_short_cut) and self.has_in_out_gnd() and self.has_all_ports() and nx.is_connected(G)
        else:
            return False

    def get_idx_graph(self):
        if self.graph:
            list_of_node = list(range(len(self.port_pool)))
            list_of_edge = convert_graph_without_parents(self.graph)
            G = nx.Graph()
            G.add_nodes_from(list_of_node)
            G.add_edges_from(list_of_edge)
            return G
        else:
            return None

    def has_short_cut(self):
        for node in self.graph:
            root_node = find(node, self.parent)

            # cur_node_the_other_port_root = find(cur_node_the_other_port, parent)

            if node in self.same_device_mapping:
                cur_node_the_other_port = self.same_device_mapping[node]
                cur_node_the_other_port_root = find(cur_node_the_other_port, self.parent)
                if cur_node_the_other_port_root == root_node:
                    return True
        return False

    def visualize(self, title=None, figure_folder=None):
        list_of_node, list_of_edge = convert_to_netlist(self.component_pool, self.port_pool, self.parent,
                                                        self.comp2port_mapping)
        T = nx.Graph()
        T.add_nodes_from(list_of_node)
        T.add_edges_from(list_of_edge)
        if bool(title):
            print('title', title)
            plt.title(title)
        nx.draw(T, with_labels=True)

        # plt.show()
        dt = datetime.datetime.now().strftime(figure_folder + '%Y-%m-%d-%H-%M-%S')
        # dt = datetime.datetime.now().strftime(figure_folder + title)
        plt.savefig(dt)
        plt.close()


class TopoGenAction(uct.SimAction):
    def __init__(self, action_type, value):
        self.type = action_type
        self.value = value

    def duplicate(self):
        other = TopoGenAction(self.type, self.value)
        return other

    def print(self):
        print(' ({}, {})'.format(self.type, self.value))

    def equal(self, other):
        if isinstance(other, TopoGenAction):
            return other.type == self.type and other.value == self.value
        return False

    def equivalent(self, other):
        if isinstance(other, TopoGenAction):
            if other.type != self.type:
                return False
            else:
                if other.type == 'node':
                    return other.value == self.value
                elif other.type == 'edge':
                    return (other.value == self.value) or (other.value == [self.value[1], self.value[0]])
            return other.type == self.type and other.value == self.value
        return False


def sample_from_paths(paths_with_probability):
    sample_prob = random.random()
    sum_prob = 0
    for path_idx, edge_and_prob in paths_with_probability.items():
        if sum_prob < sample_prob <= sum_prob + edge_and_prob[1]:
            return edge_and_prob[0]
        else:
            sum_prob += edge_and_prob[1]
    return None


def port_list_multiply(port_lists_0, port_lists_1):
    merge_port_list = []
    for port_list_0 in port_lists_0:
        for port_list_1 in port_lists_1:
            tmp_list_0 = deepcopy(port_list_0)
            tmp_list_0.extend(port_list_1)
            merge_port_list.append(tmp_list_0)
    return merge_port_list


def path_add_edges(port_lists_0, port_lists_1):
    merge_port_list = []
    for port_list_0 in port_lists_0:
        for port_list_1 in port_lists_1:
            tmp_list_0 = deepcopy(port_list_0)
            tmp_list_0.append(port_list_1)
            merge_port_list.append(tmp_list_0)
    return merge_port_list


def port_multiply(port_list_0, port_list_1):
    edge_list = []
    for port_0 in port_list_0:
        for port_1 in port_list_1:
            edge = [port_0, port_1] if port_0 < port_1 else [port_1, port_0]
            edge_list.append(edge)
    return edge_list


def edge_list_multiply(candidate_edge_list_0, candidate_edge_list_1):
    merge_edge_list = []
    for candidate_edge_0 in candidate_edge_list_0:
        for candidate_edge_1 in candidate_edge_list_1:
            tmp_list_0 = deepcopy(candidate_edge_0)
            tmp_list_0.extend(candidate_edge_1)
            merge_edge_list.append(tmp_list_0)
    return merge_edge_list


def find_action(action, action_set):
    if isinstance(action, TopoGenAction):
        for i in range(len(action_set)):
            tmp_action = action_set[i]
            if action.type == tmp_action.type and action.value == tmp_action.value:
                return True
        return False
    return None


def remove_one_action(action, action_set):
    if isinstance(action, TopoGenAction):
        for i in range(action_set):
            tmp_action = action_set(i)
            if action.type == tmp_action.type and action.value == tmp_action.value:
                return True
        return False
    return None


def generate_port_path_add(path_adding_cost):
    """
    find the min length path to add and the edge start with current port when adding this path
    :param path_adding_cost: [(path, edge)]
    :return:
    """
    candidate_edge_list = []
    max_add_length = -1
    for path_edge in path_adding_cost:
        if len(path_edge[0]) > max_add_length:
            candidate_edge_list.clear()
            candidate_edge_list.append(path_edge[1])
        elif len(path_edge[0]) == max_add_length:
            candidate_edge_list.append(path_edge[1])
    return random.choice(candidate_edge_list)


class TopoGenSimulator(uct.Simulator):
    def __init__(self, _configs, _approved_path_freq, _component_condition_prob, _num_component=4, target=None):
        self.necessary_components = ["FET-A", "FET-B"]
        self.basic_components = ["FET-A", "FET-B", "capacitor", "inductor"]
        self.reward = 0
        self.current = TopoGenState(init=True)
        self.configs_ = _configs
        self.num_component_ = _num_component
        self.query_counter = 0
        self.hash_counter = 0
        self.graph_2_reward = {}
        self.encode_graph_2_reward = {}
        self.no_isom_graph_2_reward = {}
        self.encode_no_isom_graph_2_reward = {}
        self.prohibited_path = ['VIN - inductor - GND', 'VIN - inductor - VOUT', 'VOUT - inductor - GND',
                                'VIN - FET-A - GND', 'VIN - FET-B - GND', 'VOUT - FET-A - GND', 'VOUT - FET-B - GND',
                                'VIN - FET-A - FET-A - GND', 'VIN - FET-B - FET-B - GND', 'VOUT - FET-A - FET-A - GND',
                                'VOUT - FET-B - FET-B - GND']
        self.approved_path_freq = _approved_path_freq
        self.component_condition_prob_ = _component_condition_prob

        # move to state
        # self.step = 0
        self.act_vect = []

        self.update_action_set(None)

    def set_state(self, state):
        self.current = state.duplicate()
        e1 = self.current.step - (len(self.current.component_pool) - 3)
        e1 %= len(self.current.port_pool)
        self.update_action_set(e1)

    def get_state(self):
        return self.current

    def finish_node_set(self):
        self.current.init_disjoint_set()

    # tested
    def add_node(self, node_id):
        # if self.current.num_component >= self.target.num_component:
        #     print('Error: Node action should not be able.')
        count = str(self.current.count_map[self.basic_components[node_id]])
        self.current.count_map[self.basic_components[node_id]] += 1
        # component = self.basic_components[node_id] + '-' + str(len(self.current.component_pool))
        component = self.basic_components[node_id] + '-' + count
        self.current.component_pool.append(component)
        idx_component_in_pool = len(self.current.component_pool) - 1
        self.current.port_pool.append(component + '-left')
        self.current.port_pool.append(component + '-right')
        self.current.port_2_idx[component + '-left'] = len(self.current.port_2_idx)
        self.current.port_2_idx[component + '-right'] = len(self.current.port_2_idx)
        self.current.comp2port_mapping[idx_component_in_pool] = [self.current.port_2_idx[component + '-left'],
                                                                 self.current.port_2_idx[component + '-right']]
        self.current.port2comp_mapping[self.current.port_2_idx[component + '-left']] = idx_component_in_pool
        self.current.port2comp_mapping[self.current.port_2_idx[component + '-right']] = idx_component_in_pool
        self.current.idx_2_port[len(self.current.idx_2_port)] = component + '-left'
        self.current.idx_2_port[len(self.current.idx_2_port)] = component + '-right'
        self.current.same_device_mapping[self.current.port_2_idx[component + '-left']] = self.current.port_2_idx[
            component + '-right']
        self.current.same_device_mapping[self.current.port_2_idx[component + '-right']] = self.current.port_2_idx[
            component + '-left']
        self.current.num_component += 1

    # tested
    def add_edge(self, edge):
        if edge[0] < 0:
            return
        # print('edge', edge)
        self.current.graph[edge[0]].add(edge[1])
        self.current.graph[edge[1]].add(edge[0])
        # print('parent value', self.current.parent)
        union(edge[0], edge[1], self.current.parent)
        return

    # basic_parameters = {"VIN": [1, 0, 0, 0, 0, 0, 0, 0],
    #            "VOUT": [0, 1, 0, 0, 0, 0, 0, 0],
    #            "GND": [0, 0, 1, 0, 0, 0, 0, 0],
    #            "FET-A-0": [0, 0, 0, 0, 0, 0.1, 0, 0],
    #            "FET-B-0": [0, 0, 0, 0, 0, 0, 0.25, 0],
    #            "inductor-0-left": [0, 0, 0, 0, 0, 0, 0, 1],
    #            "inductor-0-right": [0, 0, 0, 0, 0, 0, 0, 1],
    #            "FET-A-0-left": [0, 0, 0, 0, 0, 0, 0, 1],
    #            "FET-A-0-right": [0, 0, 0, 0, 0, 0, 0, 1],
    #            "FET-B-0-left": [0, 0, 0, 0, 0, 0, 0, 1],
    #            "FET-B-0-right": [0, 0, 0, 0, 0, 0, 0, 1],
    #            "inductor-0": [0, 0, 0, 0, 0.75, 0, 0, 0]}

    def edge_lead_to_prohibited_path(self, edge):
        tmp_state = deepcopy(self.get_state())
        if edge[0] < 0:
            return False
        tmp_state.graph[edge[0]].add(edge[1])
        tmp_state.graph[edge[1]].add(edge[0])
        union(edge[0], edge[1], tmp_state.parent)
        list_of_node, list_of_edge, netlist, joint_list = gen_topo.convert_to_netlist(tmp_state.graph,
                                                                                      tmp_state.component_pool,
                                                                                      tmp_state.port_pool,
                                                                                      tmp_state.parent,
                                                                                      tmp_state.comp2port_mapping)
        path = gen_topo.find_paths_from_edges(list_of_node, list_of_edge)
        # if path:
        #     print(path)
        check_result = gen_topo.check_topo_path(path, self.prohibited_path)
        # if not check_result:
        #     print(edge)
        #     print(path)
        #     print(self.prohibited_path)
        return not check_result

    def check_prohibited_path(self):
        tmp_state = deepcopy(self.get_state())
        list_of_node, list_of_edge, netlist, joint_list = gen_topo.convert_to_netlist(self.current.graph,
                                                                                      self.current.component_pool,
                                                                                      self.current.port_pool,
                                                                                      self.current.parent,
                                                                                      self.current.comp2port_mapping)
        path = gen_topo.find_paths_from_edges(list_of_node, list_of_edge)
        check_result = gen_topo.check_topo_path(path, self.prohibited_path)
        return check_result

    def get_edge_weight(self, e1, e2):
        weight = 0
        component_e1_idx = self.current.port2comp_mapping[e1]
        component_e2_idx = self.current.port2comp_mapping[e2]
        component_e1 = self.current.component_pool[component_e1_idx]
        component_e2 = self.current.component_pool[component_e2_idx]
        edge = get_component_type(component_e1) + ' - ' + get_component_type(component_e2)
        for path_freq in self.approved_path_freq:
            if edge in path_freq[0]:
                weight += 1
        return weight

    def get_edge_weight_with_freq(self, e1, e2):
        weight = 0
        component_e1_idx = self.current.port2comp_mapping[e1]
        component_e2_idx = self.current.port2comp_mapping[e2]
        component_e1 = self.current.component_pool[component_e1_idx]
        component_e2 = self.current.component_pool[component_e2_idx]
        edge = get_component_type(component_e1) + " - " + get_component_type(component_e2)
        reversed_edge = get_component_type(component_e2) + " - " + get_component_type(component_e1)
        for path_freq in self.approved_path_freq:
            if (edge in path_freq[0]) or (reversed_edge in path_freq[0]):
                weight += path_freq[1]
        return weight

    def reach_component_number(self):
        if len(self.current.component_pool) < self.num_component_:
            return False
        return True

    def update_action_set(self, e1):
        """
        After action, the topology is changed, we need to find out which edges we can connect
        """
        if len(self.current.component_pool) < self.num_component_:
            if self.configs_['must_have_switch']:
                if len(self.current.component_pool) == 3:
                    # print('first step')
                    self.act_vect = [TopoGenAction('node', i) for i in range(len(self.necessary_components))]
                else:
                    self.act_vect = [TopoGenAction('node', i) for i in range(len(self.basic_components))]
            else:
                self.act_vect = [TopoGenAction('node', i) for i in range(len(self.basic_components))]

        # if len(self.current.component_pool) < self.num_component_:
        #     if len(self.current.component_pool) == 3:
        #         self.act_vect = [TopoGenAction('node', i) for i in range(len(self.basic_components))]
        #     else:
        #         self.act_vect = []
        #         last_seleted_component = self.current.component_pool[-1]
        #         type_idx = last_seleted_component.rfind('-')
        #         last_seleted_component_type = last_seleted_component[:type_idx]
        #         component_priorities = get_component_priorities()
        #         for i in range(len(self.basic_components)):
        #             if component_priorities[self.basic_components[i]] >= \
        #                     component_priorities[last_seleted_component_type]:
        #                 self.act_vect.append(TopoGenAction('node', i))
        else:

            edge_weight_list = []

            self.act_vect.clear()
            self.act_vect.append(TopoGenAction('edge', [-1, -1]))
            if self.current.graph_is_valid():
                self.act_vect.append(TopoGenAction('terminal', 0))

            # e1 = self.current.step - (len(self.current.component_pool) - 3)
            # e1 %= len(self.current.port_pool)

            # TODO to let the ground search first
            # if e1 == 0:
            #     e1 = 2
            # elif e1 == 2:
            #     e1 = 0
            # if e1 >= len(self.current.port_pool):
            #     return
            # all the available edge set with e1 as a node
            e2_pool = list(range(len(self.current.port_pool)))
            random.shuffle(e2_pool)

            for e2 in e2_pool:
                # the same port
                if e1 == e2:
                    continue
                # TODO assume we can not let large port to connect small port
                if e1 >= e2:
                    continue
                # from the same device
                if e2 in self.current.same_device_mapping and \
                        e1 == self.current.same_device_mapping[e2]:
                    continue
                # existing edges
                if (e1 in self.current.graph and e2 in self.current.graph[e1]) or \
                        (e2 in self.current.graph and e1 in self.current.graph[e2]):
                    continue

                # disjoint set
                e1_root = find(e1, self.current.parent)
                e2_root = find(e2, self.current.parent)
                # TODO fix the order, althouth currently not effect the result
                gnd_root = find(0, self.current.parent)
                vin_root = find(1, self.current.parent)
                vout_root = find(2, self.current.parent)
                special_roots = [vin_root, vout_root, gnd_root]

                if e1_root in special_roots and e2_root in special_roots:
                    continue
                if e1_root == e2_root:
                    continue

                if e1 in self.current.same_device_mapping:
                    e1_other_port = self.current.same_device_mapping[e1]
                    e1_other_port_root = find(e1_other_port, self.current.parent)
                    if e1_other_port_root == e2_root:
                        continue

                if e2 in self.current.same_device_mapping:
                    e2_other_port = self.current.same_device_mapping[e2]
                    e2_other_port_root = find(e2_other_port, self.current.parent)
                    if e2_other_port_root == e1_root:
                        continue
                if self.configs_['prohibit_path']:
                    if self.edge_lead_to_prohibited_path([e1, e2]):
                        continue
                if self.configs_['approve_path']:
                    # edge_weight = {'edge':[e1, e2], 'weight':self.get_edge_weight(e1, e2)}
                    edge_weight = {'edge': [e1, e2], 'weight': self.get_edge_weight_with_freq(e1, e2)}
                    insert_flag = None
                    for i in range(len(edge_weight_list)):
                        if edge_weight['weight'] > edge_weight_list[i]['weight']:
                            edge_weight_list.insert(i, edge_weight)
                            insert_flag = True
                            break
                    if not insert_flag:
                        edge_weight_list.append(edge_weight)
                else:
                    edge_weight = {'edge': [e1, e2], 'weight': -1}
                    edge_weight_list.append(edge_weight)
            for edge_weight in edge_weight_list:
                self.act_vect.append(TopoGenAction('edge', edge_weight['edge']))
        # if len(self.act_vect) == 0:
        #     self.act_vect.append(TopoGenAction('edge', [-1, -1]))
        #     self.act_vect.append(TopoGenAction('terminal', 0))
        # if len(self.act_vect) > 0:
        #     # if e1 in self.current.graph and len(self.current.graph[e1]) > 0:
        #     #     self.act_vect.append(TopoGenAction('edge', [-1, -1]))
        #     if e1 in self.current.graph and len(self.current.graph[e1]) > 0:
        #         e1_name = self.current.idx_2_port[e1]
        #         if e1_name in self.target.port_2_idx:
        #             e1_target = self.target.port_2_idx[e1_name]
        #             if len(self.target.graph[e1_target]) == len(self.current.graph[e1]):
        #                 self.act_vect.append(TopoGenAction('edge', [-1, -1]))
        return

    def act(self, _action, want_reward=True):
        if want_reward:
            parent_reward = self.get_reward()
        if _action.type == 'node':
            self.add_node(_action.value)
            self.current.step += 1
        elif _action.type == 'edge':
            self.add_edge(_action.value)
            self.current.step += 1
        elif _action.type == 'terminal':
            self.current.step = len(self.current.component_pool) - 3 + len(self.current.port_pool)
        else:
            print('Error: Unsupported Action Type!')
        if len(self.current.component_pool) == self.num_component_ and \
                not bool(self.current.parent):
            self.finish_node_set()
        if self.configs_['prohibit_path']:
            if _action.type == 'edge':
                if not self.check_prohibited_path():
                    print("errrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrror")
                assert self.check_prohibited_path()

        e1 = self.current.step - (len(self.current.component_pool) - 3)
        e1 %= len(self.current.port_pool)
        self.update_action_set(e1)

        if want_reward:
            self.reward = self.get_reward()
            self.reward = self.reward - parent_reward
            return self.reward
        else:
            return None

    def add_edges(self, edges, mc_return, gamma, discount, final_return, reward_list):
        for edge in edges:
            r = self.act(TopoGenAction('edge', edge))
            reward_list.append(r)

            mc_return += discount * r
            if not final_return or (final_return < mc_return):
                final_return = mc_return
            discount *= gamma
        return mc_return, discount, final_return, reward_list

    def get_adjust_parameter_set(self):
        L_param = 8
        empty_param = [0] * L_param
        nodes_param = empty_param[:]
        nodes_param[L_param - 1] = 1
        adjust_para_set = []
        inductor_parameters_set = []
        candidate_para_for_components = []
        inductor_num = 0
        for component in self.current.component_pool:
            if "inductor" in component:
                candidate_para_for_components.append(self.current.candidate_para["inductor"])
        for cand_para_set in product(*candidate_para_for_components):
            # print(cand_para_set)
            adjust_parameters = {}
            t = 0
            for component in self.current.component_pool:
                if "inductor" in component:
                    adjust_parameters[component] = empty_param[:]
                    adjust_parameters[component][4] = cand_para_set[t]
                    adjust_parameters[component + "-left"] = nodes_param
                    adjust_parameters[component + "-right"] = nodes_param
                    t += 1
            inductor_parameters_set.append(adjust_parameters)
        return inductor_parameters_set

    def get_basic_parameter(self):
        L_param = 8
        empty_param = [0] * L_param
        nodes_param = empty_param[:]
        nodes_param[L_param - 1] = 1
        basic_parameters = {"VIN": [1, 0, 0, 0, 0, 0, 0, 0],
                            "VOUT": [0, 1, 0, 0, 0, 0, 0, 0],
                            "GND": [0, 0, 1, 0, 0, 0, 0, 0]}
        for component in self.current.component_pool:
            if "capacitor" in component:
                basic_parameters[component] = empty_param[:]
                basic_parameters[component][3] = self.current.candidate_para["capacitor"][0]
                basic_parameters[component + "-left"] = nodes_param
                basic_parameters[component + "-right"] = nodes_param
            elif "FET-A" in component:  # nmos
                basic_parameters[component] = empty_param[:]
                basic_parameters[component][5] = self.current.candidate_para["FETA"][0]
                basic_parameters[component + "-left"] = nodes_param
                basic_parameters[component + "-right"] = nodes_param
            elif "FET-B" in component:
                basic_parameters[component] = empty_param[:]
                basic_parameters[component][6] = self.current.candidate_para["FETB"][0]
                basic_parameters[component + "-left"] = nodes_param
                basic_parameters[component + "-right"] = nodes_param

        return basic_parameters

    def get_highest_reward(self):
        highest_reward = None
        assert (self.current.graph_is_valid())
        basic_para = self.get_basic_parameter()
        adj_inductor_paras = self.get_adjust_parameter_set()
        for inductor_para in adj_inductor_paras:
            tmp_basic_para = deepcopy(basic_para)
            self.current.parameters = Merge(tmp_basic_para, inductor_para)
            self.get_reward()

    def get_reward(self):
        query_flag = 0
        highest_reward = None
        highest_reward_efficiency = None
        highest_reward_out_voltage = None
        tmp_reward = None
        if not self.current.graph_is_valid():
            highest_reward = 0
            self.reward = highest_reward
            return self.reward
        basic_para = self.get_basic_parameter()
        adj_inductor_paras = self.get_adjust_parameter_set()
        for inductor_para in adj_inductor_paras:

            tmp_basic_para = deepcopy(basic_para)
            self.current.parameters = Merge(tmp_basic_para, inductor_para)

            graph_info = str(self.get_state().port_pool) + "|" + str(sort_dict_string(self.current.graph))
            encode_graph_info = hash_str_encoding(graph_info)

            isom_str_set = instance_to_isom_str(self.get_state())
            topo_info = str(isom_str_set[0]) + "|" + str(isom_str_set[1])
            encode_topo_info = hash_str_encoding(topo_info)

            # if self.graph_2_reward.__contains__(hash_str_encoding(topo_info)):
            #     self.reward = self.graph_2_reward[hash_str_encoding(topo_info)][0]
            # if self.graph_2_reward.__contains__(topo_info):
            # print(self.encode_graph_2_reward.__contains__(encode_graph_info))
            if self.encode_graph_2_reward.__contains__(encode_graph_info):
                self.reward = self.encode_graph_2_reward[encode_graph_info][0]
                self.hash_counter += 1
                return self.reward
            elif self.encode_no_isom_graph_2_reward.__contains__(encode_topo_info):
                tmp_reward = self.encode_no_isom_graph_2_reward[encode_topo_info][0]
                tmp_efficiency = self.encode_no_isom_graph_2_reward[encode_topo_info][1]
                tmp_out_voltage = self.encode_no_isom_graph_2_reward[encode_topo_info][2]
                if query_flag == 0:
                    self.query_counter += 1
                    query_flag = 1
            else:
                topologies = [self.get_state()]
                nets_to_ngspice_files(topologies, self.configs_, self.num_component_)
                simulate_topologies(len(topologies), self.num_component_, self.configs_['sys_os'])
                effis = analysis_topologies(self.configs_, len(topologies), self.num_component_)

                effi = effis[0]
                if query_flag == 0:
                    self.query_counter += 1
                    query_flag = 1

                if not effi.__contains__('output_voltage'):
                    tmp_reward = 0
                    effi['output_voltage'] = 0
                else:
                    tmp_reward = calculate_reward(effi)
                tmp_efficiency = effi['output_voltage']
                tmp_out_voltage = effi['efficiency']
                # self.graph_2_reward[graph_info] = [tmp_reward,
                #                                    effi['efficiency'],
                #                                    effi['output_voltage']]
                #
                # self.encode_graph_2_reward[encode_graph_info] = [tmp_reward,
                #                                                  effi['efficiency'],
                #                                                  effi['output_voltage']]

                self.no_isom_graph_2_reward[topo_info] = [tmp_reward,
                                                          effi['efficiency'],
                                                          effi['output_voltage']]

                self.encode_no_isom_graph_2_reward[encode_topo_info] = [tmp_reward,
                                                                        effi['efficiency'],
                                                                        effi['output_voltage']]

                save_one_in_hash_file(topo_info,
                                      [tmp_reward, effi['efficiency'], effi['output_voltage']],
                                      "topo_info_hash.txt")
                save_one_in_hash_file(encode_topo_info,
                                      [tmp_reward, effi['efficiency'], effi['output_voltage']],
                                      "encode_topo_info_hash.txt")
                if tmp_reward > 0.1:
                    file_name = "Results/special.txt"
                    fo = open(file_name, "w")
                    fo.write("================================================" + "\n")
                    fo.write(str(self.current.port_pool) + "\n")
                    fo.write(str(self.current.graph) + "\n")
                    fo.write(str(effi) + "\n")
                    fo.close()
            if not highest_reward or (highest_reward < tmp_reward):
                highest_reward = tmp_reward
                highest_reward_efficiency = tmp_efficiency
                highest_reward_out_voltage = tmp_out_voltage
                highest_parameter = deepcopy(self.current.parameters)
        self.graph_2_reward[graph_info] = [highest_reward,
                                           highest_reward_efficiency,
                                           highest_reward_out_voltage]
        self.encode_graph_2_reward[encode_graph_info] = [highest_reward,
                                                         highest_reward_efficiency,
                                                         highest_reward_out_voltage]
        save_one_in_hash_file(graph_info,
                              [tmp_reward, highest_reward_efficiency, highest_reward_out_voltage],
                              "graph_info_hash.txt")
        save_one_in_hash_file(encode_graph_info,
                              [tmp_reward, highest_reward_efficiency, highest_reward_out_voltage],
                              "encode_graph_info_hash.txt")

        self.reward = highest_reward
        self.current.parameters = highest_parameter
        return self.reward

    def supplement_components(self):
        while len(self.current.component_pool) < self.num_component_:
            actions = self.get_actions()
            act_ID = int(random.random() * len(actions))
            r = self.act(actions[act_ID])

    def current_checking_port(self):
        e1 = (self.current.step - (len(self.current.component_pool) - 3))
        e1 %= len(self.current.port_pool)
        return e1

    def component_set_in_current(self, component_set):
        """
        add counts on current path's components and find whether the are exist in current sub graph
        :param component_set: components without count in path
        :return: components with count in path
        """
        count_set = {"FET-A": 0, "FET-B": 0, "capacitor": 0, "inductor": 0}
        for idx in range(len(component_set)):
            if component_set[idx] in ['GND', 'VIN', 'VOUT']:
                continue
            else:
                count_set[component_set[idx]] += 1
                # in analytics, no - before count
                component_set[idx] = component_set[idx] + '-' + str(count_set[component_set[idx]] - 1)
        if set(component_set).issubset(set(self.current.component_pool)):
            return component_set
        else:
            return None

    def generate_possible_port_lists(self, components_with_count):
        component_port_pairs = []
        for comp_idx in range(1, len(components_with_count) - 1):
            component_port_pairs.append([[components_with_count[comp_idx] + '-left',
                                          components_with_count[comp_idx] + '-right'],
                                         [components_with_count[comp_idx] + '-right',
                                          components_with_count[comp_idx] + '-left']])
        path_port_list = component_port_pairs[0]
        for i in range(1, len(component_port_pairs)):
            path_port_list = port_list_multiply(path_port_list, component_port_pairs[i])
        return path_port_list

    def generate_primary_edges_to_add(self, path_port):
        edge_union_find_set = {}
        for i in range(0, len(path_port), 2):
            start = self.current.port_2_idx[path_port[i]]
            end = self.current.port_2_idx[path_port[i + 1]]
            edge = [start, end] if start < end else [end, start]
            if edge[0] in edge_union_find_set:
                edge_union_find_set[edge[0]].append(edge[1])
            else:
                edge_union_find_set[edge[0]] = [edge[1]]
        primary_edges_to_add = []
        for start, ports in edge_union_find_set.items():
            ports.sort()
            ports.insert(0, start)
            for i in range(len(ports) - 1):
                primary_edges_to_add.append([ports[i], ports[i + 1]])

        def keyFunc(element):
            return element[0]

        primary_edges_to_add.sort(key=keyFunc)
        return primary_edges_to_add

    def triangle_connection(self, edge):
        edge_0_other_port = self.current.same_device_mapping[edge[0]]
        edge_1_other_port = self.current.same_device_mapping[edge[1]]
        edge_0_root = find(edge[0], self.current.parent)
        edge_1_root = find(edge[1], self.current.parent)
        edge_0_other_port_root = find(edge_0_other_port, self.current.parent)
        edge_1_other_port_root = find(edge_1_other_port, self.current.parent)

        if (edge_0_other_port_root == edge_1_root) or \
                (edge_1_other_port_root == edge_0_root):
            return True
        return False

    def check_add_path(self, path_port, port_0, allowed_actions):
        current_parent = deepcopy(self.current.parent)
        current_graph = deepcopy(self.current.graph)
        primary_edges_to_add = self.generate_primary_edges_to_add(path_port)
        # primary_edges_to_add: [[0,7],[1,3],[4,8]]
        # print(primary_edges_to_add)
        path_possible_edges = [[]]
        for primary_edge in primary_edges_to_add:
            # primary_edge [0,7]

            start = primary_edge[0]
            end = primary_edge[1]
            if already_connected(start, end, current_parent):
                continue
            net_of_start = find_connected_set(start, current_parent)
            net_of_end = find_connected_set(end, current_parent)
            possible_edges = port_multiply(net_of_start, net_of_end)
            i = 0
            check_root_connected_flag = False
            while i < len(possible_edges):
                edge = possible_edges[i]
                if not check_root_connected_flag:
                    e1_root = find(edge[0], self.current.parent)
                    e2_root = find(edge[1], self.current.parent)
                    gnd_root = find(0, self.current.parent)
                    vin_root = find(1, self.current.parent)
                    vout_root = find(2, self.current.parent)
                    special_roots = [vin_root, vout_root, gnd_root]
                    if e1_root in special_roots and e2_root in special_roots:
                        return None, []
                    else:
                        check_root_connected_flag = 1

                if edge[0] == port_0:
                    if not find_action(TopoGenAction('edge', edge), allowed_actions):
                        possible_edges.remove(edge)
                        continue
                if edge[0] < port_0 or edge[1] < port_0:
                    possible_edges.remove(edge)
                    continue
                elif edge[0] in self.current.same_device_mapping and \
                        edge[1] in self.current.same_device_mapping:
                    if self.triangle_connection(edge):
                        possible_edges.remove(edge)
                        continue
                    else:
                        i += 1
                        continue
                else:
                    i += 1
                    continue
            if not possible_edges:
                return None, []
            path_possible_edges = path_add_edges(path_possible_edges, possible_edges)

        # print(path_possible_edges)

        def keyFunc(element):
            return element[0]

        considerable_skip = []
        considerable_path_edge = []

        for possible_path in path_possible_edges:
            current_parent = deepcopy(self.current.parent)
            current_graph = deepcopy(self.current.graph)
            pre_start = -2
            need_add_list = []
            edge_to_add_current_port = []

            possible_path.sort(key=keyFunc)
            # print(possible_path)
            for edge in possible_path:
                start = edge[0]
                end = edge[1]
                # need add two ports on one edge
                if start == pre_start:
                    continue
                if not already_connected(start, end, current_parent):
                    # we need to add port on previous considered ports
                    if start < port_0 or end < port_0:
                        continue
                    else:
                        if start == port_0:
                            # we need to add some port on current considering port which is
                            # not allowed
                            if not find_action(TopoGenAction('edge', edge), allowed_actions):
                                continue
                            else:
                                edge_to_add_current_port = edge
                    need_add_list.append([start, end])
                    union(start, end, current_parent)
                    current_graph[start].add(end)
                    current_graph[end].add(start)
                pre_start = start
            if graph_has_short_cut(current_graph, current_parent, self.current.same_device_mapping):
                continue
            elif graph_connect_roots(current_parent):
                continue
            else:
                """we use a rule that for a path:
                    if: we both skip([-1,-1]) and adding an edge[port_0, port_1] is possible, we only 
                    consider the minimum number of adding edges for adding an edge
                    else, if we only have skip[-1,-1] for choise, we only consider skip, still minimum 
                    number of adding edges
                """
                if need_add_list:
                    if edge_to_add_current_port:
                        if (not considerable_path_edge) or \
                                len(need_add_list) < len(considerable_path_edge[0]):
                            considerable_path_edge = [need_add_list, edge_to_add_current_port]
                    else:
                        if (not considerable_skip) or \
                                len(need_add_list) < len(considerable_skip[0]):
                            considerable_skip = [need_add_list, [-1, -1]]
                else:
                    continue
        if considerable_path_edge:
            return considerable_path_edge[0], considerable_path_edge[1]
        elif considerable_skip:
            return considerable_skip[0], considerable_skip[1]
        else:
            return None, []

    # def check_add_path(self, path_port, port_0, allowed_actions):
    #     current_parent = deepcopy(self.current.parent)
    #     current_graph = deepcopy(self.current.graph)
    #     primary_edges_to_add = self.generate_primary_edges_to_add(path_port)
    #     print(primary_edges_to_add)
    #     need_add_list = []
    #     edge_to_add_current_port = []
    #     pre_start = -2
    #     for edge in primary_edges_to_add:
    #         start = edge[0]
    #         if start == pre_start:
    #             return None, []
    #
    #         end = edge[1]
    #
    #         if not already_connected(start, end, current_parent):
    #             if start < port_0 or end < port_0:
    #                 return None, []
    #             else:
    #                 if start == port_0:
    #                     if not find_action(TopoGenAction('edge', edge), allowed_actions):
    #                         return None, []
    #                     else:
    #                         edge_to_add_current_port = edge
    #             need_add_list.append([start, end])
    #             union(start, end, current_parent)
    #             current_graph[start].add(end)
    #             current_graph[end].add(start)
    #         pre_start = start
    #     if graph_has_short_cut(current_graph, current_parent, self.current.same_device_mapping):
    #         return None, []
    #     elif graph_connect_roots(current_parent):
    #         return None, []
    #     else:
    #         return need_add_list, edge_to_add_current_port

    def edges_adding_from_path(self, allowed_root_pair, path, port_0, allowed_actions):
        """
        return the edges need to be add if we want to let this path generated in current sub-graph
        :param allowed_root_pair:
        :param path: path we consider
        :param port_0:
        :param allowed_actions:
        :return:
        """
        path = str(path)
        can_add = True
        edges_to_add = []
        ''' 'VIN - FET-A - L - GND' '''
        components = path.split(' - ')
        ports_idx = []
        ''' components: [VIN, FET-A, L, GND] '''
        components_with_count = self.component_set_in_current(components)
        ''' components_with_count: [VIN, FET-A-0, L-0, GND] '''
        if not components_with_count:
            return False, []
        elif not find_roots(allowed_root_pair, [components[0], components[-1]]):
            return False, []
        else:
            """
            path port_list: 
                [FET-A-0-left,FET-A-0-right, L-0-left, L-0-right]
                [FET-A-0-right,FET-A-0-left, L-0-left, L-0-right]
                [FET-A-0-left,FET-A-0-right, L-0-right, L-0-left]
                [FET-A-0-right,FET-A-0-left, L-0-right, L-0-left]
                """
            path_port_list = self.generate_possible_port_lists(components_with_count)

        path_adding_cost = []
        for path_port in path_port_list:
            path_port.insert(0, components[0])
            path_port.append(components[-1])

            need_add_edge_list, edge_to_add_current_port = self.check_add_path(path_port,
                                                                               port_0,
                                                                               allowed_actions)
            if need_add_edge_list:
                if edge_to_add_current_port:
                    path_adding_cost.append((need_add_edge_list, edge_to_add_current_port))
                else:
                    path_adding_cost.append((need_add_edge_list, [-1, -1]))
            else:
                continue
        if path_adding_cost:
            edge_to_add = generate_port_path_add(path_adding_cost)
            return True, edge_to_add
        return False, []

    def choose_path_to_add(self, allowed_root_pair, port_0, allowed_actions):
        """
        we get the path that allowed to be add in
        :param allowed_root_pair: the root pair that has not been connected in  current graph
        :param port_0: the start port that current round is considering
        :param allowed_actions: all the actions that allowed in this round(will not lead to shortcut and direct of roots
        :return:  path_with_probability: key: path's index(idx), value:[edge_to_add(from port_0), probability]
        """
        sum_edge_count = 0
        sum_freq = 0
        edge_with_probability = {}
        approved_path = []
        for path_freq in self.approved_path_freq:
            approved_path.append(path_freq[0])
        for path_idx in range(len(approved_path)):
            can_add, edge_to_add = self.edges_adding_from_path(allowed_root_pair, approved_path[path_idx],
                                                               port_0, allowed_actions)
            if can_add:
                edge_with_probability[path_idx] = edge_to_add
                sum_freq += self.approved_path_freq[path_idx][1]
        #         allowed_actions.remove()
        # for actions in allowed_actions:
        for path_idx, edge_to_add in edge_with_probability.items():
            probability = self.approved_path_freq[path_idx][1] / sum_freq
            edge_with_probability[path_idx] = (edge_to_add, probability)
        return edge_with_probability

    def get_roots_of_path(self, path):
        """
        get the root pair of a path
        :param path: VIN-L-C-VOUT, string form
        :return: [VIN, VOUT]
        """
        root_0 = self.current.idx_2_port[path[0][0]]
        root_1 = self.current.idx_2_port[path[-1][-1]]
        return [root_0, root_1]

    def find_not_connected_roots(self):
        """
        find the not connected root pairs, we just allow using approve path to connect one path between
        each pair of roots
        :return: list of root pair that not connected,
        """
        allowed_root_pair = [['VIN', 'VOUT'], ['GND', 'VOUT'], ['VIN', 'GND']]
        current_state = self.get_state()
        list_of_node, list_of_edge, netlist, joint_list = \
            gen_topo.convert_to_netlist(current_state.graph,
                                        current_state.component_pool,
                                        current_state.port_pool,
                                        current_state.parent,
                                        current_state.comp2port_mapping)
        paths = gen_topo.find_paths_from_edges(list_of_node, list_of_edge)
        for path in paths:
            path_comp = path.split(' - ')
            allowed_root_pair = remove_roots(allowed_root_pair, [path_comp[0], path_comp[-1]])
            if not allowed_root_pair:
                return []
        return allowed_root_pair

    def get_action_using_default_policy(self, allowed_actions,dp_for_component, dp_for_path):
        """
        get an action that (1) can be add in current topology according to approve path set(find a path which can be add
        into current topology, then return the edge which is (2) allowed in current round
        :param dp_for_path: true for using dp for choosing a path, false for randomly
        :param dp_for_component: true for using dp for choosing a component, false for randomly
        :param allowed_actions: the allowed action set in current round. for example, we want to add the edges one of
        whose end is 3, then all the allowed actions are [3. x] or [-1, -1]
        :return: action(edge, [port_0, port_1])
        """
        if not self.reach_component_number():
            if dp_for_component:
                action = sample_component(self.current.component_pool, self.component_condition_prob_, allowed_actions,
                                          self.basic_components)
            else:
                act_ID = int(random.random() * len(allowed_actions))
                action = allowed_actions[act_ID]
            return action
        else:
            if dp_for_path:
                allowed_root_pair = self.find_not_connected_roots()
                # print('allowed_root_pair:', allowed_root_pair)
                if not allowed_root_pair:
                    return None

                port_0 = self.current_checking_port()
                # path_with_probability: key: path's index(idx), value:[edge_to_add(from port_0), probability]
                edges_with_probability = self.choose_path_to_add(allowed_root_pair, port_0, allowed_actions)

                if not edges_with_probability:
                    return None

                edge_to_add = sample_from_paths(edges_with_probability)
                return TopoGenAction('edge', edge_to_add)
            else:
                act_ID = int(random.random() * len(allowed_actions))
                action = allowed_actions[act_ID]
                return action
        return None

    def default_policy(self, mc_return, gamma, discount, final_return, reward_list, dp_for_component, dp_for_path):
        while not self.is_terminal():
            allowed_actions = self.get_actions()
            action = self.get_action_using_default_policy(allowed_actions, dp_for_component, dp_for_path)
            if action is None:
                act_ID = int(random.random() * len(allowed_actions))
                action = allowed_actions[act_ID]
            r = self.act(action)
            reward_list.append(r)
            mc_return += discount * r
            if not final_return or (final_return < mc_return):
                final_return = mc_return
            discount *= gamma
        self.get_state()
        # print(reward_list)
        if not final_return:
            final_return = 0
        return final_return

    def random_policy(self, mc_return, gamma, discount, final_return, reward_list):
        while not self.is_terminal():
            actions = self.get_actions()
            act_ID = int(random.random() * len(actions))
            r = self.act(actions[act_ID])
            reward_list.append(r)
            mc_return += discount * r
            if not final_return or (final_return < mc_return):
                final_return = mc_return
            discount *= gamma
        self.get_state()
        if not final_return:
            final_return = 0
        return final_return

    def get_actions(self):
        return self.act_vect

    def is_terminal(self):
        if self.current.step - (len(self.current.component_pool) - 3) >= len(self.current.port_pool):
            return True
        return False

#
# def construct_target_v2():
#     simulator = TopoGenSimulator(None)
#     # ["FET-A", "FET-B", "capacitor", "inductor"]
#     simulator.add_node(0)
#     simulator.add_node(0)
#     simulator.add_node(0)
#     simulator.add_node(0)
#     simulator.add_node(0)
#
#     simulator.add_node(1)
#     simulator.add_node(1)
#     simulator.add_node(1)
#     simulator.add_node(1)
#     simulator.add_node(1)
#
#     simulator.add_node(2)
#     simulator.add_node(2)
#     simulator.add_node(2)
#
#     simulator.add_node(3)
#     simulator.add_node(3)
#
#     simulator.current.init_disjoint_set()
#     # edges:
#     edges = [['VIN', 'FET-A-0-left'],  # 1
#              ['FET-A-0-right', 'FET-B-0-left'],  # 2
#              ['FET-A-0-right', 'capacitor-0-left'],  # 3
#              ['FET-B-0-right', 'FET-A-2-left'],  # 4
#              ['FET-B-0-right', 'capacitor-1-left'],  # 5
#              ['FET-A-2-right', 'FET-B-3-left'],  # 6
#              ['FET-A-2-right', 'capacitor-2-left'],  # 7
#              ['FET-B-3-right', 'VOUT'],  # 8
#              ['capacitor-0-right', 'inductor-0-left'],  # 9
#              ['capacitor-1-right', 'FET-B-2-left'],  # 10
#              ['capacitor-2-right', 'inductor-1-left'],  # 11
#              ['inductor-0-right', 'FET-A-1-left'],  # 12
#              ['inductor-0-right', 'FET-B-1-left'],  # 13
#              ['capacitor-1-right', 'FET-A-3-left'],  # 14
#              ['inductor-1-right', 'FET-A-4-left'],  # 15
#              ['inductor-1-right', 'FET-B-4-left'],  # 16
#              ['FET-A-1-right', 'VOUT'],  # 17
#              ['FET-B-1-right', 'GND'],  # 18
#              ['FET-B-2-right', 'VOUT'],  # 19
#              ['FET-A-3-right', 'GND'],  # 20
#              ['FET-A-4-right', 'VOUT'],  # 21
#              ['FET-B-4-right', 'GND']
#              ]
#     for edge in edges:
#         p1, p2 = simulator.current.port_2_idx[edge[0]], simulator.current.port_2_idx[edge[1]]
#         simulator.add_edge([p1, p2])
#     return simulator.get_state()
