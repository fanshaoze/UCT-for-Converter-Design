import random
from ucts import uct
from utils.util import hash_str_encoding, save_one_in_hash_file, Merge, sample_component, get_component_type, \
    get_component_count_suffix
from utils.eliminate_isomorphism import instance_to_isom_str, maintain_reward_hash_with_edge \
    , get_component_priorities
import collections
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import product
# from simulator3component.build_topology import nets_to_ngspice_files
# from simulator3component.simulation import simulate_topologies
# from simulator3component.simulation_analysis import analysis_topologies
from SimulatorSetPara.build_topology import nets_to_ngspice_files
from SimulatorSetPara.simulation import simulate_topologies
from SimulatorSetPara.simulation_analysis import analysis_topologies
from SimulatorAnalysis.UCT_data_collection import find_one_analytics_result, get_one_analytics_result, \
    save_one_analytics_result
import datetime
from SimulatorAnalysis.UCT_data_collection import *
from SimulatorAnalysis import gen_topo
import warnings
import scipy


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


def find_connected_set(x, parent):
    net_list = []
    for i in range(len(parent)):
        if already_connected(x, i, parent):
            net_list.append(i)
    return net_list


def already_connected(x, y, parent):
    return find(x, parent) == find(y, parent)


def get_component_type(component):
    if component.startswith('L'):
        ret = 'L'
    elif component.startswith('C'):
        ret = 'C'
    elif component.startswith('Sa'):
        ret = 'Sa'
    elif component.startswith('Sb'):
        ret = 'Sb'
    else:
        ret = component
    return ret


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


def calculate_reward(effi, target_vout, min_vout=None, max_vout=None):
    a = abs(target_vout) / 15
    if effi['efficiency'] > 1 or effi['efficiency'] < 0:
        return 0
    else:
        return effi['efficiency'] * (1.1 ** (-((effi['output_voltage'] - target_vout) / a) ** 2))


def print_calculate_rewards(target_vout, min_vout, max_vout):
    i = min_vout
    while i < max_vout:
        effi = {'efficiency': 1, 'output_voltage': i}
        reward = calculate_reward(effi, target_vout, min_vout, max_vout)
        print(i, reward)
        i += 0.1


def remove_roots(allowed_root_pair, current_root_pair):
    for root_pair in allowed_root_pair:
        if (current_root_pair[0] in root_pair) and (current_root_pair[1] in root_pair):
            allowed_root_pair.remove(root_pair)
            break
    return allowed_root_pair


def find_roots(allowed_root_pair, current_root_pair):
    for root_pair in allowed_root_pair:
        if (current_root_pair[0] in root_pair) and (current_root_pair[1] in root_pair):
            random_prob = random.random()
            # print(random_prob, allowed_root_pair[root_pair],root_pair, current_root_pair)
            if random_prob <= allowed_root_pair[root_pair]:
                # print(True)
                return True
            else:
                # print(False)
                return False
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
            self.count_map = {"Sa": 0, "Sb": 0, "C": 0, "L": 0}
            self.comp2port_mapping = {0: [0], 1: [1],
                                      2: [2]}  # key is the idx in component pool, value is idx in port pool
            self.port2comp_mapping = {0: 0, 1: 1, 2: 2}

            self.port_2_idx = {'GND': 0, 'VIN': 1, "VOUT": 2}
            self.idx_2_port = {0: 'GND', 1: 'VIN', 2: "VOUT"}
            self.same_device_mapping = {}
            self.graph = collections.defaultdict(set)
            self.parent = None
            self.step = 0
            self.parameters = "None"

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
        return len(self.component_pool) - len(['GND', 'VIN', 'VOUT'])

    def get_edge_num(self):
        edge_num = 0
        for key, val in self.graph.items():
            edge_num += len(val)
        return edge_num / 2

    def check_have_no_GND_path(self):
        list_of_node, list_of_edge, netlist, joint_list = \
            gen_topo.convert_to_netlist(self.graph, self.component_pool, self.port_pool,
                                        self.parent, self.comp2port_mapping)
        paths = gen_topo.find_paths_from_edges(list_of_node, list_of_edge, True)
        gnd_root = find(self.port_2_idx['GND'], self.parent)
        for path in paths:
            components = path.split(' - ')
            if find_roots({('VIN', 'VOUT'): 1}, [components[0], components[-1]]):
                find_flag = True
                for i in range(1, len(components) - 1):
                    component = components[i]
                    left_port = component + '-left'
                    right_port = component + '-right'
                    if left_port not in self.port_2_idx:
                        print(left_port)
                    if right_port not in self.port_2_idx:
                        print(right_port)
                    if find(self.port_2_idx[left_port], self.parent) == gnd_root or \
                            find(self.port_2_idx[right_port], self.parent) == gnd_root:
                        # means this path has gnd on it, we need to search the next path
                        find_flag = False
                        break
                if find_flag:
                    # only in the case that we find a vin-vout path and on this path we do not meet a gnd that
                    # we can reach here. So we can return that,
                    # we find a valid graph which has a vin-vout path that do not go over gnd
                    return True
        return False

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
        for i in range(len(['GND', 'VIN', 'VOUT']), len(self.port_pool)):
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
            return self.check_have_no_GND_path() and (not has_short_cut) and self.has_in_out_gnd() and \
                   self.has_all_ports() and nx.is_connected(G)
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
    def __init__(self, _configs, _approved_path_freq, _component_condition_prob, _key_expression={}, _key_sim_effi={},
                 _fix_para_set=None,
                 _num_component=4, target=None):
        self.necessary_components = ["Sa", "Sb"]
        self.basic_components = ["Sa", "Sb", "C", "L"]
        self.reward = 0
        self.current = TopoGenState(init=True)
        # current_max: {state:, reward:, para:}
        self.current_max = {}
        self.configs_ = _configs
        self.num_component_ = _num_component
        self.query_counter = 0
        self.hash_counter = 0
        self.graph_2_reward = {}
        self.encode_graph_2_reward = {}
        # key=key, value=expression
        self.key_expression = _key_expression
        # key=key, value=[list_of_node, list_of_edge, effi, vout, para, target_ratio]
        self.key_sim_effi_ = _key_sim_effi
        self.prohibit_path = ['VIN - L - GND', 'VIN - L - VOUT', 'VOUT - L - GND', 'VIN - Sa - GND', 'VIN - Sb - GND',
                              'VOUT - Sa - GND', 'VOUT - Sb - GND', 'VIN - Sa - Sa - GND', 'VIN - Sb - Sb - GND',
                              'VOUT - Sa - Sa - GND', 'VOUT - Sb - Sb - GND']

        self.topk = []
        self.approved_path_freq = _approved_path_freq
        self.component_condition_prob_ = _component_condition_prob
        # move to state
        # self.step = 0
        # weights are the sample weights of act_vect, same idx, same length
        self.act_vect = []
        self.weights = []
        self.fix_para_set = _fix_para_set

        self.update_action_set()

    def set_state(self, state):
        self.current = state.duplicate()
        self.update_action_set()

    def get_state(self):
        return self.current

    def finish_node_set(self):
        self.current.init_disjoint_set()

    def add_node(self, node_id):
        # if self.current.num_component >= self.target.num_component:
        #     print('Error: Node action should not be able.')
        count = str(self.current.count_map[self.basic_components[node_id]])
        self.current.count_map[self.basic_components[node_id]] += 1
        # component = self.basic_components[node_id] + '-' + str(len(self.current.component_pool))
        component = self.basic_components[node_id] + count
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

    def edge_lead_to_prohibit_path(self, edge):
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
        path = find_paths_from_edges(list_of_node, list_of_edge)
        # if path:
        #     print(list_of_node,list_of_edge)
        #     print(path)
        check_result = check_topo_path(path, self.prohibit_path)
        return not check_result

    def check_prohibit_path(self):
        list_of_node, list_of_edge, netlist, joint_list = gen_topo.convert_to_netlist(self.current.graph,
                                                                                      self.current.component_pool,
                                                                                      self.current.port_pool,
                                                                                      self.current.parent,
                                                                                      self.current.comp2port_mapping)
        path = find_paths_from_edges(list_of_node, list_of_edge)
        check_result = check_topo_path(path, self.prohibit_path)
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

    def update_action_set(self):
        """
        After action, the topology is changed, we need to find out which edges we can connect
        """
        if len(self.current.component_pool) < self.num_component_:
            self.act_vect = [TopoGenAction('node', i) for i in range(len(self.basic_components))]
            self.weights = [1 / len(self.basic_components) for i in range(len(self.basic_components))]
            # if len(self.current.component_pool) == len(['GND', 'VIN', 'VOUT']):
            #     self.act_vect = [TopoGenAction('node', i) for i in range(len(self.basic_components))]
            # else:
            #     self.act_vect = []
            #     last_seleted_component = self.current.component_pool[-1]
            #     type_idx = last_seleted_component.rfind('-')
            #     last_seleted_component_type = last_seleted_component[:type_idx]
            #     component_priorities = get_component_priorities()
            #     for i in range(len(self.basic_components)):
            #         if component_priorities[self.basic_components[i]] >= \
            #                 component_priorities[last_seleted_component_type]:
            #             self.act_vect.append(TopoGenAction('node', i))
        elif self.is_terminal():
            self.act_vect.append(TopoGenAction('terminal', 0))
        else:
            edge_weight_list = []
            self.act_vect.clear()
            if self.current.graph_is_valid():
                self.act_vect.append(TopoGenAction('terminal', 0))
            e1 = self.current.step - (len(self.current.component_pool) - len(['GND', 'VIN', 'VOUT']))

            e1 %= len(self.current.port_pool)
            # We do not need to skip for port 0, 1, 2, as we can not connect GND, VIN and VOUT
            # after skipping them

            connected_set = find_connected_set(e1, self.current.parent)
            # if connected_set != [e1], it means e1 has connected with some other ports,
            # so we can skip, else we can not skip as we only allowed small port to connect large port
            if connected_set != [e1]:
                self.act_vect.append(TopoGenAction('edge', [-1, -1]))
            # self.act_vect.append(TopoGenAction('edge', [-1, -1]))
            # TODO to let the ground search first
            # if e1 >= len(self.current.port_pool):
            #     return
            # all the available edge set with e1 as a node
            e2_pool = list(range(len(self.current.port_pool)))
            random.shuffle(e2_pool)

            for e2 in e2_pool:
                # the same port
                # if e1 == e2:
                #     continue
                # TODO assume we can not let large port to connect small port
                # if e1 >= e2:
                #     continue
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
                    if self.edge_lead_to_prohibit_path([e1, e2]):
                        continue
                # TODO need to be careful about the two layer shuffle in this function,
                #  will cover following method
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

                self.act_vect.append(TopoGenAction('edge', [e1, e2]))
            # It means, we can not skip e1, but we do not have allowed port to connect, what we can do is terminal
            if not self.act_vect:
                print("has to be unconnected")
                self.act_vect.append(TopoGenAction('terminal', 0))

        random.shuffle(self.act_vect)
        if len(self.act_vect) == 1:
            self.weights = [1]
        else:
            self.weights = [1 for i in range(len(self.act_vect))]
            for i in range(len(self.act_vect)):
                if self.act_vect[i].equal(TopoGenAction('edge', [-1, -1])):
                    self.weights[i] = (0.8 / (1 - 0.8)) * (len(self.act_vect) - 1)
        return

    def act(self, _action, reward_method='analytics', want_reward=True):
        if want_reward:
            if reward_method == 'analytics':
                parent_reward = self.get_reward()
            elif reward_method == 'simulator':
                parent_reward = self.get_reward_using_sim()
        if _action.type == 'node':
            self.add_node(_action.value)
            self.current.step += 1
        elif _action.type == 'edge':
            self.add_edge(_action.value)
            self.current.step += 1
        elif _action.type == 'terminal':
            self.current.step = len(self.current.component_pool) - len(['GND', 'VIN', 'VOUT']) + len(
                self.current.port_pool)
        else:
            print('Error: Unsupported Action Type!')
        if len(self.current.component_pool) == self.num_component_ and \
                not bool(self.current.parent):
            self.finish_node_set()
        if self.configs_['prohibit_path']:
            if _action.type == 'edge':
                if not self.check_prohibit_path():
                    print("errrrrrrrrrrrrrrrrrrrrrrrrrrror!!!")
                assert self.check_prohibit_path()
        self.update_action_set()
        if want_reward:
            if reward_method == 'analytics':
                self.reward = self.get_reward()
            elif reward_method == 'simulator':
                self.reward = self.get_reward_using_sim()
            self.reward = self.reward - parent_reward
            return self.reward
        else:
            return None

    def get_reward(self):
        if not self.current.graph_is_valid():
            self.current.parameters = 'None'
            self.reward = 0
            return self.reward

        list_of_node, list_of_edge, netlist, joint_list = gen_topo.convert_to_netlist(self.current.graph,
                                                                                      self.current.component_pool,
                                                                                      self.current.port_pool,
                                                                                      self.current.parent,
                                                                                      self.current.comp2port_mapping)
        key = key_circuit_from_lists(list_of_edge, list_of_node, netlist)
        # isom_str_set = instance_to_isom_str(self.get_state())
        # topo_info = str(isom_str_set[0]) + "|" + str(isom_str_set[1])

        # graph_info = str(self.get_state().port_pool) + "|" + str(sort_dict_string(self.current.graph))
        # encode_graph_info = hash_str_encoding(graph_info)
        # to isom, relpece graph_2_reward[key] woth ...key and the next line
        if self.graph_2_reward.__contains__(key):
            efficiency = self.graph_2_reward[key][1]
            vout = self.graph_2_reward[key][2]
            effi = {'efficiency': efficiency, 'output_voltage': vout}
            tmp_reward = calculate_reward(effi, self.configs_['target_vout'], self.configs_['min_vout'],
                                          self.configs_['max_vout'])

            self.current.parameters = self.graph_2_reward[key][0]
            self.reward = tmp_reward
            self.hash_counter += 1
            # print("find in hash------------------------------------------------------")
        else:
            start_time = datetime.datetime.now()
            para_result = find_one_analytics_result(key, self.key_expression, self.current.graph,
                                                    self.current.comp2port_mapping,
                                                    self.current.port2comp_mapping, self.current.idx_2_port,
                                                    self.current.port_2_idx,
                                                    self.current.parent, self.current.component_pool,
                                                    self.current.same_device_mapping, self.current.port_pool)
            print("find_one_analytics_result using: ", str((datetime.datetime.now() - start_time).seconds))
            if para_result:
                tmp_reward = 0
                tmp_para = 'None'
                if 'None' in para_result:
                    tmp_reward = 0
                    tmp_para = 'None'
                    self.graph_2_reward[key] = ['None',
                                                self.key_expression[key + '$' + 'Invalid']['Effiency'],
                                                self.key_expression[key + '$' + 'Invalid']['Vout']]
                else:
                    for k, v in para_result.items():
                        # print(k, v)
                        effi = {'efficiency': float(int(v[0]) / 100), 'output_voltage': float(v[1])}
                        # print(effi, self.configs_['target_vout'], self.configs_['min_vout'], self.configs_['max_vout'])
                        # print(tmp_reward,
                        #       calculate_reward(effi, self.configs_['target_vout'], self.configs_['min_vout'],
                        #                        self.configs_['max_vout']))
                        if tmp_reward <= calculate_reward(effi, self.configs_['target_vout'], self.configs_['min_vout'],
                                                          self.configs_['max_vout']):
                            tmp_reward = calculate_reward(effi, self.configs_['target_vout'], self.configs_['min_vout'],
                                                          self.configs_['max_vout'])
                            tmp_para = k
                            tmp_effi = float(int(v[0]) / 100)
                            tmp_vout = float(v[1])
                    # We only save the highest rewards' parameter, effi and cout(vout)
                    self.graph_2_reward[key] = [tmp_para, tmp_effi, tmp_vout]
            else:
                tmp_reward = 0
                tmp_para = 'None'
                # print("stuck here")
                # self.current.visualize("stuck!!!!", "figures/unexpected")
                para_result = get_one_analytics_result(self.key_expression, self.current.graph,
                                                       self.current.comp2port_mapping,
                                                       self.current.port2comp_mapping, self.current.idx_2_port,
                                                       self.current.port_2_idx,
                                                       self.current.parent, self.current.component_pool,
                                                       self.current.same_device_mapping, self.current.port_pool,
                                                       self.configs_['min_vout'])
                if para_result:
                    if 'None' in para_result:
                        tmp_para = 'None'
                        tmp_reward = 0
                        self.graph_2_reward[key] = ['None',
                                                    self.key_expression[key + '$' + 'Invalid']['Effiency'],
                                                    self.key_expression[key + '$' + 'Invalid']['Vout']]

                    else:
                        for k, v in para_result.items():
                            effi = {'efficiency': float(int(v[0]) / 100), 'output_voltage': float(v[1])}
                            # print(effi, self.configs_['target_vout'], self.configs_['min_vout'],
                            #       self.configs_['max_vout'])
                            # print(tmp_reward,
                            #       calculate_reward(effi, self.configs_['target_vout'], self.configs_['min_vout'],
                            #                        self.configs_['max_vout']))
                            if tmp_reward <= calculate_reward(effi, self.configs_['target_vout'],
                                                              self.configs_['min_vout'],
                                                              self.configs_['max_vout']):
                                tmp_reward = calculate_reward(effi, self.configs_['target_vout'],
                                                              self.configs_['min_vout'],
                                                              self.configs_['max_vout'])
                                tmp_para = k
                                tmp_effi = float(int(v[0]) / 100)
                                tmp_vout = float(v[1])
                        self.graph_2_reward[key] = [tmp_para, tmp_effi, tmp_vout]
                else:
                    tmp_para = 'None'
                    tmp_reward = 0
                    tmp_effi = 0
                    tmp_vout = -500
                    self.graph_2_reward[key] = [tmp_para, tmp_effi, tmp_vout]

            self.reward = tmp_reward
            self.current.parameters = tmp_para
            self.query_counter += 1

            if (self.current_max == {}) or self.current_max['reward'] < self.reward:
                self.current_max['state'] = self.current.duplicate()
                self.current_max['reward'] = self.reward
                self.current_max['para'] = self.current.parameters

            def keyFunc(element):
                return element[1]

            if not self.topk_include_current(key):
                self.topk.append([self.current.duplicate(), self.reward, self.current.parameters, key])
                if len(self.topk) > 5:
                    self.topk.sort(key=keyFunc)
                    self.topk.pop(0)

        return self.reward

    def topk_include_current(self, key):
        for top_item in self.topk:
            if key == top_item[3]:
                return True
        return False

    def get_reward_using_sim(self):
        if not self.current.graph_is_valid():
            self.current.parameters = 'None'
            self.reward = 0
            return self.reward

        list_of_node, list_of_edge, netlist, joint_list = gen_topo.convert_to_netlist(self.current.graph,
                                                                                      self.current.component_pool,
                                                                                      self.current.port_pool,
                                                                                      self.current.parent,
                                                                                      self.current.comp2port_mapping)
        key = key_circuit_from_lists(list_of_edge, list_of_node, netlist)
        # isom_str_set = instance_to_isom_str(self.get_state())
        # topo_info = str(isom_str_set[0]) + "|" + str(isom_str_set[1])

        # graph_info = str(self.get_state().port_pool) + "|" + str(sort_dict_string(self.current.graph))
        # encode_graph_info = hash_str_encoding(graph_info)
        # to isom, relpece graph_2_reward[key] woth ...key and the next line
        if self.graph_2_reward.__contains__(key):
            efficiency = self.graph_2_reward[key][1]
            vout = self.graph_2_reward[key][2]
            effi = {'efficiency': efficiency, 'output_voltage': vout}
            tmp_reward = calculate_reward(effi, self.configs_['target_vout'], self.configs_['min_vout'],
                                          self.configs_['max_vout'])

            self.current.parameters = self.graph_2_reward[key][0]
            self.reward = tmp_reward
            self.hash_counter += 1
            # print("find in hash------------------------------------------------------")
        else:
            if key in self.key_sim_effi_:
                # key=key,
                # value=[list_of_node, list_of_edge, effi, vout, para, target_ratio]

                tmp_effi = self.key_sim_effi_[key][2]
                tmp_vout = self.key_sim_effi_[key][3]
                effi = {'efficiency': tmp_effi, 'output_voltage': tmp_vout}
                tmp_reward = calculate_reward(effi, self.configs_['target_vout'], self.configs_['min_vout'],
                                              self.configs_['max_vout'])
                tmp_para = self.key_sim_effi_[key][4]
            else:
                start_time = datetime.datetime.now()
                tmp_reward, effi_info, tmp_para = self.current_sim_reward()
                print("find_one_simulate_result using: ", str((datetime.datetime.now() - start_time).seconds))
                if tmp_para != -1:
                    tmp_effi = effi_info['efficiency']
                    tmp_vout = effi_info['Vout']
                    # self.graph_2_reward[key] = [tmp_para, tmp_effi, tmp_vout]
                else:
                    tmp_para = 'None'
                    tmp_reward = 0
                    tmp_effi = 0
                    tmp_vout = -500
                # TODO the 100 here is the hard code vin
                self.key_sim_effi_[key] = [list_of_node, list_of_edge, tmp_effi, tmp_vout,
                                           tmp_para, self.configs_['target_vout'] / 100]
            self.graph_2_reward[key] = [tmp_para, tmp_effi, tmp_vout]

            self.reward = tmp_reward
            self.current.parameters = tmp_para
            self.query_counter += 1

            if (self.current_max == {}) or self.current_max['reward'] < self.reward:
                self.current_max['state'] = self.current.duplicate()
                self.current_max['reward'] = self.reward
                self.current_max['para'] = self.current.parameters

            def keyFunc(element):
                return element[1]

            if self.current.parameters == 'None':
                self.topk.append([self.current.duplicate(), self.reward, None])
            else:
                self.topk.append([self.current.duplicate(), self.reward, float(self.current.parameters)])
            if len(self.topk) > 5:
                self.topk.sort(key=keyFunc)
                self.topk.pop(0)
        return self.reward

    def get_max_seen(self):
        return self.current_max['state'], self.current_max['reward'], self.current_max['para']

    def get_effi_info(self):

        highest_reward = None
        tmp_reward = None
        if not self.current.graph_is_valid():
            self.current.parameters = 'None'
            self.reward = 0
            return {'parameter': 'None', 'efficiency': 0, 'output_voltage': -500}

        list_of_node, list_of_edge, netlist, joint_list = gen_topo.convert_to_netlist(self.current.graph,
                                                                                      self.current.component_pool,
                                                                                      self.current.port_pool,
                                                                                      self.current.parent,
                                                                                      self.current.comp2port_mapping)
        key = key_circuit_from_lists(list_of_edge, list_of_node, netlist)
        # graph_info = str(self.get_state().port_pool) + "|" + str(sort_dict_string(self.current.graph))
        # encode_graph_info = hash_str_encoding(graph_info)

        if self.graph_2_reward.__contains__(key):
            efficiency = self.graph_2_reward[key][1]
            vout = self.graph_2_reward[key][2]
            effi = {'efficiency': efficiency, 'output_voltage': vout}
            para = self.graph_2_reward[key][0]
            effis = {'parameter': para, 'efficiency': efficiency, 'output_voltage': vout}
            return effis
        else:
            para_result = find_one_analytics_result(key, self.key_expression, self.current.graph,
                                                    self.current.comp2port_mapping,
                                                    self.current.port2comp_mapping, self.current.idx_2_port,
                                                    self.current.port_2_idx,
                                                    self.current.parent, self.current.component_pool,
                                                    self.current.same_device_mapping, self.current.port_pool)
            if para_result:
                tmp_reward = 0
                tmp_para = 'None'
                if 'None' in para_result:
                    self.graph_2_reward[key] = ['None',
                                                self.key_expression[key + '$' + 'Invalid']['Effiency'],
                                                self.key_expression[key + '$' + 'Invalid']['Vout']]
                    tmp_reward = 0
                    tmp_para = 'None'
                    tmp_effi = self.key_expression[key + '$' + 'Invalid']['Effiency']
                    tmp_vout = self.key_expression[key + '$' + 'Invalid']['Vout']
                    effis = {'parameter': tmp_para, 'efficiency': tmp_effi, 'output_voltage': tmp_vout}

                else:
                    for k, v in para_result.items():
                        # print(k, v)
                        effi = {'efficiency': float(int(v[0]) / 100), 'output_voltage': float(v[1])}
                        if tmp_reward <= calculate_reward(effi, self.configs_['target_vout'], self.configs_['min_vout'],
                                                          self.configs_['max_vout']):
                            tmp_reward = calculate_reward(effi, self.configs_['target_vout'], self.configs_['min_vout'],
                                                          self.configs_['max_vout'])
                            tmp_para = k
                            tmp_effi = float(int(v[0]) / 100)
                            tmp_vout = float(v[1])
                    # We only save the highest rewards' parameter, effi and cout(vout)
                    self.graph_2_reward[key] = [tmp_para, tmp_effi, tmp_vout]
                    effis = {'parameter': tmp_para, 'efficiency': tmp_effi, 'output_voltage': tmp_vout}
                self.reward = tmp_reward
                self.current.parameters = tmp_para

            else:
                tmp_reward = 0
                tmp_para = 'None'
                para_result = get_one_analytics_result(self.key_expression, self.current.graph,
                                                       self.current.comp2port_mapping,
                                                       self.current.port2comp_mapping, self.current.idx_2_port,
                                                       self.current.port_2_idx,
                                                       self.current.parent, self.current.component_pool,
                                                       self.current.same_device_mapping, self.current.port_pool,
                                                       self.configs_['min_vout'])
                if para_result:
                    if 'None' in para_result:
                        self.graph_2_reward[key] = ['None',
                                                    self.key_expression[key + '$' + 'Invalid']['Effiency'],
                                                    self.key_expression[key + '$' + 'Invalid']['Vout']]
                        self.current.parameters = 'None'
                        tmp_reward = 0
                        tmp_para = 'None'
                        tmp_effi = self.key_expression[key + '$' + 'Invalid']['Effiency']
                        tmp_vout = self.key_expression[key + '$' + 'Invalid']['Vout']
                    else:
                        for k, v in para_result.items():
                            effi = {'efficiency': float(int(v[0]) / 100), 'output_voltage': float(v[1])}
                            if tmp_reward <= calculate_reward(effi, self.configs_['target_vout'],
                                                              self.configs_['min_vout'],
                                                              self.configs_['max_vout']):
                                tmp_reward = calculate_reward(effi, self.configs_['target_vout'],
                                                              self.configs_['min_vout'],
                                                              self.configs_['max_vout'])
                                tmp_para = k
                                tmp_effi = float(int(v[0]) / 100)
                                tmp_vout = float(v[1])
                        self.graph_2_reward[key] = [tmp_para, tmp_effi, tmp_vout]
                    effis = {'parameter': tmp_para, 'efficiency': tmp_effi, 'output_voltage': tmp_vout}
                else:
                    tmp_reward = 0
                    tmp_para = 'None'
                    effis = {'parameter': 'None', 'efficiency': 0, 'output_voltage': -500}
                self.query_counter += 1
                self.reward = tmp_reward
                self.current.parameters = tmp_para
        return effis

    def replace_component_name(self):
        tmp_current = deepcopy(self.current)
        for node in tmp_current.component_pool:
            node.replace("L", "inductor-")
            node.replace("C", "capacitor-")
            node.replace("Sa", "FET-A-")
            node.replace("Sb", "FET-B-")
        for port in tmp_current.port_pool:
            port.replace("L", "inductor-")
            port.replace("C", "capacitor-")
            port.replace("Sa", "FET-A-")
            port.replace("Sb", "FET-B-")
        for k, v in tmp_current.port_2_idx.items():
            k.replace("L", "inductor-")
            k.replace("C", "capacitor-")
            k.replace("Sa", "FET-A-")
            k.replace("Sb", "FET-B-")
        for k, v in tmp_current.idx_2_port.items():
            v.replace("L", "inductor-")
            v.replace("C", "capacitor-")
            v.replace("Sa", "FET-A-")
            v.replace("Sb", "FET-B-")
        return tmp_current

    # def get_performance_from_analysis(self):
    #     current = self.get_state()
    #     return effi, param

    def supplement_components(self):
        while len(self.current.component_pool) < self.num_component_:
            actions = self.get_actions()
            act_ID = int(random.random() * len(actions))
            r = self.act(actions[act_ID])

    def current_checking_port(self):
        e1 = (self.current.step - (len(self.current.component_pool) - len(['GND', 'VIN', 'VOUT'])))
        e1 %= len(self.current.port_pool)
        return e1

    def component_set_in_current(self, component_set):
        """
        add counts on current path's components and find whether the are exist in current sub graph
        :param component_set: components without count in path
        :return: components with count in path
        """
        count_set = {"Sa": 0, "Sb": 0, "C": 0, "L": 0}
        for idx in range(len(component_set)):
            if component_set[idx] in ['GND', 'VIN', 'VOUT']:
                continue
            else:
                count_set[component_set[idx]] += 1
                # in analytics, no - before count
                component_set[idx] = component_set[idx] + str(count_set[component_set[idx]] - 1)
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
                    # else:
                    #     if (not considerable_skip) or \
                    #             len(need_add_list) < len(considerable_skip[0]):
                    #         if port_0 > 2:
                    #             considerable_skip = [need_add_list, [-1, -1]]
                else:
                    continue
        if considerable_path_edge:
            return considerable_path_edge[0], considerable_path_edge[1]
        elif considerable_skip:
            return considerable_skip[0], considerable_skip[1]
        else:
            return None, []

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
        ''' 'VIN - Sa - L - GND' '''
        components = path.split(' - ')
        ports_idx = []
        ''' components: [VIN, Sa, L, GND] '''
        components_with_count = self.component_set_in_current(components)
        ''' components_with_count: [VIN, Sa0, L0, GND] '''
        if not components_with_count:
            return False, []
        elif not find_roots(allowed_root_pair, [components[0], components[-1]]):
            return False, []
        else:
            """
            path port_list: 
                [Sa0-left,Sa0-right, L-0-left, L-0-right]
                [Sa0-right,Sa0-left, L-0-left, L-0-right]
                [Sa0-left,Sa0-right, L-0-right, L-0-left]
                [Sa0-right,Sa0-left, L-0-right, L-0-left]
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
                # else:
                #     if port_0 > 2:
                #         path_adding_cost.append((need_add_edge_list, [-1, -1]))
            else:
                continue
        if path_adding_cost:
            edge_to_add = generate_port_path_add(path_adding_cost)
            return True, edge_to_add
        return False, []

    def generate_path_weight(self, allowed_root_pair, port_0, allowed_actions):
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
            edge_with_probability[path_idx] = (edge_to_add, self.approved_path_freq[path_idx][1])
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

    def find_path_in_approved_path(self, path):
        for path_freq in self.approved_path_freq:
            if path == path_freq[0]:
                return True
        return False

    def find_not_connected_roots(self):
        """
        find the not connected root pairs, we just allow using approve path to connect one path between
        each pair of roots
        :return: list of root pair that not connected,
        """
        allowed_root_pair = [['VIN', 'VOUT'], ['VOUT', 'GND'], ['VIN', 'GND']]
        current_state = self.get_state()
        list_of_node, list_of_edge, netlist, joint_list = \
            gen_topo.convert_to_netlist(current_state.graph,
                                        current_state.component_pool,
                                        current_state.port_pool,
                                        current_state.parent,
                                        current_state.comp2port_mapping)
        paths = gen_topo.find_paths_from_edges(list_of_node, list_of_edge)
        for path in paths:
            # print(path)
            if self.find_path_in_approved_path(path):
                # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$built a good path:", path)
                path_comp = path.split(' - ')
                allowed_root_pair = remove_roots(allowed_root_pair, [path_comp[0], path_comp[-1]])
                if not allowed_root_pair:
                    return []
        return allowed_root_pair

    def generate_pair_prob(self):
        """
        generate pairs probs, the pairs that has already connected with good path has a lower prob,
        which will be less considered in the good-path selection process for generate the weights
        for edges
        :return: root pairs and prob
        """
        allowed_root_pair = {('VIN', 'VOUT'): 1, ('VOUT', 'GND'): 1, ('VIN', 'GND'): 1}
        current_state = self.get_state()
        list_of_node, list_of_edge, netlist, joint_list = \
            gen_topo.convert_to_netlist(current_state.graph,
                                        current_state.component_pool,
                                        current_state.port_pool,
                                        current_state.parent,
                                        current_state.comp2port_mapping)
        paths = gen_topo.find_paths_from_edges(list_of_node, list_of_edge)
        for path in paths:
            # print(path)
            if self.find_path_in_approved_path(path):
                # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$built a good path:", path)
                path_comp = path.split(' - ')
                # allowed_root_pair = remove_roots(allowed_root_pair, [path_comp[0], path_comp[-1]])
                # allowed_root_pair[[path_comp[0], path_comp[-1]]] = 0
                # allowed_root_pair[[path_comp[0], path_comp[-1]]] = 1
                allowed_root_pair[(path_comp[0], path_comp[-1])] = 0.5
        return allowed_root_pair

    def sample_from_paths(self, paths_with_weights):
        assert paths_with_weights
        edges = []
        edge_weights = []
        for path_idx, edge_and_prob in paths_with_weights.items():
            if edge_and_prob[0] in edges:
                edge_weights[edges.index(edge_and_prob[0])] += edge_and_prob[1]
            else:
                edges.append(edge_and_prob[0])
                edge_weights.append(self.configs_["DP_path_basic_weight"] + edge_and_prob[1])
        for action in self.get_actions():
            if action.value not in edges:
                edges.append(action.value)
                # TODO update new weight assignment method
                edge_weights.append(self.configs_["DP_path_basic_weight"])
        # for i in range(len(edges)):
        #     print(edges[i], ':', edge_weights[i])
        edge = random.choices(edges, weights=edge_weights, k=1)
        # print('add:', edge)
        return edge[0]

    def get_action_using_default_policy(self, allowed_actions, dp_for_component, dp_for_path):
        """
        get an action that (1) can be add in current topology according to approve path set(find a path which can be add
        into current topology, then return the edge which is (2) allowed in current round
        :param allowed_actions: the allowed action set in current round. for example, we want to add the edges one of
        whose end is 3, then all the allowed actions are [3. x] or [-1, -1]
        :return: action(edge, [port_0, port_1])
        """
        # print("----------------------------------")
        if not self.reach_component_number():
            if dp_for_component:
                action = sample_component(self.current.component_pool, self.component_condition_prob_, allowed_actions,
                                          self.basic_components)
            else:
                # act_ID = int(random.random() * len(allowed_actions))
                if len(self.act_vect) != len(self.weights):
                    print(len(self.act_vect), len(self.weights))
                choose_act = random.choices(self.act_vect, weights=self.weights, k=1)
                action = choose_act[0]
            return action
        else:
            # for action in allowed_actions:
            #     print(action.value, end=',')
            # print()
            if dp_for_path:
                # allowed_root_pair = self.find_not_connected_roots()
                allowed_root_pair = self.generate_pair_prob()
                # print('allowed root pairs:', allowed_root_pair)
                if not allowed_root_pair:
                    return None
                port_0 = self.current_checking_port()
                edges_with_weights = self.generate_path_weight(allowed_root_pair, port_0, allowed_actions)
                # print(edges_with_weights)
                if not edges_with_weights:
                    return None
                edge_to_add = self.sample_from_paths(edges_with_weights)
                if edge_to_add == 0:
                    return TopoGenAction('terminal', 0)
                return TopoGenAction('edge', edge_to_add)
            else:
                if len(self.act_vect) != len(self.weights):
                    print(len(self.act_vect), len(self.weights))
                choose_act = random.choices(self.act_vect, weights=self.weights, k=1)
                action = choose_act[0]
                return action

        return None

    # def default_policy(self, mc_return, gamma, discount, final_return, reward_list, dp_for_component, dp_for_path):
    #     while not self.is_terminal():
    #         allowed_actions = self.get_actions()
    #         action = self.get_action_using_default_policy(allowed_actions, dp_for_component, dp_for_path)
    #         if action is None:
    #             act_ID = int(random.random() * len(allowed_actions))
    #             action = allowed_actions[act_ID]
    #             # print("random action to add:", action.value)
    #         r = self.act(action)
    #         reward_list.append(r)
    #         mc_return += discount * r
    #         if not final_return or (final_return < mc_return):
    #             final_return = mc_return
    #         discount *= gamma
    #     self.get_state()
    #     if not final_return:
    #         final_return = 0
    #     return final_return

    def default_policy(self, mc_return, gamma, discount, final_return, reward_list, dp_for_component, dp_for_path,
                       reward_method='analytics'):
        while not self.is_terminal():
            allowed_actions = self.get_actions()
            action = self.get_action_using_default_policy(allowed_actions, dp_for_component, dp_for_path)
            if action is None:
                if len(self.act_vect) != len(self.weights):
                    print(len(self.act_vect), len(self.weights))
                choose_act = random.choices(self.act_vect, weights=self.weights, k=1)
                action = choose_act[0]
                # print("random action to add:", action.value)
            r = self.act(action, reward_method)
            reward_list.append(r)
            mc_return += discount * r
            # if not final_return or (final_return < mc_return):
            #     final_return = mc_return
            discount *= gamma
        # TODO invalid??
        return mc_return

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

    def change_node(self, pre_component_id, changed_node_id):
        # count and replace, nothing else
        pre_component = self.current.component_pool[pre_component_id]
        pre_ports_idx = self.current.comp2port_mapping[pre_component_id]
        pre_left_port_index = pre_ports_idx[0]
        pre_right_port_index = pre_ports_idx[1]
        pre_component_type = get_component_type(pre_component)
        changed_component_type = self.basic_components[changed_node_id]
        # we need to adjust the suffix(index of adding) of  the effected component types
        # delete one: count -1
        print('pre ', self.current.component_pool)
        print('pre ', self.current.port_2_idx)
        print('pre ', self.current.count_map)
        print('pre ', pre_component)

        self.current.count_map[pre_component_type] -= 1
        pre_component_suffix = get_component_count_suffix(pre_component)

        # add one: count +=1
        count = str(self.current.count_map[changed_component_type])
        self.current.count_map[changed_component_type] += 1
        changed_component = self.basic_components[changed_node_id] + str(count)
        print('changed ', changed_component)
        self.current.component_pool[pre_component_id] = changed_component

        self.current.port_pool[pre_left_port_index] = changed_component + '-left'
        self.current.port_pool[pre_right_port_index] = changed_component + '-right'
        self.current.idx_2_port[pre_left_port_index] = changed_component + '-left'
        self.current.idx_2_port[pre_right_port_index] = changed_component + '-right'
        self.current.port_2_idx[changed_component + '-left'] = \
            self.current.port_2_idx.pop(pre_component + '-left')
        self.current.port_2_idx[changed_component + '-right'] = \
            self.current.port_2_idx.pop(pre_component + '-right')

        # find all the components whose type same as pre_component_type and
        # suffix(count idx) larger than pre_component and -1 for there components' suffix
        # example: L0,L1,L2, if delete L1, we just need to change L2 to L1， no need to change L0
        tmp_change_set = []
        for i in range(len(self.current.component_pool)):
            component = self.current.component_pool[i]
            if i != pre_component_id and get_component_type(component) == pre_component_type:
                suffix = get_component_count_suffix(component)
                if suffix > pre_component_suffix:
                    # replace component's suffix
                    print()
                    tmp_change_set.append([component, suffix, i])

        def keyFunc(element):
            return element[1]

        tmp_change_set.sort(key=keyFunc)
        for t in range(len(tmp_change_set)):
            component = tmp_change_set[t][0]
            suffix = tmp_change_set[t][1]
            i = tmp_change_set[t][2]
            smaller_component = get_component_type(component) + str(suffix - 1)
            print(component, smaller_component)
            self.current.component_pool[i] = smaller_component
            # replace ports' suffix
            ports = self.current.comp2port_mapping[i]
            self.current.port_pool[ports[0]] = self.current.port_pool[ports[0]].replace(
                component, smaller_component)
            self.current.port_pool[ports[1]] = self.current.port_pool[ports[1]].replace(
                component, smaller_component)
            self.current.idx_2_port[ports[0]] = smaller_component + '-left'
            self.current.idx_2_port[ports[1]] = smaller_component + '-right'
            if i == len(tmp_change_set) - 1:
                self.current.port_2_idx[smaller_component + '-left'] = \
                    self.current.port_2_idx.pop(component + '-left')
                self.current.port_2_idx[smaller_component + '-right'] = \
                    self.current.port_2_idx.pop(component + '-right')
            else:
                self.current.port_2_idx[smaller_component + '-left'] = \
                    self.current.port_2_idx[component + '-left']
                self.current.port_2_idx[smaller_component + '-right'] = \
                    self.current.port_2_idx[component + '-right']

        print('changed ', self.current.component_pool)
        print('changed ', self.current.port_2_idx)
        print('changed ', self.current.count_map)
        return

    def delete_edge(self, edge):
        # remove edge 0 and edge 1 from graph
        self.current.graph[edge[0]].remove(edge[1])
        self.current.graph[edge[1]].remove(edge[0])
        if not self.current.graph[edge[0]]:
            self.current.graph.pop(edge[0])
        if not self.current.graph[edge[1]]:
            self.current.graph.pop(edge[1])
        root_port = self.current.parent[edge[0]]
        # print("e0 and root,e2 and root", edge[0], self.current.parent[edge[0]], edge[1], self.current.parent[edge[1]])
        idx_graph = self.current.get_idx_graph()
        # self.current.step -= 1
        # for i in range(len(self.current.parent)):
        #     print(i, nx.has_path(idx_graph, i, root_port))
        '''
        if 0, 1 and 2r(root port) are connected together, three possible ways(0-1 had an edge):
        0-1, 0-r: delete 0-1, 1 and r are not connected, so the ports connected with r through 1 change root to 1  
        0-1, 1-r: delete 0-1, 0 and r are not connected, so the ports connected with r through 0 change root to 0
        0-1, 0-r, 1-r: impossible, as in update_action_set we prohibit adding edge between indirectly connected ports
        '''
        if nx.has_path(idx_graph, edge[0], root_port):
            for i in range(len(self.current.parent)):
                if self.current.parent[i] == root_port:
                    if not nx.has_path(idx_graph, i, root_port):
                        self.current.parent[i] = edge[1]
        elif nx.has_path(idx_graph, edge[1], root_port):
            for i in range(len(self.current.parent)):
                if self.current.parent[i] == root_port:
                    if not nx.has_path(idx_graph, i, root_port):
                        self.current.parent[i] = edge[0]
        else:
            print("delete edge wrongly")

    def find_new_edges(self, e1):
        # print(e1)
        edge_weight_list = []
        self.act_vect.clear()

        connected_set = find_connected_set(e1, self.current.parent)
        # if connected_set != [e1], it means e1 has connected with some other ports,
        # so we can skip, else we can not skip as we only allowed small port to connect large port
        if connected_set != [e1]:
            self.act_vect.append(TopoGenAction('edge', [-1, -1]))
        # self.act_vect.append(TopoGenAction('edge', [-1, -1]))
        # TODO to let the ground search first
        # if e1 >= len(self.current.port_pool):
        #     return
        # all the available edge set with e1 as a node
        e2_pool = list(range(len(self.current.port_pool)))
        random.shuffle(e2_pool)

        for e2 in e2_pool:
            # the same port
            # if e1 == e2:
            #     continue

            # if e1 >= e2:
            #     continue
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
                if self.edge_lead_to_prohibit_path([e1, e2]):
                    continue
            # TODO need to be careful about the two layer shuffle in this function,
            #  will cover following method
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

            self.act_vect.append(TopoGenAction('edge', [e1, e2]))
        # It means, we can not skip e1, but we do not have allowed port to connect, what we can do is terminal
        if not self.act_vect:
            print("has to be unconnected")
            self.act_vect.append(TopoGenAction('terminal', 0))

        random.shuffle(self.act_vect)
        if len(self.act_vect) == 1:
            self.weights = [1]
        else:
            self.weights = [1 for i in range(len(self.act_vect))]
            for i in range(len(self.act_vect)):
                if self.act_vect[i].equal(TopoGenAction('edge', [-1, -1])):
                    self.weights[i] = (0.8 / (1 - 0.8)) * (len(self.act_vect) - 1)
        return

    def change_edge(self, e1, e2):
        self.delete_edge([e1, e2])
        print("Graph after delete edge", self.current.graph)
        print("parent after delete edge", self.current.parent)
        self.find_new_edges(e2)
        for i in range(len(self.act_vect)):
            self.act_vect[i].print()
        if len(self.act_vect) == 1:
            warnings.warn("The deleted edge is added into the graph again")
        if len(self.act_vect) != len(self.weights):
            print(len(self.act_vect), len(self.weights))
        choose_act = random.choices(self.act_vect, weights=self.weights, k=1)
        action = choose_act[0]
        print("selected action from adding the edge in mutation")
        self.act(action, 'analytics', False)
        return

    def mutate(self, action_porbs=None):
        if action_porbs is None:
            action_porbs = [0.1, 0.4, 0.2, 0.2]
        reward = 0
        action_to_take = random.choices(['change node', 'change edge', 'delete edge', 'add edge'],
                                        weights=action_porbs, k=1)[0]
        if action_to_take == 'change node':
            # cannot change vin vout gnd
            while True:
                changing_node_idx = int(random.random() * (len(self.current.component_pool) - 3)) + 3
                changing_node = self.current.component_pool[changing_node_idx]
                target_type_idx = int(random.random() * len(self.basic_components))
                target_type = self.basic_components[target_type_idx]
                if target_type not in changing_node:
                    # Same type means no change, we need to choose another one
                    break
            self.change_node(changing_node_idx, target_type_idx)
            reward = self.get_reward()
            change = "change node:" + str(changing_node) + " to " + str(target_type)
            return self.current, reward, change
        elif action_to_take == 'change edge':
            e1 = random.choice(list(self.current.graph.keys()))
            if len(self.current.graph[e1]) <= 0:
                return None, -1, None
            e2 = self.current.graph[e1].pop()
            self.current.graph[e1].add(e2)
            # if (e1 == 0 and e2 == 4) or (e1 == 4 and e2 == 0):
            #     return None, -1, None
            self.change_edge(e1, e2)
            reward = self.get_reward()
            change = "change edge:" + str(e1) + " and " + str(e2)
        elif action_to_take == 'delete edge':
            if not self.current.graph_is_valid():
                return None, -1, None
            e1 = random.choice(list(self.current.graph.keys()))
            e2 = self.current.graph[e1].pop()
            self.current.graph[e1].add(e2)
            self.delete_edge([e1, e2])
            reward = self.get_reward()
            change = "delete edge:" + str(e1) + " and " + str(e2)
        elif action_to_take == 'add edge':
            if self.is_terminal():
                return None, -1, None
            e1 = self.current.step - (len(self.current.component_pool) - 3)
            e1 %= len(self.current.port_pool)
            current_edge = e1
            while True:
                self.find_new_edges(e1)
                if len(self.act_vect) == 1 and self.act_vect[0].value == [-1, -1]:
                    e1 += 1
                    e1 %= len(self.current.port_pool)
                    if e1 == current_edge:
                        return None, -1, None
                    continue
                else:
                    if len(self.act_vect) != len(self.weights):
                        print(len(self.act_vect), len(self.weights))
                    choose_act = random.choices(self.act_vect, weights=self.weights, k=1)
                    action = choose_act[0]
                    self.act(action, 'analytics', False)
                    reward = self.get_reward()
                    break
            change = "add edge on port:" + str(e1)
            # TODO random the select process in mutate and print
        return self.current, reward, change

    def get_performance_from_ngspice(self):
        current = self.get_state()
        replaced_state = self.replace_component_name(deepcopy(current))

        topologies = [replaced_state]
        nets_to_ngspice_files(topologies, self.configs_, self.num_component_)
        simulate_topologies(len(topologies), self.num_component_, self.configs_['sys_os'])
        effis = analysis_topologies(self.configs_, len(topologies), self.num_component_)
        effi = effis[0]
        return effi

    def simulate_one_analytics_result(self, analytics_info):
        result_dict = {}
        if analytics_info == None:
            return {0.0: {'efficiency': 0, 'Vout': 0}}
        analytic = analytics_info
        cki_folder = 'database/cki/'
        for fn in analytic:
            print(int(fn[6:9]))
            count = 0
            for param in analytic[fn]:
                param_value = [float(value) for value in param[1:-1].split(', ')]
                device_name = analytic[fn][param][0]
                netlist = analytic[fn][param][1]
                file_name = fn + '-' + str(count) + '.cki'
                convert_cki_full_path(file_name, param_value, device_name, netlist)
                print(file_name)
                simulate(file_name)
                count = count + 1
        data = {}
        data_csv = []
        # only one fn in analytic if only simulate one
        for fn in analytic:
            count = 0
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
                # print(result)
                param = str(param)
                param_spl = param.split(',')
                para_val = param_spl[0].replace('(', '')
                result_dict[para_val] = result
                count = count + 1
            return result_dict

    def current_sim_reward(self):
        if self.current.graph_is_valid():
            results_tmp = get_analytics_file(self.key_expression, self.current.graph,
                                             self.current.comp2port_mapping,
                                             self.current.port2comp_mapping, self.current.idx_2_port,
                                             self.current.port_2_idx,
                                             self.current.parent, self.current.component_pool,
                                             self.current.same_device_mapping, self.current.port_pool,
                                             self.fix_para_set, self.configs_['min_vout'])
        else:
            results_tmp = None
        effi_info = self.simulate_one_analytics_result(results_tmp)
        max_reward, max_effi_info, max_para = self.find_simu_max_reward_para(effi_info)
        return max_reward, max_effi_info, max_para

    def get_tops_sim_info(self):
        max_sim_reward_result = {'max_para': 0, 'max_sim_reward': -1, 'max_sim_effi': 0, 'max_sim_vout': -500}

        for i in range(len(self.topk)):
            current = self.topk[i][0]
            if current.graph_is_valid():
                results_tmp = get_analytics_file(self.key_expression, current.graph,
                                                 current.comp2port_mapping,
                                                 current.port2comp_mapping, current.idx_2_port,
                                                 current.port_2_idx,
                                                 current.parent, current.component_pool,
                                                 current.same_device_mapping, current.port_pool,
                                                 None, self.configs_['min_vout'])
            else:
                results_tmp = None
            list_of_node, list_of_edge, netlist, joint_list = gen_topo.convert_to_netlist(current.graph,
                                                                                          current.component_pool,
                                                                                          current.port_pool,
                                                                                          current.parent,
                                                                                          current.comp2port_mapping)
            current_key = key_circuit_from_lists(list_of_edge, list_of_node, netlist)
            if self.key_sim_effi_.__contains__(current_key):
                effi_info = self.key_sim_effi_[current_key]
                print('find pre simulated &&&&&&&&&&&&&&&&&&&&&&')
            else:
                effi_info = self.simulate_one_analytics_result(results_tmp)
                saved_effi_info = {}
                for para, simu_info in effi_info.items():
                    effi = {'efficiency': simu_info['efficiency'], 'Vout': simu_info['Vout']}
                    saved_effi_info[para] = effi
                self.key_sim_effi_[current_key] = saved_effi_info
            # [state, anal_reward, anal_para, sim_reward, sim_effi, sim_vout, sim_para]
            topk_max_reward = -1
            topk_max_para = -1
            print(effi_info)
            for para, simu_info in effi_info.items():
                effi = {'efficiency': simu_info['efficiency'],
                        'output_voltage': simu_info['Vout']}
                tmp_reward = calculate_reward(effi, self.configs_['target_vout'],
                                              self.configs_['min_vout'],
                                              self.configs_['max_vout'])
                if tmp_reward >= topk_max_reward:
                    topk_max_para = para
                    topk_max_reward = tmp_reward
            self.topk[i].append(topk_max_reward)
            self.topk[i].append(effi_info[topk_max_para]['efficiency'])
            self.topk[i].append(effi_info[topk_max_para]['Vout'])
            self.topk[i].append(topk_max_para)
            if max_sim_reward_result['max_sim_reward'] < topk_max_reward:
                max_sim_reward_result['max_sim_state'] = self.topk[i][0]
                max_sim_reward_result['max_sim_reward'] = topk_max_reward
                max_sim_reward_result['max_sim_effi'] = effi_info[topk_max_para]['efficiency']
                max_sim_reward_result['max_sim_vout'] = effi_info[topk_max_para]['Vout']
                max_sim_reward_result['max_sim_para'] = topk_max_para
        return max_sim_reward_result

    def find_simu_max_reward_para(self, para_effi_info):
        max_reward = -1
        if 'result_valid' in para_effi_info and para_effi_info['result_valid'] == False:
            return 0, {'efficiency': 0, 'Vout': -500}, -1
        for para in para_effi_info:
            effi_info = para_effi_info[para]
            effi = {'efficiency': effi_info['efficiency'], 'output_voltage': effi_info['Vout']}
            tmp_reward = calculate_reward(effi, self.configs_['target_vout'])
            if tmp_reward > max_reward:
                max_reward = tmp_reward
                max_effi_info = deepcopy(effi_info)
                max_para = para
        return max_reward, max_effi_info, max_para

    def get_actions(self):
        return self.act_vect

    def get_weights(self):
        return self.weights

    def is_terminal(self):
        if self.current.step - (len(self.current.component_pool) - len(['GND', 'VIN', 'VOUT'])) >= len(
                self.current.port_pool):
            return True
        return False


def get_comp_types(sim):
    comp_type_set = []
    for component in sim.current.component_pool:
        for comp_type in sim.basic_components:
            if comp_type in component:
                if comp_type not in comp_type_set:
                    comp_type_set.append(comp_type)
    return comp_type_set


def get_type_components_idx(type_p, sim):
    type_components_idx = []
    for i in range(len(sim.current.component_pool)):
        component = sim.current.component_pool[i]
        if type_p in component:
            type_components_idx.append(i)
    return type_components_idx


def choose_left_or_right():
    if random.random() < 0.5:
        return 'left'
    return 'right'


def crossover(sim_0, sim_1):
    comp_type_set_0 = get_comp_types(sim_0)
    comp_type_set_1 = get_comp_types(sim_1)
    common_comp_types = list(set(comp_type_set_0).intersection(set(comp_type_set_1)))
    if not common_comp_types:
        return None, -1, None, None, -1, None

    type_p = random.choice(common_comp_types)
    cv_sim_0_comp_idx = random.choice(get_type_components_idx(type_p, sim_0))
    cv_sim_1_comp_idx = random.choice(get_type_components_idx(type_p, sim_1))

    cv_sim_0_port = min(sim_0.current.comp2port_mapping[cv_sim_0_comp_idx]) if choose_left_or_right() == 'left' \
        else max(sim_0.current.comp2port_mapping[cv_sim_0_comp_idx])
    cv_sim_1_port = min(sim_1.current.comp2port_mapping[cv_sim_1_comp_idx]) if choose_left_or_right() == 'left' \
        else max(sim_1.current.comp2port_mapping[cv_sim_1_comp_idx])

    ports_on_port_0 = list(sim_0.current.graph[cv_sim_0_port])
    edges_on_port_0 = [[cv_sim_0_port, port] for port in ports_on_port_0]
    ports_on_port_1 = list(sim_1.current.graph[cv_sim_1_port])
    edges_on_port_1 = [[cv_sim_1_port, port] for port in ports_on_port_1]

    # Crossover
    for edge in edges_on_port_0:
        sim_0.delete_edge(edge)
    for edge in edges_on_port_1:
        sim_0.add_edge(edge)
    for edge in edges_on_port_1:
        sim_1.delete_edge(edge)
    for edge in edges_on_port_0:
        sim_1.add_edge(edge)

    return sim_0.current, sim_0.get_reward(), \
           "crossover sim_0:" + str(cv_sim_0_port) + " and " + str(cv_sim_1_port), \
           sim_1.current, sim_1.get_reward(), \
           "crossover sim_1:" + str(cv_sim_0_port) + " and " + str(cv_sim_1_port)
