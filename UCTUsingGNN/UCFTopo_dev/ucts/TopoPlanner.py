import random
from ucts import uct
from utils.util import hash_str
import collections
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
from simulator3component.build_topology import nets_to_ngspice_files
from simulator3component.simulation import simulate_topologies
from simulator3component.simulation_analysis import analysis_topologies
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
    graph_dict_str = hash_str(graph_dict_str)
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
        plt.savefig(dt)
        plt.close()

    def get_nodes_and_edges(self):
        """
        :return: (list of nodes, list of edges) of this state
        """
        return convert_to_netlist(self.component_pool, self.port_pool, self.parent, self.comp2port_mapping)


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


class TopoGenSimulator(uct.Simulator):
    def __init__(self, _configs, _num_component=4, target=None):
        # self.target = target
        # self.target_edges = None
        # if target:
        #     self.target_edges = target.get_edges()

        # self.node_reward = 1.0
        # self.node_penalty = -0.0
        # self.edge_reward = 1.0
        # self.edge_penalty = -0.0
        self.basic_components = ["FET-A", "FET-B", "capacitor", "inductor"]
        self.reward = 0
        self.current = TopoGenState(init=True)
        self.configs_ = _configs
        self.num_component_ = _num_component
        self.query_counter = 0
        self.hash_counter = 0
        self.graph_2_reward = {}

        # move to state
        # self.step = 0
        self.act_vect = []

        self.update_action_set()

    def set_state(self, state):
        self.current = state.duplicate()
        self.update_action_set()

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

    def change_node(self, pre_component_id, changed_node_id):
        """
        change a node's type
        pre_component_id: the index of the component that will be changed
        changed_node_id: the index of the type of the target component
        """
        pre_component = self.current.component_pool[pre_component_id]
        pre_ports_idx = self.current.comp2port_mapping[pre_component_id]
        pre_left_port_index = pre_ports_idx[0]
        pre_right_port_index = pre_ports_idx[1]
        pre_component_type = pre_component[:pre_component.rfind('-')]
        # print(pre_component_type)
        self.current.count_map[pre_component_type] -= 1
        self.current.same_device_mapping.pop(self.current.port_2_idx[pre_component + '-left'])
        self.current.same_device_mapping.pop(self.current.port_2_idx[pre_component + '-right'])

        self.current.count_map[self.basic_components[changed_node_id]] += 1
        changed_component = self.basic_components[changed_node_id] + '-' + str(pre_component_id)

        self.current.component_pool[pre_component_id] = changed_component

        self.current.port_pool[pre_left_port_index] = changed_component + '-left'
        self.current.port_pool[pre_right_port_index] = changed_component + '-right'

        self.current.idx_2_port[pre_left_port_index] = changed_component + '-left'
        self.current.idx_2_port[pre_right_port_index] = changed_component + '-right'

        self.current.port_2_idx[changed_component + '-left'] = self.current.port_2_idx.pop(pre_component + '-left')
        self.current.port_2_idx[changed_component + '-right'] = self.current.port_2_idx.pop(pre_component + '-right')
        self.current.same_device_mapping[self.current.port_2_idx[changed_component + '-left']] = \
            self.current.port_2_idx[changed_component + '-right']
        self.current.same_device_mapping[self.current.port_2_idx[changed_component + '-right']] = \
            self.current.port_2_idx[changed_component + '-left']

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

    def delete_edge(self, edge):
        self.current.graph[edge[0]].remove(edge[1])
        self.current.graph[edge[1]].remove(edge[0])
        if not self.current.graph[edge[0]]:
            self.current.graph.pop(edge[0])
        if not self.current.graph[edge[1]]:
            self.current.graph.pop(edge[1])
        root_port = self.current.parent[edge[0]]
        idx_graph = self.current.get_idx_graph()
        self.current.step -= 1
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
        self.act_vect.clear()
        e2_pool = list(range(len(self.current.port_pool)))
        random.shuffle(e2_pool)
        for e2 in e2_pool:
            # the same port
            # if e1 == e2:
            #     continue
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
            vin_root = find(0, self.current.parent)
            vout_root = find(1, self.current.parent)
            gnd_root = find(2, self.current.parent)
            special_roots = [vin_root, vout_root, gnd_root]

            if e1_root in special_roots and e2_root in special_roots:
                continue
            if e1_root == e2_root:
                continue

            # TODO Partial short cut check
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

            self.act_vect.append(TopoGenAction('edge', [e1, e2]))

    def change_edge(self, e1, e2):
        self.delete_edge([e1, e2])
        # print("Graph after delete edge", self.current.graph)
        # print("parent after delete edge", self.current.parent)
        self.find_new_edges(e2)
        if len(self.act_vect) == 0:
            self.act_vect.append(TopoGenAction('edge', [e1, e2]))
        for i in range(len(self.act_vect)):
            self.act_vect[i].print()
        if len(self.act_vect) == 1:
            warnings.warn("The deleted edge is added into the graph again")
        new_action = random.choice(self.act_vect)
        r = self.act(new_action)
        return r

    def random_generate_graph(self):
        while True:
            r = 0
            self.reward = 0
            self.current = TopoGenState(init=True)
            self.act_vect = []
            self.update_action_set()
            # init the components
            init_nodes = [0, 3, 1]
            for e in init_nodes:
                new_action = TopoGenAction('node', e)
                self.act(new_action)
            # init the edges
            '''
            init_edges = [(0, 4)]
            for edge in init_edges:
                action = TopoPlanner.TopoGenAction('edge', edge)
                sim.act(action)
            '''
            while not self.is_terminal():
                rand_act_idx = int(random.random() * len(self.act_vect))
                new_action = self.act_vect[rand_act_idx]
                r = self.act(new_action)
            init_state = self.current
            if init_state.graph_is_valid():
                break
            else:
                continue

    def mutate(self):
        """
        The mutation for genetic search
        """
        choice = random.random()
        print(choice)
        reward = 0
        if choice < -1:
            changed_node = int(random.random() * (len(self.current.component_pool) - 3)) + 3
            target_type = int(random.random() * len(self.basic_components))
            self.change_node(changed_node, target_type)
            reward = self.get_reward()
            change = "change node:" + str(changed_node) + " to " + str(target_type)
            return self.current, reward, change
        elif choice < 0.5:
            e1 = random.choice(list(self.current.graph.keys()))
            if len(self.current.graph[e1]) <= 0:
                return None, -1, None
            e2 = self.current.graph[e1].pop()
            self.current.graph[e1].add(e2)
            # if (e1 == 0 and e2 == 4) or (e1 == 4 and e2 == 0):
            #     return None, -1, None
            reward = self.change_edge(e1, e2)
            change = "change edge:" + str(e1) + " and " + str(e2)
        elif choice < 0.75:
            if not self.current.graph_is_valid():
                return None, -1, None
            e1 = random.choice(list(self.current.graph.keys()))
            e2 = self.current.graph[e1].pop()
            self.current.graph[e1].add(e2)
            self.delete_edge([e1, e2])
            reward = self.get_reward()
            change = "delete edge:" + str(e1) + " and " + str(e2)
        else:
            if self.is_terminal():
                return None, -1, None
            e1 = self.current.step - (len(self.current.component_pool) - 3)
            e1 %= len(self.current.port_pool)
            current_edge = e1
            while True:
                self.find_new_edges(e1)
                if len(self.act_vect) == 0:
                    e1 += 1
                    e1 %= len(self.current.port_pool)
                    if e1 == current_edge:
                        return None, -1, None
                    continue
                else:
                    action = random.choice(self.act_vect)
                    reward = self.act(action)
                    break
            change = "add edge on port:" + str(e1)
        return self.current, reward, change

    def update_action_set(self):
        """
        After action, the topology is changed, we need to find out which edges we can connect
        """
        if len(self.current.component_pool) < self.num_component_:
            self.act_vect = [TopoGenAction('node', i) for i in range(len(self.basic_components))]
        else:
            self.act_vect.clear()
            self.act_vect.append(TopoGenAction('edge', [-1, -1]))
            if self.current.graph_is_valid():
                self.act_vect.append(TopoGenAction('terminal', 0))
            e1 = self.current.step - (len(self.current.component_pool) - 3)

            e1 %= len(self.current.port_pool)
            # TODO to let the ground search first
            if e1 == 0:
                e1 = 2
            elif e1 == 2:
                e1 = 0
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

                self.act_vect.append(TopoGenAction('edge', [e1, e2]))
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

    def act(self, _action):

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

        self.update_action_set()

        self.reward = self.get_reward()
        self.reward = self.reward - parent_reward
        # if self.reward < 0:
        #     self.reward = 0

        return self.reward

    def get_reward(self):
        if not self.current.graph_is_valid():
            self.reward = 0
        elif self.graph_2_reward.__contains__(sort_dict_string(self.current.graph)):
            self.reward = self.graph_2_reward[sort_dict_string(self.current.graph)][0]
            self.hash_counter += 1
        else:
            topologies = [self.get_state()]
            nets_to_ngspice_files(topologies, self.configs_, self.num_component_)
            simulate_topologies(len(topologies), self.num_component_, self.configs_['sys_os'])
            effis = analysis_topologies(self.configs_, len(topologies), self.num_component_)
            effi = effis[0]
            self.query_counter += 1

            if not effi.__contains__('output_voltage'):
                self.reward = 0
                effi['output_voltage'] = 0
            else:
                self.reward = calculate_reward(effi)
            self.graph_2_reward[sort_dict_string(self.current.graph)] = [self.reward,
                                                                         effi['efficiency'], effi['output_voltage']]
            if self.reward > 0.1:
                file_name = "Results/special.txt"
                fo = open(file_name, "w")
                fo.write("================================================" + "\n")
                fo.write(str(self.current.port_pool) + "\n")
                fo.write(str(self.current.graph) + "\n")
                fo.write(str(effi) + "\n")
                fo.close()
        return self.reward

    def get_actions(self):
        return self.act_vect

    def is_terminal(self):
        if self.current.step - (len(self.current.component_pool) - 3) >= len(self.current.port_pool):
            return True
        return False

    def log_info(self, effis, time, fig_dir):
        return {}


def construct_target_v1():
    simulator = TopoGenSimulator(None)
    # ["FET-A", "FET-B", "capacitor", "inductor"]
    simulator.add_node(0)
    simulator.add_node(1)
    simulator.add_node(3)
    simulator.current.init_disjoint_set()
    # edges:
    edges = [['VIN', 'FET-A-0-left'], ['FET-A-0-right', 'FET-B-0-left'], ['FET-B-0-right', 'GND'],
             ['inductor-0-left', 'FET-A-0-right'], ['inductor-0-right', 'VOUT']]
    for edge in edges:
        p1, p2 = simulator.current.port_2_idx[edge[0]], simulator.current.port_2_idx[edge[1]]
        simulator.add_edge([p1, p2])
    return simulator.get_state()

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
