import copy
from abc import abstractmethod, ABC

import config

if config.task == 'uct_3_comp':
    from UCFTopo_dev.ucts.TopoPlanner import TopoGenSimulator, calculate_reward, sort_dict_string
elif config.task == 'uct_5_comp':
    from UCT_5_UCB_unblc_restruct_DP_v1.ucts.TopoPlanner import TopoGenSimulator, calculate_reward, sort_dict_string
    from UCT_5_UCB_unblc_restruct_DP_v1.SimulatorAnalysis.gen_topo import key_circuit_from_lists, convert_to_netlist

from topo_data_util.analysis.topoGraph import TopoGraph
from topo_data_util.analysis.graphUtils import nodes_and_edges_to_adjacency_matrix


class SurrogateRewardTopologySim(TopoGenSimulator, ABC):
    def __init__(self, debug, *args):
        self.debug = debug
        # for fair comparison with simulator, create a hash table here
        self.surrogate_hash_table = {}
        self.no_isom_seen_state_list = []

        super().__init__(*args)

    def find_paths(self):
        """
        Useful for GP and transformer based surrogate model
        Return the list of paths in the current state
        e.g. ['VIN - inductor - VOUT', ...]
        """
        node_list, edge_list = self.get_state().get_nodes_and_edges()

        adjacency_matrix = nodes_and_edges_to_adjacency_matrix(node_list, edge_list)

        # convert graph to paths, and find embedding
        topo = TopoGraph(adj_matrix=adjacency_matrix, node_list=node_list, new_repr=True)
        return topo.find_end_points_paths_as_str()

    def get_topo_key(self, state=None):
        """
        the key of topology used by hash table

        :return:  the key representation of the state (self.current if state == None)
        """
        if state is None:
            state = self.get_state()

        if config.task == 'uct_3_comp':
            topo_key = sort_dict_string(state.graph)
        elif config.task == 'uct_5_comp':
            list_of_node, list_of_edge, netlist, joint_list = convert_to_netlist(state.graph,
                                                                                 state.component_pool,
                                                                                 state.port_pool,
                                                                                 state.parent,
                                                                                 state.comp2port_mapping)
            topo_key = key_circuit_from_lists(list_of_edge, list_of_node, netlist)
        else:
            raise Exception()

        return topo_key

    def get_reward(self, state=None):
        if not self.configs_['sweep']:
            return self.get_no_sweep_reward()
        else:
            return self.get_sweep_reward_with_para()

    def get_no_sweep_reward(self):
        reward = self.get_reward_using_gnn()
        return reward

    def get_sweep_reward_with_para(self):
        tmp_para = -1
        tmp_max_reward = -1
        for duty_cycle in self.candidate_duty_cycles:
            self.current.parameters = duty_cycle
            reward = self.get_reward_using_gnn()
            if tmp_max_reward < reward:
                tmp_max_reward = reward
                tmp_para = duty_cycle
        self.current.parameters = tmp_para
        return tmp_max_reward

    def get_reward_using_gnn(self, state=None):
        """
        Use surrogate reward function
        imp-wise, not sure why keeping a reward attribute
        """

        if state is not None:
            self.set_state(None, None, state)

        if not self.is_terminal():
            self.current.parameters = -1
            self.reward = 0
            return self.reward

        if not self.current.graph_is_valid():
            self.current.parameters = -1
            self.reward = 0
            self.effi = 0
            self.vout = -500
            return self.reward

        topo_key = self.get_topo_key()
        if topo_key + '$' + str(self.current.parameters) in self.surrogate_hash_table:
            self.hash_counter += 1
            return self.surrogate_hash_table[topo_key + '$' + str(self.current.parameters)]
        else:
            if self.configs_['skip_sim'] and \
                    (topo_key + '$' + str(self.current.parameters) not in self.key_sim_effi_):
                reward = 0
                effi_info = {'efficiency': 0, 'Vout': 500}
                tmp_para = self.current.parameters
                eff = effi_info['efficiency']
                vout = effi_info['Vout']
                parameter = self.current.parameters
                print('skip as not in sim hash')
            else:

                # eff = self.get_surrogate_eff(self.get_state())
                # vout = self.get_surrogate_vout(self.get_state())
                eff, vout, reward, parameter = self.get_surrogate_reward(self.get_state())
                # reward_sim, effi_sim, vout_sim = self.get_true_performance(self.get_state())
                # print('gnn effi:', eff, ' vout:', vout, ' reward:', reward)
                # print('simulation effi:', effi_sim, ' vout:', vout_sim, ' reward:', reward_sim)
                # # an object for computing reward
                # eff_obj = {'efficiency': eff,
                #            'output_voltage': vout}

        self.query_counter += 1
        self.reward = reward

        if self.debug:
            print('estimated reward {}, eff {}, vout {}'.format(self.reward, eff, vout))
            print('true performance {}'.format(self.get_true_performance()))

        self.surrogate_hash_table[topo_key + '$' + str(self.current.parameters)] = self.reward
        print(topo_key, eff, vout, reward, parameter)
        if self.configs_['sweep']:
            self.update_topk(topo_key)
        else:
            self.update_topk_topology_with_para(topo_key + '$' + str(self.current.parameters))
        self.no_isom_seen_state_list.append(copy.deepcopy(self.current))

        return self.reward

    @abstractmethod
    def get_surrogate_eff(self, state):
        """
        return the eff prediction of state, and of self.get_state() if None
        """
        pass

    @abstractmethod
    def get_surrogate_vout(self, state):
        """
        return the vout prediction of state, and of self.get_state() if None
        """
        pass

    def get_surrogate_reward(self, state):
        """
        return the vout prediction of state, and of self.get_state() if None
        """
        pass

    def get_true_performance(self, state=None):
        if not self.configs_['sweep']:
            return self.get_no_sweep_true_performance(state)
        else:
            return self.get_sweep_true_performance_with_para(state)

    def get_no_sweep_true_performance(self, state=None):
        reward, eff, vout = self.get_true_performance_of_sim(state)
        return reward, eff, vout

    def get_sweep_true_performance_with_para(self, state=None):
        tmp_para = -1
        tmp_max_reward = -1
        tmp_max_eff = -1
        tmp_max_vout = -500
        for duty_cycle in self.candidate_duty_cycles:
            state.parameters = duty_cycle
            reward, eff, vout = self.get_true_performance_of_sim(state)
            if tmp_max_reward < reward:
                tmp_max_reward = reward
                tmp_para = duty_cycle
                tmp_max_eff = eff
                tmp_max_vout = vout
        self.current.parameters = tmp_para
        return tmp_max_reward, tmp_max_eff, tmp_max_vout

    def get_true_performance_of_sim(self, state):
        # call the file
        # TODO forget to deal with the sweep!
        """
        :return: [reward, eff, vout]
        """
        if state is not None:
            self.set_state(None, None, state)
        else:
            return [0, -1, -500]

        if not self.current.graph_is_valid():
            return [0, -1, -500]


        hash = self.get_topo_key()

        # if not in hash table, call ngspice
        if hash + '$' + str(state.parameters) not in self.graph_2_reward.keys():
            if hash + '$' + str(state.parameters) in self.key_sim_effi_:
                eff = self.key_sim_effi_[hash + '$' + str(state.parameters)][0]
                vout = self.key_sim_effi_[hash + '$' + str(state.parameters)][1]
                effi = {'efficiency': eff, 'output_voltage': vout}
                reward = calculate_reward(effi, self.configs_['target_vout'], self.configs_['min_vout'],
                                          self.configs_['max_vout'])
            else:
                if self.configs_['skip_sim']:
                    reward = 0
                    eff = 0
                    vout = 500
                else:
                    reward, eff, vout, para = super().get_single_topo_sim_result(state)

            self.graph_2_reward[hash + '$' + str(state.parameters)] = [reward, eff, vout]
            return reward, eff, vout

        else:
            if config.task == 'uct_3_comp' or config.task == 'rs_3_comp':
                return self.graph_2_reward[hash + '$' + str(state.parameters)]
            elif config.task == 'uct_5_comp':
                para, eff, vout = self.graph_2_reward[hash + '$' + str(state.parameters)]

                eff_obj = {'efficiency': eff,
                           'output_voltage': vout}
                reward = calculate_reward(eff_obj, self.configs_['target_vout'], self.configs_['min_vout'],
                                          self.configs_['max_vout'])

                return reward, eff, vout

    def get_true_reward(self, state=None):
        return self.get_true_performance(state)[0]
