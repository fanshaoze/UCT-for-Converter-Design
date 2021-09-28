import json
import os

import torch

from PM_GNN.code.topo_data import split_balance_data, Autopo
from topo_envs.surrogateRewardSim import SurrogateRewardTopologySim
from UCT_5_UCB_unblc_restruct_DP_v1.ucts.TopoPlanner import TopoGenState, calculate_reward
from PM_GNN.code.generate_dataset import generate_topo_for_GNN_model
from PM_GNN.code.ml_utils import get_output_with_model, initialize_model


class GNNRewardSim(SurrogateRewardTopologySim):
    def __init__(self, eff_model_file, vout_model_file, debug=False, *args):
        super().__init__(debug, *args)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.configs_['round'] == 'vout':
            self.eff_y_select = 'reg_eff5'
            self.vout_y_select = 'cls_buck5'
        else:
            self.eff_y_select = 'reg_eff5'
            self.vout_y_select = 'reg_vout5'
        # self.eff_y_select = 'reg_eff-old'
        # self.vout_y_select = 'reg_vout-old'
        self.eff_model = self.load_model(self.eff_y_select)
        self.vout_model = self.load_model(self.vout_y_select)
        self.raw_dataset_file = 'raw_dataset.json'
        self.reg_data_folder = './PM_GNN/2_dataset/'

        self.num_node = 4
        self.batch_size = 1

    def load_model(self, y_select):
        # device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model = initialize_model(model_index=1, gnn_nodes=100, pred_nodes=100, gnn_layers=3,
        #                          nf_size=4, ef_size=3, device=device_)
        # model = initialize_model(model_index=1, gnn_nodes=100, pred_nodes=100, gnn_layers=3,
        #                          nf_size=4, ef_size=3, device=self.device)
        # model = initialize_model(model_index=1, gnn_nodes=100, pred_nodes=100, gnn_layers=4,
        #                          nf_size=4, ef_size=3, device=self.device) # for pre reg_eff/vout.pt
        model = initialize_model(model_index=1, gnn_nodes=100, pred_nodes=100, gnn_layers=6,
                                 nf_size=4, ef_size=3, device=self.device)

        pt_filename = y_select + '.pt'
        if os.path.exists(pt_filename):
            print('loading model from pt file', y_select)
            model_state_dict, _ = torch.load(pt_filename)
            model.load_state_dict(model_state_dict)
        return model

    def get_surrogate_reward(self, state: TopoGenState):
        if state.parameters == -1:
            return -1, -500, 0, -1
        os.system('rm ' + self.reg_data_folder + self.eff_y_select + '/processed/data.pt')
        os.system('rm ' + self.reg_data_folder + self.vout_y_select + '/processed/data.pt')
        raw_dataset = generate_topo_for_GNN_model(state)
        self.save_dataset_to_file(raw_dataset)
        out_effi_list = []
        out_vout_list = []
        print(len(raw_dataset))
        # get_effciencies
        dataset = Autopo(self.raw_dataset_file, self.reg_data_folder + self.eff_y_select, self.eff_y_select)
        print(len(dataset))
        # _, _, test_loader = split_balance_data(dataset, False, batch_size=self.batch_size)
        # print(len(test_loader))
        test_loader = dataset
        for data in test_loader:
            out_effi = get_output_with_model(data, self.eff_model, self.device,
                                             num_node=self.num_node, model_index=1)
            out_effi_list.append(out_effi)

        dataset = Autopo(self.raw_dataset_file, self.reg_data_folder + self.vout_y_select, self.vout_y_select)
        # _, _, test_loader = split_balance_data(dataset, False, batch_size=self.batch_size)
        test_loader = dataset
        for data in test_loader:
            out_vout = get_output_with_model(data, self.vout_model, self.device,
                                             num_node=self.num_node, model_index=1)
            out_vout_list.append(out_vout)
        # print(out_effi_list, out_vout_list)

        # print(out_effi_list)
        # print(out_vout_list)
        eff = out_effi_list[0][0][0]
        vout = 100 * out_vout_list[0][0][0]
        eff_obj = {'efficiency': float(eff),
                   'output_voltage': float(vout)}
        if self.configs_['round'] == 'vout':
            reward = float(eff) * float(vout)/100
        else:
            reward = calculate_reward(eff_obj, self.configs_['target_vout'],
                                      self.configs_['min_vout'],
                                      self.configs_['max_vout'])
        os.system('rm ' + self.reg_data_folder + self.eff_y_select + '/processed/data.pt')
        os.system('rm ' + self.reg_data_folder + self.vout_y_select + '/processed/data.pt')

        # get max reward

        return eff, vout, reward, state.parameters

    def save_dataset_to_file(self, dataset):
        with open(self.reg_data_folder + self.eff_y_select + '/' + self.raw_dataset_file, 'w') as f:
            json.dump(dataset, f)
        f.close()
        with open(self.reg_data_folder + self.vout_y_select + '/' + self.raw_dataset_file, 'w') as f:
            json.dump(dataset, f)
        f.close()

    def get_surrogate_vout(self, state: TopoGenState):
        # TODO
        pass

    def get_surrogate_eff(self, state: TopoGenState):
        """
        return the eff prediction of state, and of self.get_state() if None
        """
        pass

    def get_single_topo_sim_result(self, state: TopoGenState):
        pass
