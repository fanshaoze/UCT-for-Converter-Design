import json

import gpytorch
import torch
import numpy as np

from transformer_SVGP.model.Models import get_model, GPModel
from topo_envs.surrogateRewardSim import SurrogateRewardTopologySim

from transformer_SVGP.build_vocab import Vocabulary

class TransformerRewardSim(SurrogateRewardTopologySim):
    def __init__(self, eff_model_file, vout_model_file, vocab_file, device, debug, *args):
        super().__init__(debug, *args)

        vocab = Vocabulary()
        vocab.load(vocab_file)
        self.vocab = vocab

        self.device = device
        self.debug = debug

        self.eff_model, self.eff_gp = self.load_model(eff_model_file)
        self.vout_model, self.vout_gp = self.load_model(vout_model_file)

        #self.test()

    def load_model(self, file_name):
        # load transformer model, using Yupeng's code
        model = get_model(cuda=(self.device == 'gpu'), pretrained_model=file_name, load_weights=True)
        model = model.to(self.device)

        checkpoint = torch.load(file_name + '.chkpt')

        gp_para = checkpoint["gp_model"]

        gp = GPModel(gp_para["variational_strategy.inducing_points"])
        gp.load_state_dict(gp_para)
        gp = gp.to(self.device)

        model.eval()
        gp.eval()

        return model, gp

    def get_transformer_predict(self, paths, model, gp):
        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
            path_list = [self.vocab(token) for token in paths]
            path_tensor = torch.Tensor(path_list).unsqueeze(0).long()

            padding_mask = (path_tensor != 0)

            path_tensor = path_tensor.to(self.device)
            padding_mask = padding_mask.to(self.device)

            pred, final = model(path_tensor, padding_mask)
            pred = gp(pred)

        return pred.mean.item()

    def get_surrogate_eff(self, state):
        self.set_state(None, None, state)
        paths = self.find_paths()

        return np.clip(self.get_transformer_predict(paths, self.eff_model, self.eff_gp), 0., 1.)

    def get_surrogate_vout(self, state):
        self.set_state(None, None, state)
        paths = self.find_paths()

        return np.clip(50 * self.get_transformer_predict(paths, self.vout_model, self.vout_gp), 0., 50.)
