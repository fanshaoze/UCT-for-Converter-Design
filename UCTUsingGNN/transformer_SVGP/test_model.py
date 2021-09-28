''' Translate input text with trained model. '''
import logging

import torch
import torch.utils.data
import argparse
from tqdm import tqdm
from model.dataset import get_loader
from build_vocab import Vocabulary
from model.Models import get_model, create_masks,GPModel
# from Beam import beam_search
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import time
import gpytorch



def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='test_model.py')

    parser.add_argument('-pretrained_model', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-data_test', required=True,
                        help='Path to input file')
    parser.add_argument('-vocab', required=True,
                        help='Path to vocab file')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=1,
                        help='Batch size must be 1')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-crop_size', type=int, default=224, help="""crop size""")
    parser.add_argument('-max_seq_len', type=int, default=64, help="""seq length""")
    parser.add_argument('-attribute_len', type=int, default=5, help="""attribute length""")

    parser.add_argument('-target', type=str, choices=['eff', 'vout'], default='eff')


    opt = parser.parse_args()
    # if args.batch_size != 1:
    #     print("batch size must be 1")
    #     exit()

    opt.cuda = not opt.no_cuda
    
    opt.device = torch.device('cuda' if opt.cuda else 'cpu')

    # print(args)
    test(opt)

def test(opt, model=None, gp=None):
    """
    Evaluate model and gp if set, otherwise load them from opt.pretrained_model
    """
    vocab = Vocabulary()

    vocab.load(opt.vocab)
  
    data_loader = get_loader(opt.data_test,
                                 vocab, None,
                                 opt.batch_size, shuffle=False,max_seq_len=opt.max_seq_len,
                                attribute_len=opt.attribute_len
                                     )

    if model is None or gp is None:
        # if model or gp is not set, read from file named pretrained_model
        checkpoint = torch.load(opt.pretrained_model + '.chkpt')
        gp_para = checkpoint["gp_model"]

        model = get_model(opt, load_weights=True)
        model = model.to(opt.device)

        gp = GPModel(gp_para["variational_strategy.inducing_points"])
        gp.load_state_dict(gp_para)
        gp = gp.to(opt.device)

    count = 0

    model.eval()
    gp.eval()

    model_errors = []
    all_y = []

    start = time.time()
    with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
        for batch in tqdm(data_loader, mininterval=2, desc='  - (Test)', leave=False):
            path, eff, vout, padding_mask  = map(lambda x: x.to(opt.device), batch)

            pred, final = model(path,padding_mask)
            pred = gp(pred)

            if opt.target == 'eff':
                y = eff.squeeze(1)
            elif opt.target == 'vout':
                y = vout.squeeze(1) / 50
            else:
                raise Exception('unknown target ' + opt.target)

            model_errors.append(torch.square(pred.mean - y))
            all_y.append(y)

            count += path.size(0)
    end = time.time()

    model_errors = np.array(torch.cat(model_errors).cpu())
    all_y = np.array(torch.cat(all_y).cpu())

    model_mse = np.mean(model_errors)

    y_mean = np.mean(all_y)
    baseline_mse = np.mean([(y - y_mean) ** 2 for y in all_y])

    rmse = model_mse / baseline_mse

    print("rmse is: ", rmse)
    print("avg inference time is: ", (end - start)/count)

    logging.info('rmse is ' + str(rmse))



if __name__ == "__main__":
    main()



