import os

import numpy as np
import torch
from torch.nn import Linear, MSELoss

from topo_data import Autopo, split_balance_data

from ml_utils import train, test, rse, initialize_model
import argparse

if __name__ == '__main__':

    # ======================== Arguments ==========================#

    parser = argparse.ArgumentParser()

    parser.add_argument('-path', type=str, default="../0_rawdata", help='raw data path')
    parser.add_argument('-y_select', type=str, default='reg_eff', help='define target label')
    parser.add_argument('-batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-n_epoch', type=int, default=1, help='number of training epoch')
    parser.add_argument('-gnn_nodes', type=int, default=100, help='number of nodes in hidden layer in GNN')
    parser.add_argument('-predictor_nodes', type=int, default=100,
                        help='number of MLP predictor nodes at output of GNN')
    parser.add_argument('-gnn_layers', type=int, default=3, help='number of layer')
    parser.add_argument('-model_index', type=int, default=1, help='model index')
    parser.add_argument('-threshold', type=float, default=0, help='classification threshold')

    parser.add_argument('-retrain', action='store_true', default=False, help='force retrain model')
    parser.add_argument('-seed', type=int, default=0, help='random seed')

    args = parser.parse_args()

    path = args.path
    y_select = args.y_select
    data_folder = '../2_dataset/' + y_select
    batch_size = args.batch_size
    n_epoch = args.n_epoch
    th = args.threshold

    # ======================== Data & Model ==========================#

    dataset = Autopo(data_folder, path, y_select)

    train_loader, val_loader, test_loader = split_balance_data(dataset, y_select[:3] == 'cls', batch_size)

    # set random seed for training
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    nf_size = 4
    ef_size = 3
    nnode = 4
    if args.model_index == 0:
        ef_size = 6

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)

    model = initialize_model(args.model_index, args.gnn_nodes, args.predictor_nodes, args.gnn_layers, nf_size, ef_size,
                             device)

    pt_filename = y_select + '.pt'
    if os.path.exists(pt_filename) and not args.retrain:
        print('loading model from pt file')

        model_state_dict, _ = torch.load(pt_filename)
        model.load_state_dict(model_state_dict)
    else:
        print('training')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = MSELoss(reduction='mean').to(device)
        model = train(train_loader, val_loader, model, n_epoch, batch_size, nnode, device, args.model_index, optimizer)

        # save model and test data
        torch.save((model.state_dict(), test_loader), y_select + '.pt')

    # test(test_loader, model, n_epoch, batch_size, nnode, args.model_index, y_select[:3] == 'cls', device, th)
    test(test_loader, model, n_epoch, batch_size, nnode, args.model_index, y_select[:3] == 'cls', device, th)
