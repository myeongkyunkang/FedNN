import argparse
import os
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from Datasets import DatasetObject
from Models import client_model
from algo.fedavg import train_FedAvg


def main(args):
    # Create dataset
    data_obj = DatasetObject(dataset=args.dataset_name, n_client=args.n_client, seed=args.data_seed, result_path=args.result_path, data_dir=args.data_dir)

    # Get model
    model_func = lambda: client_model(args.model_name, num_clients=args.n_client, pretrained=args.pretrained)

    # Initialize the model for all methods with a random seed or load it from a saved initial model
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training.')
    else:
        cudnn.benchmark = True

    init_model = model_func()
    if not os.path.exists('%sModel/%s/%s_init_model.pt' % (args.result_path, data_obj.name, args.model_name)):
        os.makedirs('%sModel/%s/' % (args.result_path, data_obj.name), exist_ok=True)
        torch.save(init_model.state_dict(), '%sModel/%s/%s_init_model.pt' % (args.result_path, data_obj.name, args.model_name))
    else:
        # Load model
        init_model.load_state_dict(torch.load('%sModel/%s/%s_init_model.pt' % (args.result_path, data_obj.name, args.model_name)))

    print(args.method)
    if args.method == 'fedavg':
        train_FedAvg(args, data_obj, model_func, init_model)
    else:
        raise ValueError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--result_path', default='./results/')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--method', default='fedavg', help='fedavg')
    parser.add_argument('--model_name', default='LeNet', help='LeNet, ResNet18')
    parser.add_argument('--dataset_name', default='DIGIT', help='DIGIT')
    parser.add_argument('--data_seed', default=23, type=int)
    parser.add_argument('--seed', default=23, type=int)
    parser.add_argument('--com_amount', default=200, type=int)
    parser.add_argument('--save_period', default=50, type=int)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--learning_rate', default=0.1, type=float)
    parser.add_argument('--lr_decay_per_round', default=0.998, type=float)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--epoch', default=1, type=int)
    parser.add_argument('--n_client', default=100, type=int)
    parser.add_argument('--sch_step', default=1, type=int)
    parser.add_argument('--sch_gamma', default=1, type=int)
    parser.add_argument('--act_prob', default=0.4, type=float)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--tau', default=5.0, type=float)
    args = parser.parse_args()

    if args.result_path[-1] != '/':
        args.result_path += '/'

    if args.dataset_name == 'DIGIT':
        args.n_client = 5
        print('Reset client:', args.n_client)

    main(args)
