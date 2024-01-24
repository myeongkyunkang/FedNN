import copy

import numpy as np
import torch
from torch.utils import data

from Datasets import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_NORM = 10


def get_acc_loss(data_x, data_y, model, dataset_name, w_decay=None):
    acc_overall = 0
    loss_overall = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    batch_size = min(100, len(data_x))
    n_tst = len(data_x)
    tst_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)

            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss_overall += loss.item()

            # Accuracy calculation
            y_pred = y_pred.cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == batch_y)
            acc_overall += batch_correct

    loss_overall /= n_tst
    if w_decay != None:
        # Add L2 loss
        params = get_model_params([model], n_par=None)
        loss_overall += w_decay / 2 * np.sum(params * params)

    model.train()
    return loss_overall, acc_overall / n_tst


def set_client_from_params(model, params, strict=True):
    dict_param = copy.deepcopy(dict(model.state_dict()))
    idx = 0
    for name, param in model.state_dict().items():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx + length].reshape(weights.shape)).to(device))
        idx += length
    model.load_state_dict(dict_param, strict=strict)
    return model


def get_model_params(model_list, n_par=None):
    if n_par == None:
        exp_model = model_list[0]
        n_par = 0
        for name, param in exp_model.state_dict().items():
            n_par += len(param.data.reshape(-1))
    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, model in enumerate(model_list):
        idx = 0
        for name, param in model.state_dict().items():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)
