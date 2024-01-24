import copy
import os
from time import time

import numpy as np
import torch
from torch.utils import data

from Datasets import Dataset
from algo.base import device, MAX_NORM
from algo.base import get_model_params, set_client_from_params, get_acc_loss
from .utils import SummaryWriter


def train_model_fedavg(model, trn_x, trn_y, learning_rate, batch_size, epoch, weight_decay, dataset_name, sch_step=1, sch_gamma=1):
    n_trn = len(trn_x)
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True, num_workers=2)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train()
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)

    duration_list = []
    for e in range(epoch):
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            t = time()

            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss = loss / list(batch_y.size())[0]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=MAX_NORM)  # Clip gradients to prevent exploding
            optimizer.step()

            if i > 0:
                duration_list.append(time() - t)

        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


def train_FedAvg(args, data_obj, model_func, init_model):
    act_prob = args.act_prob
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epoch = args.epoch
    com_amount = args.com_amount
    weight_decay = args.weight_decay
    sch_step = args.sch_step
    sch_gamma = args.sch_gamma
    save_period = args.save_period
    suffix = args.model_name
    result_path = args.result_path
    rand_seed = 0
    lr_decay_per_round = args.lr_decay_per_round
    temperature_list = (np.arange(0.001, com_amount + 1) / com_amount)[::-1] * args.tau

    suffix = 'FedAvg_' + suffix

    n_clnt = data_obj.n_client

    clnt_x = data_obj.clnt_x
    clnt_y = data_obj.clnt_y

    cent_x = []
    for l in clnt_x:
        cent_x.extend(l)
    cent_y = np.concatenate(clnt_y, axis=0)

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))

    if not os.path.exists('%sModel/%s/%s' % (result_path, data_obj.name, suffix)):
        os.mkdir('%sModel/%s/%s' % (result_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_models_sel = list(range(n_save_instances))
    fed_models_all = list(range(n_save_instances))

    trn_perf_sel = np.zeros((com_amount, 2))
    trn_perf_all = np.zeros((com_amount, 2))
    tst_perf_sel = np.zeros((com_amount, 2))
    tst_perf_all = np.zeros((com_amount, 2))
    n_par = len(get_model_params([model_func()])[0])

    init_par_list = get_model_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par

    writer = SummaryWriter('%sRuns/%s/%s' % (result_path, data_obj.name, suffix))

    clnt_models = []
    for clnt in range(n_clnt):
        clnt_model = model_func().to(device)
        clnt_model.load_state_dict(copy.deepcopy(dict(init_model.state_dict())))
        clnt_models.append(clnt_model)

    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.state_dict())))

    all_model = model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.state_dict())))

    for i in range(com_amount):
        # Train if doesn't exist, Fix randomness
        inc_seed = 0
        while (True):
            np.random.seed(i + rand_seed + inc_seed)
            act_list = np.random.uniform(size=n_clnt)
            act_clients = act_list <= act_prob
            selected_clnts = np.sort(np.where(act_clients)[0])
            inc_seed += 1
            if len(selected_clnts) != 0:  # Choose at least one client in each synch
                break

        print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))

        for clnt in selected_clnts:
            print('---- Training client %d' % clnt)
            trn_x = clnt_x[clnt]
            trn_y = clnt_y[clnt]

            clnt_models[clnt] = model_func().to(device)
            clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.state_dict())))

            for params in clnt_models[clnt].parameters():
                params.requires_grad = True

            for name, params in clnt_models[clnt].state_dict().items():
                # update temperature for AGN
                if 'temperature' in name:
                    params.fill_(temperature_list[i])
            clnt_models[clnt] = train_model_fedavg(clnt_models[clnt], trn_x, trn_y,
                                                   learning_rate * (lr_decay_per_round ** i), batch_size, epoch,
                                                   weight_decay,
                                                   data_obj.dataset, sch_step, sch_gamma)

            clnt_params_list[clnt] = get_model_params([clnt_models[clnt]], n_par)[0]

        # Scale with weights
        avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list[selected_clnts] * weight_list[selected_clnts] / np.sum(weight_list[selected_clnts]), axis=0))
        all_model = set_client_from_params(model_func(), np.sum(clnt_params_list * weight_list / np.sum(weight_list), axis=0))

        loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, avg_model, data_obj.dataset, 0)
        tst_perf_sel[i] = [loss_tst, acc_tst]
        print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f" % (i + 1, acc_tst, loss_tst))

        loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset, 0)
        trn_perf_sel[i] = [loss_tst, acc_tst]
        print("**** Communication sel %3d, Cent Accuracy: %.4f, Loss: %.4f" % (i + 1, acc_tst, loss_tst))

        loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, all_model, data_obj.dataset, 0)
        tst_perf_all[i] = [loss_tst, acc_tst]
        print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" % (i + 1, acc_tst, loss_tst))

        loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset, 0)
        trn_perf_all[i] = [loss_tst, acc_tst]
        print("**** Communication all %3d, Cent Accuracy: %.4f, Loss: %.4f" % (i + 1, acc_tst, loss_tst))

        writer.add_scalars('Loss/train_wd',
                           {
                               'Sel clients': get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset, weight_decay)[0],
                               'All clients': get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset, weight_decay)[0]
                           }, i
                           )

        writer.add_scalars('Loss/train',
                           {
                               'Sel clients': trn_perf_sel[i][0],
                               'All clients': trn_perf_all[i][0]
                           }, i
                           )

        writer.add_scalars('Accuracy/train',
                           {
                               'Sel clients': trn_perf_sel[i][1],
                               'All clients': trn_perf_all[i][1]
                           }, i
                           )

        writer.add_scalars('Loss/test',
                           {
                               'Sel clients': tst_perf_sel[i][0],
                               'All clients': tst_perf_all[i][0]
                           }, i
                           )

        writer.add_scalars('Accuracy/test',
                           {
                               'Sel clients': tst_perf_sel[i][1],
                               'All clients': tst_perf_all[i][1]
                           }, i
                           )

        # Freeze model
        for params in avg_model.parameters():
            params.requires_grad = False

        if (i + 1) % save_period == 0:
            torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt' % (result_path, data_obj.name, suffix, (i + 1)))
            torch.save(all_model.state_dict(), '%sModel/%s/%s/%dcom_all.pt' % (result_path, data_obj.name, suffix, (i + 1)))

            np.save('%sModel/%s/%s/%dcom_trn_perf_sel.npy' % (result_path, data_obj.name, suffix, (i + 1)), trn_perf_sel[:i + 1])
            np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (result_path, data_obj.name, suffix, (i + 1)), tst_perf_sel[:i + 1])

            np.save('%sModel/%s/%s/%dcom_trn_perf_all.npy' % (result_path, data_obj.name, suffix, (i + 1)), trn_perf_all[:i + 1])
            np.save('%sModel/%s/%s/%dcom_tst_perf_all.npy' % (result_path, data_obj.name, suffix, (i + 1)), tst_perf_all[:i + 1])

            np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (result_path, data_obj.name, suffix, (i + 1)), clnt_params_list)

            if (i + 1) > save_period:
                if os.path.exists('%sModel/%s/%s/%dcom_trn_perf_sel.npy' % (result_path, data_obj.name, suffix, i + 1 - save_period)):
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%dcom_trn_perf_sel.npy' % (result_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (result_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%dcom_trn_perf_all.npy' % (result_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%dcom_tst_perf_all.npy' % (result_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (result_path, data_obj.name, suffix, i + 1 - save_period))

        if ((i + 1) % save_period == 0):
            fed_models_sel[i // save_period] = avg_model
            fed_models_all[i // save_period] = all_model

    return fed_models_sel, trn_perf_sel, tst_perf_sel, fed_models_all, trn_perf_all, tst_perf_all
