import copy
import datetime
import json
import logging
import os
import sys
import time

import torch
from pathlib import Path
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import ConfigEncoder, count_parameters, save_fc, save_model

def train(args):
    _train(args)

def _train(args):
    time_str = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    args['time_str'] = time_str
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args['log_dir'] is None:
        exp_dir = exp_dir.joinpath(time_str)
    else:
        exp_dir = exp_dir.joinpath(args['log_dir'])
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
          logging.FileHandler('%s/%s.txt' % (log_dir, args['model'])),
          logging.StreamHandler(sys.stdout)
        ],
    )
    logging.info('PARAMETER ...')
    logging.info(args)

    logging.info('Load dataset ...')
    #data_path = 'data/modelnet'
    data_path = 'data/co3d'
    #data_path = 'data/shapenet'
    #data_path = 'data/nuscenes'
    data_manager = DataManager(root=data_path, args=args, process_data=args['process_data'],
                                init_cls=args['init_cls'], increment=args['increment'])
    model = factory.get_model(args["model_name"], args)
    criterion = factory.get_loss(args["loss_name"], args)

    best_instance_acc_curve, best_class_acc_curve = [], []
    start_time = time.time()
    logging.info(f"Start time:{start_time}")
    
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info("Trainable params: {}".format(count_parameters(model._network, True)))
        
        model.incremental_train(data_manager, criterion)
        #if task == data_manager.nb_tasks - 1:
        #    cnn_accy, nme_accy = model.eval_task(save_conf=False)
        #    no_nme = True if nme_accy is None else False
        #else:
        #    cnn_accy, nme_accy = model.eval_task(save_conf=False)
        model.after_task()

        '''
        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))
        '''
    
    end_time = time.time()
    logging.info(f"End Time:{end_time}")
    #cost_time = end_time - start_time
    #save_time(args, cost_time)
    #save_results(args, cnn_curve, nme_curve, no_nme)
    #if args['model_name'] not in ["podnet", "coil"]:
    #    save_fc(args, model)
    #else:
    #    save_model(args, model)

def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))

def save_time(args, cost_time):
    _log_dir = os.path.join("./results/", "times", f"{args['prefix']}")
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    with open(_log_path, "a+") as f:
        f.write(f"{args['time_str']},{args['model_name']}, {cost_time} \n")

def save_results(args, cnn_curve, nme_curve, no_nme=False):
    cnn_top1, cnn_top5 = cnn_curve["top1"], cnn_curve['top5']
    nme_top1, nme_top5 = nme_curve["top1"], nme_curve['top5']
    
    #-------CNN TOP1----------
    _log_dir = os.path.join("./results/", f"{args['prefix']}", "cnn_top1")
    os.makedirs(_log_dir, exist_ok=True)

    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    if args['prefix'] == 'benchmark':
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},")
            for _acc in cnn_top1[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top1[-1]} \n")
    else:
        assert args['prefix'] in ['fair', 'auc']
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
            for _acc in cnn_top1[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top1[-1]} \n")

    #-------CNN TOP5----------
    _log_dir = os.path.join("./results/", f"{args['prefix']}", "cnn_top5")
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    if args['prefix'] == 'benchmark':
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},")
            for _acc in cnn_top5[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top5[-1]} \n")
    else:
        assert args['prefix'] in ['auc', 'fair']
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
            for _acc in cnn_top5[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top5[-1]} \n")


    #-------NME TOP1----------
    if no_nme is False:
        _log_dir = os.path.join("./results/", f"{args['prefix']}", "nme_top1")
        os.makedirs(_log_dir, exist_ok=True)
        _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
        if args['prefix'] == 'benchmark':
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},")
                for _acc in nme_top1[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top1[-1]} \n")
        else:
            assert args['prefix'] in ['fair', 'auc']
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
                for _acc in nme_top1[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top1[-1]} \n")       

        #-------NME TOP5----------
        _log_dir = os.path.join("./results/", f"{args['prefix']}", "nme_top5")
        os.makedirs(_log_dir, exist_ok=True)
        _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
        if args['prefix'] == 'benchmark':
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},")
                for _acc in nme_top5[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top5[-1]} \n")
        else:
            assert args['prefix'] in ['auc', 'fair']
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
                for _acc in nme_top5[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top5[-1]} \n")

