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
    data_path = 'data/co3d'
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
        model.after_task()

    logging.info(f"End Time:{end_time}")

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



