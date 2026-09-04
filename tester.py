import copy
import datetime
import json
import logging
import os
import sys
import time
import numpy as np

import torch
from pathlib import Path
from utils import factory
from utils.data_manager import DataManager
from torch.utils.data import DataLoader
from utils.toolkit import ConfigEncoder, count_parameters, save_fc, save_model, tensor2numpy
from utils.inc_net import BiasLayer#, LoRALayer
from torch import nn
from convs.partpointnet_VQ_VAE import GeneralizedPointNetEncoder, GeneralizedFeatureExtractor, GeneralizedPartConceptLearning, SpecializedTransformer, SpecializedMLP, SpecializedPointNetMLP, SpecializedPartRelationEncoder, l2_normalize
from convs.partpointnet import GeneralizedPointTransformerEncoder
#from convs.partpointnet import GeneralizedPointNetEncoder, GeneralizedFeatureExtractor, GeneralizedPointTransformerEncoder, GeneralizedPartConceptLearning, SpecializedMLP, SpecializedPointNetMLP, SpecializedPartRelationEncoder, l2_normalize


num_workers = 10

def test(args):
    _test(args)

def _test(args):
    logging.info('PARAMETER ...')
    logging.info(args)

    cur_task = 5

    logging.info('Load dataset ...')
    data_path = 'data/co3d'
    data_manager = DataManager(root=data_path, args=args, process_data=args['process_data'],
                                init_cls=args['init_cls'], increment=args['increment'])
    test_dataset = data_manager.get_dataset(np.arange(0, 50), source='test')
    test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False,
                                  num_workers=num_workers)

    model = factory.get_model(args["model_name"], args)
    for i in range(cur_task + 1):
        if i > 0:
            _new_bias = nn.ModuleList([BiasLayer() for j in range(i)])
            model._network.biases.append(_new_bias)
            #_new_lora = LoRALayer()
            #model._network.loras.append(_new_lora)
        _new_extractor = SpecializedMLP(i)
        #_new_extractor = SpecializedTransformer(i)
        #_new_part_learner = GeneralizedPartConceptLearning(k=256, num_points=64, emb_dim=256)
        #model._network.AdaptivePartLearners.append(_new_part_learner)
        model._network.AdaptiveExtractors.append(_new_extractor)

    model._network.cuda()
    checkpoint = torch.load('./log/classification/co3d/best_codebook_{}.pkl'.format(cur_task))
    model._network.load_state_dict(checkpoint['network'], strict=True)
    
    model._network.eval()
    model._total_classes = 25 + 5 * cur_task
    correct, total = 0, 0
    class_correct, class_total = np.zeros((25 + 5 * cur_task)), np.zeros((25 + 5 * cur_task))
    feats_list = []
    targets_list = []
    att_list = []
    recon_criterion = nn.MSELoss()
    loss_recon_list = []
    loss_latent_list = []
    cnt = np.zeros(50)
    test_acc, class_test_acc = model._compute_accuracy(model._network, test_loader)
    before_list = []
    after_list = []
    points_list = []
    #np.save('./conf.npy', class_test_acc)
    print(test_acc, [class_test_acc[i][i] for i in range(25 + 5 * cur_task)])#, [class_correct[i][i] for i in range(50)], [class_total[i][i] for i in range(50)])
    '''
    for i, (_, points, targets) in enumerate(test_loader):
        points = points.cuda()
        points = points.transpose(2, 1)
        with torch.no_grad():
            outputs = model._network(points)
            feats = outputs["logits"]
            #att = outputs["part_logits"]
            after_list.append(feats)
            before_list.append(outputs['before_logits'])
            targets_list.append(targets)
            points_list.append(points)
            #att_list.append(att)
            #part_xyz_list.append(outputs["part_xyz"])
            #recon_list.append(outputs["recon_points"])
            #outputs = outputs["logits"]
            #loss_recon = recon_criterion(outputs['part_xyz'], outputs['recon_points'])
            #loss_latent = outputs['part_diff']
            #loss_recon_list.append(loss_recon)
            #loss_latent_list.append(loss_latent)
        #for j in range(outputs["part_xyz"].shape[0]):
        #    if cnt[targets[j]] < 20:
        #        np.save('./modelnet/xyz_{}_{}.npy'.format(targets[j], int(cnt[targets[j]])), outputs["xyz"][j].cpu().numpy())
        #        np.save('./modelnet/part_xyz_{}_{}.npy'.format(targets[j], int(cnt[targets[j]])), outputs["part_xyz"][j].cpu().numpy())
        #        np.save('./modelnet/att_{}_{}.npy'.format(targets[j], int(cnt[targets[j]])), outputs["part_logits"][j].cpu().numpy())
        #        cnt[targets[j]] += 1 
    #'''
    '''
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == targets).sum()
        total += len(targets)
        for j in range(25 + 5 * cur_task):
            class_correct[j] += ((predicts.cpu() == targets) & (targets == j)).sum()
            class_total[j] += (targets == j).sum()
        #for j in range(50):
        #    for k in range(50):
        #        class_correct[j][k] += ((predicts.cpu() == k) & (targets == j)).sum()
        #        class_total[j][k] += (targets == j).sum()
    '''
    #before = torch.cat(before_list, 0)
    #after = torch.cat(after_list, 0)
    #np.save('./before.npy', before.cpu().numpy())
    #np.save('./after.npy', after.cpu().numpy())
    #points = torch.cat(points_list, 0)
    #targets = torch.cat(targets_list, 0)
    #att = torch.cat(att_list, 0)

    #np.save('./ours_feats.npy', feats.cpu().numpy())
    #np.save('./points.npy', points.cpu().numpy())
    #np.save('./targets.npy', targets.cpu().numpy())
    #print(sum(loss_recon_list) / len(loss_recon_list), sum(loss_latent_list) / len(loss_latent_list))
    #print(model._network.part_prototypes.shape, att.shape)
    #np.save('./recon/prototypes.npy', model._network.part_prototypes.cpu().detach().numpy())
    #np.save('./modelnet/att.npy', att.cpu().numpy())
    #np.save('feats_{}_2.npy'.format(cur_task), feats.cpu().numpy())
    #np.save('./modelnet/targets.npy', targets.cpu().numpy())
    #print(np.around(tensor2numpy(correct) * 100 / total, decimals=2), np.around(class_correct * 100 / class_total, decimals=2))

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

