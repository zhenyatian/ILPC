import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
import copy
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from models.provider import random_point_dropout, random_scale_point_cloud, shift_point_cloud
from utils.inc_net import PointNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from convs.pointnet import feature_transform_reguliarzer

num_workers = 10

class ILPC(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._old_base = None
        self._network = PointNet(args['init_cls'], args['use_normals'])
        self._teacher_network = None
        logging.info(f'>>> train generalized blocks:{self.args["train_base"]} train_adaptive:{self.args["train_adaptive"]}')

    def after_task(self):
        self._known_classes = self._total_classes
        if self._cur_task == 0:
            if self.args['train_base']:
                logging.info("Train Generalized Blocks...")
                self._network.TaskAgnosticExtractor.train()
                self._network.TaskAgnosticPartLearner.train()
                for param in self._network.TaskAgnosticExtractor.parameters():
                    param.requires_grad = True
                for param in self._network.TaskAgnosticPartLearner.parameters():
                    param.requires_grad = True
            else:
                logging.info("Fix Generalized Blocks...")
                self._network.TaskAgnosticExtractor.eval()
                self._network.TaskAgnosticPartLearner.eval()
                for param in self._network.TaskAgnosticExtractor.parameters():
                    param.requires_grad = False
                for param in self._network.TaskAgnosticPartLearner.parameters():
                    param.requires_grad = False
        
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager, criterion):
        self._teacher_network = copy.deepcopy(self._network)
        for p in self._teacher_network.parameters():
            p.requires_grad = False
        self._teacher_network.eval()
        
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        if self._cur_task > 0:
            for i in range(self._cur_task):
                if self.args['train_adaptive']:
                    for p in self._network.AdaptiveExtractors[i].parameters():
                        p.requires_grad = True
                    self._network.AdaptiveExtractors[i].prototypes.requires_grad = True
                    self._network.part_prototype_list[i].requires_grad = True
                else:
                    for p in self._network.AdaptiveExtractors[i].parameters():
                        p.requires_grad = False
                    self._network.AdaptiveExtractors[i].prototypes.requires_grad = False
                    self._network.part_prototype_list[i].requires_grad = False

        if self._cur_task > 1:
            for i in range(self._cur_task - 1):
                if self.args['train_adaptive']:
                    for j in range(i + 1):
                        self._network.biases[i][j].alpha.requires_grad = True
                else:
                    for j in range(i + 1):
                        self._network.biases[i][j].alpha.requires_grad = False

        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        if self._cur_task > 0:
            train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', appendent=self._get_memory())
        else:
            train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', appendent=self._get_memory())
        self.train_loader = DataLoader(train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers, drop_last=True)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='train')
        self.test_loader = DataLoader(test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers)
        self.criterion = criterion

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.criterion)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
    
    def set_network(self):
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._network.train()                   #All status from eval to train
        if self.args['train_base']:
            self._network.TaskAgnosticExtractor.train()
            self._network.TaskAgnosticPartLearner.train()
        else:
            self._network.TaskAgnosticExtractor.eval()
            self._network.TaskAgnosticPartLearner.eval()
        
        # set adaptive extractor's status
        self._network.AdaptiveExtractors[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                if self.args['train_adaptive']:
                    self._network.AdaptiveExtractors[i].train()
                else:
                    self._network.AdaptiveExtractors[i].eval()
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
            
    def _train(self, train_loader, test_loader, criterion):
        self._network.to(self._device)
        if self._cur_task == 0:
            if self.args['optimizer'] == 'Adam':
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self._network.parameters()),
                    lr=self.args['init_lr'],
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    weight_decay=self.args['init_weight_decay']
                )
            else:
                optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, self._network.parameters()),
                    lr=self.args['init_lr'],
                    weight_decay=self.args['init_weight_decay'],
                    momentum=0.9)
            if self.args['scheduler'] == 'steplr':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
            elif self.args['scheduler'] == 'multisteplr':
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer, 
                    milestones=self.args['init_milestones'], 
                    gamma=self.args['init_lr_decay']
                )
            elif self.args['scheduler'] == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=self.args['init_epoch']
                ) 
            else:
                raise NotImplementedError
            
            if not self.args['skip']:
                self._init_train(train_loader, test_loader, optimizer, scheduler, criterion)
            else:
                if isinstance(self._network, nn.DataParallel):
                    self._network = self._network.module
                load_acc = self._network.load_checkpoint(self.args)
                self._network.to(self._device)

                if len(self._multiple_gpus) > 1:
                    self._network = nn.DataParallel(self._network, self._multiple_gpus)
                
                cur_test_acc = self._compute_accuracy(self._network, self.test_loader)
                logging.info(f"Loaded_Test_Acc:{load_acc} Cur_Test_Acc:{cur_test_acc}")
        else:
            if self.args['optimizer'] == 'Adam':
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self._network.parameters()),
                    lr=self.args['lrate'],
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    weight_decay=self.args['weight_decay']
                )
            else:
                optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, self._network.parameters()),
                    lr=self.args['lrate'],
                    weight_decay=self.args['weight_decay'],
                    momentum=0.9)
            if self.args['scheduler'] == 'steplr':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
            elif self.args['scheduler'] == 'multisteplr':
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer,
                    milestones=self.args['milestones'], 
                    gamma=self.args['lrate_decay']
                )
            elif self.args['scheduler'] == 'cosine':
                assert self.args['t_max'] is not None
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=self.args['t_max']
                )
            else:
                raise NotImplementedError
            self._update_representation(train_loader, test_loader, optimizer, scheduler, criterion)
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(self._total_classes - self._known_classes)
            else:
                self._network.weight_align(self._total_classes - self._known_classes)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, criterion):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        best_test_acc = 0.
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, points, points_bar, targets) in enumerate(train_loader):
                points = points.transpose(2, 1)
                points_bar = points_bar.transpose(2, 1)

                if not self.args['use_cpu']:
                    points, points_bar, targets = points.cuda(), points_bar.cuda(), targets.cuda()
                outputs = self._network(points)
                outputs_bar = self._network(points_bar)

                logits, loss_clf, loss_batch_list = criterion(outputs, targets, outputs_bar, 0)
                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc, class_test_acc, feat, part_related, attn = self._compute_accuracy(self._network, test_loader)
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args['init_epoch'], losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args['init_epoch'], losses/len(train_loader), train_acc)
            logging.info(info)
        info = 'Task {} => Best_test_accy {:.2f}'.format(self._cur_task, best_test_acc)
        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler, criterion):
        prog_bar = tqdm(range(self.args["epochs"]))
        best_test_acc = 0.
        for _, epoch in enumerate(prog_bar):
            self.set_network()
            losses = 0.
            losses_clf = 0.
            losses_aux = 0.
            correct, total = 0, 0
            for i, (_, points, points_bar, targets) in enumerate(train_loader):
                points = points.transpose(2, 1)
                points_bar = points_bar.transpose(2, 1)

                if not self.args['use_cpu']:
                    points, points_bar, targets = points.cuda(), points_bar.cuda(), targets.cuda()
                outputs = self._network(points)
                outputs_bar = self._network(points_bar)
                logits, loss_clf, loss_batch_list = criterion(outputs, targets, outputs_bar, 1)
                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_clf += loss_clf.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc, class_test_acc, feat, part_related, attn = self._compute_accuracy(self._network, test_loader)
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args["epochs"], losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args["epochs"], losses/len(train_loader), train_acc)
            logging.info(info)
        info = 'Task {} => Best_test_accy {:.2f}'.format(self._cur_task, best_test_acc)
        logging.info(info)
