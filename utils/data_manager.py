import os
import random
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle
import torch
import models.provider as provider
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000


class DataManager(object):
    def __init__(self, root, args, process_data, init_cls, increment):
        self.root = root
        self.npoints = args['num_point']
        self.process_data = process_data
        self.uniform = args['use_uniform_sample']
        self.use_normals = args['use_normals']
        self.num_category = args['num_category']
        self._setup_data()
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(self, indices, source, appendent=None, ret_data=False, m_rate=None, fewshot=None):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        data, targets = [], []
        for idx in indices:
            if m_rate is None and fewshot is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            elif m_rate is None and fewshot is not None:
                class_data, class_targets = self._select_fewshot(
                    x, y, low_range=idx, high_range=idx + 1, fewshot=fewshot
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, source)
        else:
            return DummyDataset(data, targets, source)

    def _setup_data(self):
        #self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.catfile = os.path.join(self.root, 'co3d_shape_names.txt')
        #self.catfile = os.path.join(self.root, 'shapenet_shape_names.txt')
        #self.catfile = os.path.join(self.root, 'nuscenes_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        '''
        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        shape_names_train = ['_'.join(x.split('_')[0:-1]) for x in shape_ids['train']]
        shape_names_test = ['_'.join(x.split('_')[0:-1]) for x in shape_ids['test']]
        self.datapath_train = [(shape_names_train[i], os.path.join(self.root, shape_names_train[i], shape_ids['train'][i]) + '.txt') for i in range(len(shape_ids['train']))]
        self.datapath_test = [(shape_names_test[i], os.path.join(self.root, shape_names_test[i], shape_ids['test'][i]) + '.txt') for i in range(len(shape_ids['test']))]
        print('The size of %s data is %d' % ('train', len(self.datapath_train)))
        print('The size of %s data is %d' % ('test', len(self.datapath_test)))
        '''

        #if self.uniform:
        #    self.save_path_train = os.path.join(self.root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, 'train', self.npoints))
        #    self.save_path_test = os.path.join(self.root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, 'test', self.npoints))
        #else:
        #    self.save_path_train = os.path.join(self.root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, 'train', self.npoints))
        #    self.save_path_test = os.path.join(self.root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, 'test', self.npoints))        
        
        self.list_of_points_train = []
        self.list_of_labels_train = []
        self.list_of_points_test= []
        self.list_of_labels_test = []

        for category in self.classes.keys():
            train_dir = os.path.join(self.root, category, 'train')
            for filename in tqdm(os.listdir(train_dir), total=len(os.listdir(train_dir))):
                if filename.endswith('.pt'):
                    cls_train = self.classes[category]
                    label_train = np.array([cls_train]).astype(np.int32)
                    point_set_train = torch.load(os.path.join(train_dir, filename)).astype(np.float32)
                    if self.uniform:
                        point_set_train = farthest_point_sample(point_set_train, self.npoints)
                    else:
                        point_set_train = point_set_train[0:self.npoints, :]
                    #point_set_train[:, 0:3] = pc_normalize(point_set_train[:, 0:3])
                    if not self.use_normals:
                        point_set_train = point_set_train[:, 0:3]
                    self.list_of_points_train.append(point_set_train)
                    self.list_of_labels_train.append(label_train)
            test_dir = os.path.join(self.root, category, 'test')
            for filename in tqdm(os.listdir(test_dir), total=len(os.listdir(test_dir))):
                if filename.endswith('.pt'):
                    cls_test = self.classes[category]
                    label_test = np.array([cls_test]).astype(np.int32)
                    point_set_test = torch.load(os.path.join(test_dir, filename)).astype(np.float32)
                    if self.uniform:
                        point_set_test = farthest_point_sample(point_set_test, self.npoints)
                    else:
                        point_set_test = point_set_test[0:self.npoints, :]
                    #point_set_test[:, 0:3] = pc_normalize(point_set_test[:, 0:3])
                    if not self.use_normals:
                        point_set_test = point_set_test[:, 0:3]
                    self.list_of_points_test.append(point_set_test)
                    self.list_of_labels_test.append(label_test)
        print('The size of %s data is %d' % ('train', len(self.list_of_points_train)))
        print('The size of %s data is %d' % ('test', len(self.list_of_points_test)))

        '''
        for index in tqdm(range(len(self.datapath_train)), total=len(self.datapath_train)):
            fn_train = self.datapath_train[index]
            cls_train = self.classes[self.datapath_train[index][0]]
            label_train = np.array([cls_train]).astype(np.int32)
            point_set_train = np.loadtxt(fn_train[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set_train = farthest_point_sample(point_set_train, self.npoints)
            else:
                point_set_train = point_set_train[0:self.npoints, :]
            point_set_train[:, 0:3] = pc_normalize(point_set_train[:, 0:3])
            if not self.use_normals:
                point_set_train = point_set_train[:, 0:3]
            self.list_of_points_train[index] = point_set_train
            self.list_of_labels_train[index] = label_train
        for index in tqdm(range(len(self.datapath_test)), total=len(self.datapath_test)):
            fn_test = self.datapath_test[index]
            cls_test = self.classes[self.datapath_test[index][0]]
            label_test = np.array([cls_test]).astype(np.int32)
            point_set_test = np.loadtxt(fn_test[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set_test = farthest_point_sample(point_set_test, self.npoints)
            else:
                point_set_test = point_set_test[0:self.npoints, :]
            point_set_test[:, 0:3] = pc_normalize(point_set_test[:, 0:3])
            if not self.use_normals:
                point_set_test = point_set_test[:, 0:3]
            self.list_of_points_test[index] = point_set_test
            self.list_of_labels_test[index] = label_test
        '''

        #with open(self.save_path_train, 'rb') as f:
        #    self.list_of_points_train, self.list_of_labels_train = pickle.load(f)
        point_set_train, label_train = np.array(self.list_of_points_train), np.array(self.list_of_labels_train)
        #with open(self.save_path_test, 'rb') as f:
        #    self.list_of_points_test, self.list_of_labels_test = pickle.load(f)
        point_set_test, label_test = np.array(self.list_of_points_test), np.array(self.list_of_labels_test)
        
        # Data
        self._train_data, self._train_targets = point_set_train, label_train
        self._test_data, self._test_targets = point_set_test, label_test

        # Order
        #order = [i for i in range(len(np.unique(self._train_targets)))]
        #order = [8, 30, 0, 4, 2, 37, 22, 33, 35, 5, 21, 36, 26, 25, 7, 12, 14, 23, 16, 17, 
        #        28, 3, 9, 34, 15, 20, 18, 11, 1, 29, 19, 31, 13, 27, 39, 32, 24, 38, 10, 6]
        order = [1, 22, 14, 40, 10, 16, 48, 8, 25, 34, 2, 21, 47, 27, 31, 45, 39, 49, 30, 15, 11, 35, 0, 12, 42, 
                7, 24, 41, 9, 29, 36, 6, 44, 17, 13, 3, 19, 38, 18, 26, 43, 33, 20, 5, 37, 4, 23, 28, 46, 32]
        #order = [49, 18, 0, 17, 47, 44, 30, 53, 6, 32, 13, 22, 12, 4, 26, 24, 19, 41, 28, 10, 31, 9, 50, 29, 52,
        #        36, 1, 25, 40, 38, 5, 48, 37, 11, 54, 42, 27, 35, 46, 51, 14, 3, 15, 39, 33, 21, 45, 2, 8, 23, 34, 43, 20, 7, 16]
        #order = [16, 1, 8, 11, 22, 10, 21, 20, 13, 17, 15, 3, 2, 12, 9, 14, 6, 0, 5, 19, 4, 7, 18]
        self._class_order = order
        logging.info(self._class_order)
        
        for i in range(50):
            _train_index = (self._train_targets[:, 0] == i).nonzero()[0]
            _test_index = (self._test_targets[:, 0] == i).nonzero()[0]
            _train_num = len(_train_index)
            _test_num = len(_test_index)
            _train_index_select = np.random.choice(_train_index, int(_test_num / 2))
            _test_index_select = np.random.choice(_test_index, int(_test_num / 2))
            self._train_data[_train_index_select] = point_set_test[_test_index_select]
            self._test_data[_test_index_select] = point_set_train[_train_index_select]

        # Map indices
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _select_fewshot(self, x, y, low_range, high_range, fewshot):
        assert fewshot is not None
        if fewshot != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            #selected_idxes = np.arange(0, fewshot)
            selected_idxes = np.random.randint(idxes.shape[0], size=fewshot)
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, points, labels, source):
        assert len(points) == len(labels), "Data size error!"
        self.points = points
        self.labels = labels
        self.source = source
        self.train_transforms = transforms.Compose(
                [
                    provider.PointcloudToTensor(),
                    # provider.PointcloudUpSampling(max_num_points=1024 * 2, centroid="random"),
                    # provider.PointcloudRandomCrop(p=0.5, min_num_points=1024),
                    # provider.PointcloudNormalize(),
                    # provider.PointcloudRandomCutout(p=0.5, min_num_points=1024),
                    provider.PointcloudScale(p=1),
                    # d_utils.PointcloudRotate(p=1, axis=[0.0, 0.0, 1.0]),
                    provider.PointcloudRotatePerturbation(p=1),
                    provider.PointcloudTranslate(p=1),
                    provider.PointcloudJitter(p=1),
                    # provider.PointcloudRandomInputDropout(p=1),
                    # d_utils.PointcloudSample(num_pt=self.hparams["num_points"])
                ]
            )

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        point = self.points[idx]
        label = self.labels[idx]
        point[:, 0:3] = pc_normalize(point[:, 0:3])
        if self.source == 'train':
            point1 = self.train_transforms(point)
            point2 = self.train_transforms(point)
            return idx, point1, point2, label
        else:
            return idx, point, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point
