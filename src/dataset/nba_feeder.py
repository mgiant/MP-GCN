import pickle
import logging
import numpy as np
import os
from torch.utils.data import Dataset
from .utils import graph_processing, multi_input


class NBA_Feeder(Dataset):
    def __init__(self, phase, graph, root_folder, inputs, debug, object_folder='', window=[0, 72], processing='default', person_id=[0], input_dims=3, ball=False, net=False, **kwargs):
        self.phase = phase
        self.window = window
        self.inputs = inputs
        self.processing = processing
        self.ball = ball
        self.net = net
        self.debug = debug

        self.graph = graph.graph
        self.conn = graph.connect_joint
        self.center = graph.center
        self.num_node = graph.num_node
        self.num_person = graph.num_person

        self.input_dims = input_dims
        self.M = len(person_id)
        self.datashape = self.get_datashape()

        data_path = os.path.join(root_folder, phase+'_data.npy')
        label_path = os.path.join(root_folder, phase+'_label.pkl')
        object_path = os.path.join(object_folder, phase+'_object_data.npy')
        try:
            logging.info(
                'Loading {} pose data from '.format(phase) + data_path)
            self.data = np.load(data_path, mmap_mode='r')

            if self.ball or self.net:
                logging.info('Loading {} object data from '.format(
                    phase) + object_path)
                self.object_data = np.load(object_path, mmap_mode='r')

            with open(label_path, 'rb') as f:
                self.label = pickle.load(f)

        except:
            logging.info('')
            logging.error('Error: Wrong in loading data files: {} or {}!'.format(
                data_path, label_path))
            logging.info('Please generate data first!')
            raise ValueError()
        if self.debug:
            self.data = self.data[:300]
            self.label = self.label[:300]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # (T, M, V, C) -> (C, T, V, M)
        pose_data = self.data[idx]
        pose_data = pose_data.transpose(3, 0, 2, 1)
        if self.ball or self.net:
            # (T, v, C) -> (C, T, v)
            object_data = self.object_data[idx]
            object_data = object_data.transpose(2, 0, 1)

        label, name = self.label[idx]

        if self.ball or self.net:
            # (C, T, V, M) + (C, T, v) -> (C, T, V+v, M)
            M = pose_data.shape[3]
            if not self.ball:
                object_data[:, :, 0] = 0
            if not self.net:
                object_data[:, :, 1] = 0
            object_data = np.expand_dims(object_data, axis=-1)
            object_data = np.tile(object_data, (1, 1, 1, M))
            pose_data = np.concatenate((pose_data, object_data), axis=2)
        data = graph_processing(pose_data, self.graph, self.processing)
        # (C, T, V, M) -> (I, C*2, T, V, M)
        data_new = multi_input(data, self.conn, self.inputs, self.center)

        try:
            assert list(data_new.shape) == self.datashape
        except AssertionError:
            logging.info('data_new.shape: {}'.format(data_new.shape))
            raise ValueError()
        return data_new, label, name

    def get_datashape(self):
        I = len(self.inputs) if self.inputs.isupper() else 1
        C = self.input_dims if self.inputs in [
            'joint', 'joint-motion', 'bone', 'bone-motion'] else self.input_dims*2
        T = len(range(*self.window))
        V = self.num_node
        M = self.M // self.num_person
        return [I, C, T, V, M]
