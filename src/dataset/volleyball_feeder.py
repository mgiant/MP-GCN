import pickle
import logging
import numpy as np
import os
from torch.utils.data import Dataset
from .utils import graph_processing, multi_input


class Volleyball_Feeder(Dataset):
    def __init__(self, phase, graph, root_folder, inputs, debug, ball=False, object_folder='', window=[0, 41], processing='default', person_id=[0], input_dims=2, **kwargs):
        self.phase = phase
        self.window = window
        self.inputs = inputs
        self.processing = processing
        self.ball = ball
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
            logging.info('Loading {} pose data from {}'.format(phase, data_path))
            self.data = np.load(data_path)
            # N, T, M, V, C -> N, C, T, V, M
            self.data = self.data.transpose(0, 4, 1, 3, 2)
            
            with open(label_path, 'rb') as f:
                self.label = pickle.load(f)
            
            if ball:
                logging.info('Loading {} object data from '.format(phase) + object_path)
                self.object_data = np.load(object_path)
                # (N, T, v, C) -> (N, C, T, v)
                self.object_data = self.object_data.transpose(0, 3, 1, 2)
                
                # (N, C, T, v) -> (N, C, T, v, M)
                self.object_data = np.expand_dims(self.object_data, axis=-1)
                self.object_data = np.tile(self.object_data, (1, 1, 1, 1, self.M))
                
                # (N, C, T, V, M) -> (N, C, T, V+v, M)
                self.data = np.concatenate((self.data, self.object_data), axis = 3)
            
            self.data = self.data[:, :self.input_dims, range(*self.window), :, :]

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
        # (C, T, V, M)
        pose_data = np.array(self.data[idx])
        label, name = self.label[idx]
        
        pose_data = graph_processing(pose_data, self.graph, self.processing)
        data_new = multi_input(pose_data, self.conn, self.inputs, self.center)
        
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
