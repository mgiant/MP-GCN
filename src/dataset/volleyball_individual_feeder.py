import pickle
import logging
import numpy as np
import os
from torch.utils.data import Dataset


class Volleyball_Individual_Feeder(Dataset):
    def __init__(self, phase, graph, root_folder, inputs, debug, ball=False, object_folder='', window=[0, 41], processing='default', crop=False, transform=False, person_id=[0], input_dims=2, **kwargs):
        self.phase = phase
        self.window = window
        self.inputs = inputs
        self.processing = processing
        self.debug = debug
        
        self.graph = graph.graph
        self.conn = graph.connect_joint
        self.center = graph.center
        self.num_node = graph.num_node

        self.input_dims = input_dims
        self.M = len(person_id)

        data_path = os.path.join(root_folder, phase+'_data.npy')
        label_path = os.path.join(root_folder, phase+'_individual_label.pkl')
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
        self.M = 1
        # [N, C, T, V, M] -> [M*N, C, T, V, 1]
        self.data = self.data.transpose(0, 4, 1, 2, 3)
        self.data = self.data.reshape(-1, self.data.shape[2],   # [N*M, C, T, V]
                                        self.data.shape[3], self.data.shape[4], 1)
        self.datashape = self.set_datashape()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # (C, T, V, M)
        data = np.array(self.data[idx])
        label, name = self.label[idx]

        C = data.shape[0]

        joint, motion, bone, bone_motion = self.multi_input(
            data[:, :, :, :])
        data_new = []
        if self.inputs.isupper():
            if 'J' in self.inputs:
                data_new.append(joint)
            if 'V' in self.inputs:
                data_new.append(motion)
            if 'B' in self.inputs:
                data_new.append(bone)
            if 'M' in self.inputs:
                data_new.append(bone_motion)
        elif self.inputs == 'joint':
            data_new = [joint[:C, :, :, :]]
        elif self.inputs == 'bone':
            data_new = [bone[:C, :, :, :]]
        elif self.inputs == 'motion':
            data_new = [motion[:C, :, :, :]]
        elif self.inputs == 'bone_motion':
            data_new = [bone_motion[:C, :, :, :]]
        else:
            logging.info('')
            logging.error('Error: No input feature!')
            raise ValueError()
        data_new = np.stack(data_new, axis=0)
        assert list(data_new.shape) == self.datashape
        return data_new, label, name

    def multi_input(self, data):
        C, T, V, M = data.shape
        joint = np.zeros((C*2, T, V, M))
        velocity = np.zeros((C*2, T, V, M))
        bone = np.zeros((C*2, T, V, M))
        bone_motion = np.zeros((C*2, T, V, M))
        joint[:C, :, :, :] = data
        for i in range(V):
            joint[C:, :, i, :] = data[:, :, i, :] - data[:, :, 1, :]
        for i in range(T-2):
            velocity[:C, i, :, :] = data[:, i+1, :, :] - data[:, i, :, :]
            velocity[C:, i, :, :] = data[:, i+2, :, :] - data[:, i, :, :]
        for i in range(len(self.conn)):
            bone[:C, :, i, :] = data[:, :, i, :] - data[:, :, self.conn[i], :]
        for i in range(T-2):
            bone_motion[:C, i, :, :] = bone[:C, i+1, :, :] - bone[:C, i, :, :]
            bone_motion[C:, i, :, :] = bone[:C, i+2, :, :] - bone[:C, i, :, :]
        bone_length = 0
        for i in range(C):
            bone_length += bone[i, :, :, :] ** 2
        bone_length = np.sqrt(bone_length) + 0.0001
        for i in range(C):
            bone[C+i, :, :, :] = np.arccos(bone[i, :, :, :] / bone_length)
        return joint, velocity, bone, bone_motion

    def set_datashape(self):
        I = len(self.inputs) if self.inputs.isupper() else 1
        C = self.input_dims if self.inputs in ['joint', 'joint-motion', 'bone', 'bone-motion'] else self.input_dims*2
        T = len(range(*self.window))
        V = self.num_node
        return [I, C, T, V, 1]
