import pickle
import logging
import numpy as np
import os
from torch.utils.data import Dataset
from .utils import graph_processing, multi_input


class K400_HRNet_Feeder(Dataset):
    def __init__(self, phase, graph, root_folder, inputs, debug, window=[0, 300], processing='default', person_id=[0], input_dims=3, **kwargs):
        self.phase = phase
        self.root_folder = root_folder
        self.window = window
        self.inputs = inputs
        self.processing = processing
        self.debug = debug
        
        self.graph = graph.graph
        self.conn = graph.connect_joint
        self.center = graph.center
        self.num_node = graph.num_node
        self.num_person = graph.num_person
        
        self.input_dims = input_dims
        self.M = len(person_id)
        self.max_frame = 300
        self.datashape = self.get_datashape()
        
        data_path = os.path.join(root_folder, 'k400_hrnet.pkl')
        label_path = os.path.join(root_folder, phase+'_label.pkl')
        try:
            logging.info(
                'Loading {} pose data from '.format(phase) + data_path)
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)
            
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
    
    def __getitem__(self, i):
        label, name, index = self.label[i]
        anno = self.data['annotations'][index]
        raw_file = anno['raw_file']
        _, filename = os.path.split(raw_file)
        pkl_data = np.load(
            os.path.join(self.root_folder, 'kpfiles', filename),
            allow_pickle=True
        )
        pose = np.array(pkl_data[name]['keypoint'])
        # (N, ) -> (T,)
        idx_range = [[] for _ in range(self.max_frame)]
        for i, t in enumerate(anno['frame_inds']):
            if t > self.max_frame:
                continue
            idx_range[t-1].append(i)
            
        res_pose = np.zeros([self.max_frame, self.M, 17, 3])
        for t in range(self.max_frame):
            pose_data_t = pose[idx_range[t]]
            scores = np.array(anno['box_score'][idx_range[t]])
            sorted_indices = np.argsort(-scores)
            
            sorted_pose = pose_data_t[sorted_indices]
            m = min(self.M, sorted_pose.shape[0])
            res_pose[t, :m] =sorted_pose[:m]
        
        # (T, M, V, C) -> (C, T, V, M)
        res_pose = res_pose.transpose(3, 0, 2, 1)
        res_pose = graph_processing(res_pose, self.graph, self.processing)
        # (C, T, V, M) -> (I, C*2, T, V, M)
        data_new = multi_input(res_pose, self.conn, self.inputs, self.center)
        
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