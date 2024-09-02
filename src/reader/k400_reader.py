import os
import pickle
import logging
import numpy as np
from tqdm import tqdm

class K400_HRNet_Reader():
    def __init__(self, dataset_root_folder, out_folder, **kwargs):
        self.max_channel = 3
        self.max_frame = 300
        self.max_joint = 17
        self.max_person = 5
        
        self.dataset_root_folder = dataset_root_folder
        self.out_folder = out_folder
        
        try:
            data_path = os.path.join(dataset_root_folder, 'k400_hrnet.pkl')
            logging.info(f'Loading label data from {data_path}')
            
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)
        except:
            logging.error(f'Error: Wrong in loading data files: {data_path}!')
            raise ValueError()
    
    def gendata(self):
        train_clip_set = set(self.data['split']['train'])
        eval_clip_set = set(self.data['split']['val'])
        train_label = []
        eval_label = []
        for i, anno in tqdm(enumerate(self.data['annotations'])):
            frame_dir = anno['frame_dir']
            # get label
            if frame_dir in train_clip_set:
                train_label.append([anno['label'], frame_dir, i])
            elif frame_dir in eval_clip_set:
                eval_label.append([anno['label'], frame_dir, i])
            
            # get keypoint data
            # raw_file = anno['raw_file']
            # _, filename = os.path.split(raw_file)
            # pkl_data = np.load(
            #     os.path.join(self.dataset_root_folder, 'kpfiles', filename),
            #     allow_pickle=True
            # )
            # pose = np.array(pkl_data[frame_dir]['keypoint']) # (N, 17, 3)
            # # (N, ) -> (T,)
            # idx_range = [[] for _ in range(self.max_frame)]
            # for i, t in enumerate(anno['frame_inds']):
            #     if t > self.max_frame:
            #         continue
            #     idx_range[t-1].append(i)
            # res_pose = np.zeros([self.max_frame, self.max_person, self.max_joint, self.max_channel])
            # for t in range(self.max_frame):
            #     pose_data_t = pose[idx_range[t]]
            #     scores = np.array(anno['box_score'][idx_range[t]])
            #     sorted_indices = np.argsort(-scores)
                
            #     sorted_pose = pose_data_t[sorted_indices]
            #     m = min(self.max_person, sorted_pose.shape[0])
            #     res_pose[t, :m] =sorted_pose[:m]
            # res_data.append(res_pose)
        
        # Save
        os.makedirs(self.out_folder, exist_ok=True)
        # np.save(
        #     os.path.join(self.out_folder, f'{phase}_data.npy'), 
        #     res_data
        # )
        with open(os.path.join(self.out_folder, 'train_label.pkl'), 'wb') as f:
            pickle.dump(train_label, f)
        with open(os.path.join(self.out_folder, 'eval_label.pkl'), 'wb') as f:
            pickle.dump(eval_label, f)
            

    def start(self):
        self.gendata()