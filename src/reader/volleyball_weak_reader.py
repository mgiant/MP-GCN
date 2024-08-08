import os
import pickle
import json
import logging
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from .pose_reassign import read_pose_tracking

class VolleyBall_Weak_Reader():
    def __init__(self, dataset_root_folder, out_folder, **kwargs):
        self.max_channel = 3
        self.max_frame = 41
        self.max_joint = 17
        self.max_person = 12
        
        self.dataset_root_folder = dataset_root_folder
        self.pose_dir = os.path.join(self.dataset_root_folder, 'joints')
        self.ball_dir = os.path.join(self.dataset_root_folder, 'volleyball_ball_annotation')
        self.out_folder = out_folder
        
        # Divide train and eval samples
        self.training_samples = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39,
                                        40, 41, 42, 48, 50, 52, 53, 54, 0, 2, 8, 12, 17, 19, 24, 26,
                                        27, 28, 30, 33, 46, 49, 51]
        self.eval_samples = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
        
        # Create Label-to-idx map
        self.class2idx = {
            'r_set': 0, 'r-set': 0, 'l_set': 1, 'l-set': 1,
            'r_spike': 2, 'r-spike': 2, 'l_spike': 3, 'l-spike': 3,
            'r_pass': 4, 'r-pass': 4, 'l_pass': 5, 'l-pass': 5,
            'r_winpoint': 6, 'r-winpoint': 6, 'l_winpoint': 7, 'l-winpoint': 7
        }
        
    def gendata(self, phase):
        res_skeleton = []
        res_ball = []
        labels = []
        trackings = defaultdict(dict)
        videos = self.training_samples if phase == 'train' else self.eval_samples
        iterizer = tqdm(videos, dynamic_ncols=True)
        for video_id in iterizer:
            vid = str(video_id)
            with open(os.path.join(self.dataset_root_folder, 'videos', vid, 'annotations.txt'), 'r') as f:
                lines = f.readlines()
            for l in lines:
                cid, label = l.split()[:2]
                cid = cid.split('.')[0]
                labels.append([self.class2idx[label], vid + '-' + cid])
                
                # read pose data
                clip_data = []
                clip_dir = os.path.join(self.pose_dir, vid, cid)
                for t in range(self.max_frame):
                    frame_dir = os.path.join(clip_dir, '{:06d}'.format(t))
                    with open(frame_dir, 'r') as f:
                        clip_data.append(json.load(f))
                keypoints, trackings[vid][cid] = read_pose_tracking(
                    clip_data,
                    T=self.max_frame,
                    M=self.max_person,
                    V=self.max_joint,
                    C=self.max_channel)
                res_skeleton.append(keypoints)
                
                # read ball data
                ball_path = os.path.join(self.ball_dir, vid, cid + '.txt')
                with open(ball_path, 'r') as f:
                    ball_data = f.readlines()
                # T, 3
                ball_data = np.array(
                    [list(map(int, t.strip().split())) + [0] for t in ball_data])
                # T, 3 -> T, 1, 3
                ball_data = np.expand_dims(ball_data, axis=1)
                res_ball.append(ball_data)
                
        res_skeleton = np.array(res_skeleton)
        res_ball = np.array(res_ball)
        
        os.makedirs(self.out_folder, exist_ok=True)
        # Save human skeletons
        np.save(
            os.path.join(
                self.out_folder, '{}_data.npy'.format(phase)
            ), 
            res_skeleton
        )
        # Save ball keypoints
        np.save(
            os.path.join(
                self.out_folder, '{}_object_data'.format(phase)
            ),
            res_ball
        )
        
        with open(os.path.join(self.out_folder, '{}_label.pkl').format(phase), 'wb') as f:
            pickle.dump(labels, f)
        with open(os.path.join(self.out_folder, '{}_trackings.pkl'.format(phase)), 'wb') as f:
            pickle.dump(trackings, f)
        
    def start(self):
        for phase in ['train', 'eval']:
            logging.info('Phase: {}'.format(phase))
            self.gendata(phase)