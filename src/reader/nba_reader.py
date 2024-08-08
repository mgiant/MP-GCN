import os
import pickle
import logging
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from .pose_reassign import read_pose_tracking

class NBA_Reader():
    def __init__(self, dataset_root_folder, out_folder, pose = True, ball = True, **kwargs):
        self.max_channel = 3
        self.max_frame = 72
        self.max_joint = 17
        self.max_person = 12
        
        self.dataset_root_folder = dataset_root_folder
        self.pose_dir = os.path.join(dataset_root_folder, "joints")
        self.ball_dir = os.path.join(dataset_root_folder, "objects")
        self.out_folder = out_folder
        
        self.pose = pose
        self.ball = ball
        
        # Create label-to-idx map
        self.class2idx = {
            "2p-succ.":0, 
            "2p-fail.-off.": 1, 
            "2p-fail.-def.":2, 
            "2p-layup-succ.":3, 
            "2p-layup-fail.-off.":4, 
            "2p-layup-fail.-def.":5, 
            "3p-succ.":6, 
            "3p-fail.-off.":7,
            "3p-fail.-def.":8 
        }

    def read_object(self, ball_data):
        ball_class_name = 'Basketball'
        net_class_name = 'Net'
        res_data = np.zeros((self.max_frame, 2, self.max_channel))
        for t, frame in enumerate(ball_data):
            for object in frame:
                ball_conf = 0
                net_conf = 0
                if object['name'] == ball_class_name and object['confidence'] > ball_conf:
                    object_xyv = np.array([(object['box']['x1'] + object['box']['x2']) / 2,
                                           (object['box']['y1'] + object['box']['y2']) / 2,
                                           object['confidence']])
                    res_data[t, 0, :] = object_xyv
                    ball_conf = object['confidence']
                if object['name'] == net_class_name and object['confidence'] > net_conf:
                    object_xyv = np.array([(object['box']['x1'] + object['box']['x2']) / 2,
                                           (object['box']['y1'] + object['box']['y2']) / 2,
                                           object['confidence']])
                    res_data[t, 1, :] = object_xyv
                    net_conf = object['confidence']
        return res_data
    
    def gendata(self, phase):
        
        label_name = '{}_video_ids'.format('test' if phase=='eval' else 'train')
        with open(os.path.join(self.dataset_root_folder, label_name), 'r') as f:
            video_ids = f.read().split(',')[:-1]
        
        res_skeleton = []
        res_object = []
        labels = []
        trackings = defaultdict(dict)
        iterizer = tqdm(video_ids, dynamic_ncols=True)
        for video_id in iterizer:
            vid = str(video_id)
            label_path = os.path.join(
                self.dataset_root_folder, 'videos', vid, 'annotations.txt')
            with open(label_path, 'r') as f:
                lines = f.readlines()
            for l in lines:
                cid, label = l.split()
                labels.append([self.class2idx[label], vid + '-' + cid])
                
                # Read pose data
                if self.pose:
                    clip_data = []
                    clip_dir = os.path.join(self.pose_dir, vid, cid)
                    for t in range(self.max_frame):
                        frame_dir = os.path.join(clip_dir, '{:06d}'.format(t))
                        with open(frame_dir, 'r') as f:
                            clip_data.append(json.load(f))
                    keypoints, trackings[vid][cid] = read_pose_tracking(clip_data,
                        T=self.max_frame,
                        M=self.max_person,
                        V=self.max_joint,
                        C=self.max_channel)
                    res_skeleton.append(keypoints)
                
                # Read ball data
                if self.ball:
                    ball_data = []
                    ball_dir = os.path.join(self.ball_dir, vid, cid)
                    for t in range(self.max_frame):
                        frame_dir = os.path.join(ball_dir, '{:06d}'.format(t))
                        with open(frame_dir, 'r') as f:
                            ball_data.append(json.load(f))
                    balls = self.read_object(ball_data)
                    res_object.append(balls)
                    
        res_skeleton = np.array(res_skeleton)
            
        os.makedirs(self.out_folder, exist_ok=True)
        # Save human skeletons
        if self.pose:
            np.save(
                os.path.join(
                    self.out_folder, '{}_data.npy'.format(phase)
                ), 
                res_skeleton
            )
            with open(os.path.join(self.out_folder, '{}_trackings.pkl'.format(phase)), 'wb') as f:
                pickle.dump(trackings, f)
        
        # Save ball keypoints
        if self.ball:
            np.save(
                os.path.join(
                    self.out_folder, '{}_object_data'.format(phase)
                ),
                res_object
            )
        
        with open(os.path.join(self.out_folder, '{}_label.pkl').format(phase), 'wb') as f:
            pickle.dump(labels, f)

    def start(self):
        for phase in ['train', 'eval']:
            logging.info('Phase: {}'.format(phase))
            self.gendata(phase)