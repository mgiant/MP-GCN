import numpy as np
import logging

def graph_processing(data, graph, processing):
    C, T, V, M = data.shape
    num_person = 1 if len(graph.split('-')) == 1 else int(graph.split('-')[1])
    
    if num_person > 1:
        if processing == 'default':
            multi_person_data = np.zeros([C, T, V*M, 1])
            for i in range(M):
                multi_person_data[:, :, V*i:V*i+V, 0] = data[:, :, :, i]
        elif processing == 'two-group':
            multi_person_data = np.zeros([C, T, V*M//2, 2])
            for i in range(M//2):
                multi_person_data[:, :, V*i:V*i+V, 0] = data[:, :, :, i]
                multi_person_data[:, :, V*i:V*i+V, 1] = data[:, :, :, i+M//2]
        else:
            logging.info('')
            logging.error('Error: Wrong in loading processing configs')
            raise ValueError()
        return multi_person_data
    return data

def multi_input(data, conn, inputs, centers):
    C, T, V, M = data.shape
    joint = np.zeros((C*2, T, V, M))
    joint_motion = np.zeros((C*2, T, V, M))
    bone = np.zeros((C*2, T, V, M))
    bone_motion = np.zeros((C*2, T, V, M))
    joint[:C, :, :, :] = data
    for i in range(V):
        center = centers[i]
        if center >= 0:
            joint[C:, :, i, :] = data[:, :, i, :] - data[:, :, center, :]
    for i in range(T-2):
        joint_motion[:C, i, :, :] = data[:, i+1, :, :] - data[:, i, :, :]
        joint_motion[C:, i, :, :] = data[:, i+2, :, :] - data[:, i, :, :]
    for i in range(len(conn)):
        if conn[i] >= 0:
            bone[:C, :, i, :] = data[:, :, i, :] - data[:, :, conn[i], :]
    for i in range(T-2):
        bone_motion[:C, i, :, :] = bone[:C, i+1, :, :] - bone[:C, i, :, :]
        bone_motion[C:, i, :, :] = bone[:C, i+2, :, :] - bone[:C, i, :, :]
    bone_length = 0
    for i in range(C):
        bone_length += bone[i, :, :, :] ** 2
    bone_length = np.sqrt(bone_length) + 0.0001
    for i in range(C):
        bone[C+i, :, :, :] = np.arccos(bone[i, :, :, :] / bone_length)

    data_new = []
    if inputs.isupper():
        if 'J' in inputs:
            data_new.append(joint)
        if 'V' in inputs:
            data_new.append(joint_motion)
        if 'B' in inputs:
            data_new.append(bone)
        if 'M' in inputs:
            data_new.append(bone_motion)
    elif inputs == 'joint':
        data_new = [joint[:C, :, :, :]]
    elif inputs == 'bone':
        data_new = [bone[:C, :, :, :]]
    elif inputs == 'motion':
        data_new = [joint_motion[:C, :, :, :]]
    elif inputs == 'bone_motion':
        data_new = [bone_motion[:C, :, :, :]]
    else:
        logging.info('')
        logging.error('Error: No input feature!')
        raise ValueError()
    return np.stack(data_new, axis=0)

