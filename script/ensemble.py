import os
import numpy as np
import argparse

def ensemble_nba(items, labels, alpha):
    num_classes = 9
    total_num = len(labels)
    cnt_top1 = 0
    cm = np.zeros(shape=(num_classes, num_classes))
    for i in range(total_num):
        name, label = labels[i][1], labels[i][0]
        r = np.zeros_like(items[0][name])
        for j, r0 in enumerate(items):
            r += np.array(r0[name])*alpha[j]
        cm[int(label)][np.argmax(r)] += 1
        # cnt_top1 += int(int(label) == np.argmax(r))

    cnt_top1 = 0
    mpca = 0
    for i in range(num_classes):
        cnt_top1 += cm[i][i]
        mpca += cm[i][i] / np.sum(cm[i])
            
    mca = cnt_top1 / total_num
    mpca = mpca / num_classes
    return mca*100, mpca*100

def ensemble_k400(items, labels, alpha):
    num_classes = 400
    total_num = len(labels)
    cnt_top1 = 0
    cm = np.zeros(shape=(num_classes, num_classes))
    for i in range(total_num):
        name, label = labels[i][1], labels[i][0]
        r = np.zeros_like(items[0][name])
        for j, r0 in enumerate(items):
            r += np.array(r0[name])*alpha[j]
        cm[int(label)][np.argmax(r)] += 1
        cnt_top1 += int(int(label) == np.argmax(r))
            
    mca = cnt_top1 / total_num
    return mca*100, 0


def ensemble_volleyball(items, labels, alpha):
    total_num = len(labels)
    cnt_top1 = 0
    for i in range(total_num):
        name, label = labels[i][1], labels[i][0]
        r = np.zeros_like(items[0][name])
        for j, r0 in enumerate(items):
            r += np.array(r0[name])*alpha[j]
        cnt_top1 += int(int(label) == np.argmax(r))

    acc = cnt_top1 / total_num
    return acc*100, 0


def load_score(path, file_name):
    # with open(os.path.join(path, file_name), 'rb') as f:
    #     return pickle.load(f)
    return np.load(os.path.join(path, file_name), allow_pickle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    alpha = [0.4, 0.4, 0.2, 0.2]

    parser.add_argument('--joint', '-j', default='')
    parser.add_argument('--bone', '-b', default='')
    parser.add_argument('--joint_motion', '-jm', default='')
    parser.add_argument('--bone_motion', '-bm', default='')
    parser.add_argument('--label_path', '-l', default=None)

    arg = parser.parse_args()
    
    joint_dir = arg.joint
    bone_dir = arg.bone
    jm_dir = arg.joint_motion
    bm_dir = arg.bone_motion
    label_path = arg.label_path
    
    if 'nba' in label_path.lower():
        ensemble = ensemble_nba
    elif 'volleyball' in label_path.lower():
        ensemble = ensemble_volleyball
    elif 'k400' in label_path:
        ensemble = ensemble_k400

    labels = np.load(label_path, allow_pickle=True)
    if arg.joint_motion is not None and arg.bone_motion is not None:
        pass
    file_name = 'score.pkl'
    if os.path.exists(joint_dir):
        joint = load_score(joint_dir, file_name)
    if os.path.exists(bone_dir):
        bone = load_score(bone_dir, file_name)
    if os.path.exists(jm_dir):
        jm = load_score(jm_dir, file_name)
    if os.path.exists(bm_dir):
        bm = load_score(bm_dir, file_name)

    if 'joint' in locals():
        mca, mpca = ensemble([joint], labels, alpha)
        print('joint: {:.2f}, MPCA: {:.2f}'.format(mca, mpca))
    if 'bone' in locals():
        mca, mpca = ensemble([bone], labels, alpha)
        print('bone: {:.2f}, MPCA: {:.2f}'.format(mca, mpca))
    if 'jm' in locals():
        mca, mpca = ensemble([jm], labels, alpha)
        print('jm: {:.2f}, MPCA: {:.2f}'.format(mca, mpca))
    if 'bm' in locals():
        mca, mpca = ensemble([bm], labels, alpha)
        print('bm: {:.2f}, MPCA: {:.2f}'.format(mca, mpca))

    if 'joint' in locals() and 'bone' in locals():
        items = [joint, bone]
        mca, mpca = ensemble(items, labels, alpha)
        print('joint+bone: {:.2f}, MPCA: {:.2f}'.format(mca, mpca))
    if 'joint' in locals() and 'bone' in locals() and 'jm' in locals(): 
        items = [joint, bone, jm]
        mca, mpca = ensemble(items, labels, alpha)
        print('joint+bone+jm: {:.2f}, MPCA: {:.2f}'.format(mca, mpca))
    if 'joint' in locals() and 'bone' in locals() and 'jm' in locals() and 'bm' in locals():
        items = [joint, bone, jm, bm]
        mca, mpca = ensemble(items, labels, alpha)
        print('joint+bone+jm+bm: {:.2f}, MPCA: {:.2f}'.format(mca, mpca))
