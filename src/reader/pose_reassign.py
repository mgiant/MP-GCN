import numpy as np
from collections import defaultdict

def read_pose_tracking(clip_data, T, M, V, C):
    keypoints = np.zeros((T, M, V, C))
    trackings = [[-1] * M for _ in range(T)]

    data = defaultdict(list)
    for t in range(T):
        for m in range(len(clip_data[t])):
            if 'track_id' not in clip_data[t][m]:
                continue
            track_id = clip_data[t][m]['track_id']
            kpts = np.zeros((V, 2))
            for v in range(V):
                kpts[v, 0] = clip_data[t][m]['keypoints']['x'][v]
                kpts[v, 1] = clip_data[t][m]['keypoints']['y'][v]
            data[track_id].append(kpts)
    energy = dict()
    for track_id in data:
        energy[track_id] = get_nonzero_std(np.array(data[track_id]))

    for t in range(T):
        people = clip_data[t]
        if len(people) == 0:
            continue
        scores = np.array(
            [energy[x['track_id']] if 'track_id' in x else 0 for x in people])
        scores -= np.max(scores)
        scores = np.exp(scores) / np.sum(np.exp(scores))
        # e = e / max(energy.values())
        confs = np.array([x['confidence'] for x in people])
        idxs = np.argsort(confs + scores)[::-1][:M]
        people = [people[i] for i in idxs]
        # people = sorted(people, key= lambda x: x['confidence'] + energy[x['track_id']]/max(energy.values()) if 'track_id' in x else 0, reverse=True)[:M]

        id_to_idx = dict()
        # First assignment
        for i, person in enumerate(people):
            try:
                track_id = person['track_id']
            except:
                track_id = 1
            m = (track_id - 1) % M
            if trackings[t][m] == -1 or trackings[t][m] > track_id:
                trackings[t][m] = track_id
            id_to_idx[track_id] = i
        # Second assignment
        pending = []
        for i in range(M):
            if trackings[t][i] == -1:
                pending.append(i)
            elif trackings[t][i] > -1:
                human_id = trackings[t][i]
                person = people[id_to_idx[human_id]]
                for v in range(V):
                    keypoints[t, i, v, 0] = person['keypoints']['x'][v]
                    keypoints[t, i, v, 1] = person['keypoints']['y'][v]
                    keypoints[t, i, v,
                                2] = person['keypoints']['visible'][v]
                id_to_idx.pop(human_id)
        for i, human_id in enumerate(id_to_idx):
            m = pending[i]
            person = people[id_to_idx[human_id]]
            for v in range(V):
                keypoints[t, m, v, 0] = person['keypoints']['x'][v]
                keypoints[t, m, v, 1] = person['keypoints']['y'][v]
                keypoints[t, m, v, 2] = person['keypoints']['visible'][v]
            trackings[t][m] = human_id

    return keypoints, trackings

def get_nonzero_std(s):  # (T,V,C)
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        # s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
        s = s[:, :, 0].std() + s[:, :, 1].std()  # three channels
    else:
        s = 0
    return s