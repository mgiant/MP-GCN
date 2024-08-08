import os
import json
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='Calculate mean and std accuracy for each config')
parser.add_argument('--config', '-c', type=str, default='abc', help='Using config')
parser.add_argument('--workdir', '-w', type=str, default='abc', help='Using workdir')
args, _ = parser.parse_known_args()

if os.path.exists(args.config):
    root_dir = args.config.replace('configs', 'workdir').replace('.yaml', '')
elif os.path.exists(args.workdir):
    root_dir = args.workdir
runs = os.listdir(root_dir)

acc_dict = {
    'acc_best': [],
    'acc_last': [],
    'runs': []
}
for run in runs:
    acc_path = os.path.join(root_dir, run, 'result.json')
    if not os.path.exists(acc_path):
        continue

    with open(acc_path, 'r') as f:
        acc = json.load(f)
        acc_dict['acc_best'].append(float(acc.get('acc_top1')*100))
        acc_dict['acc_last'].append(float(acc.get('acc_top1_last')*100))
        acc_dict['runs'].append(run)

accs_best = np.array(acc_dict['acc_best'])
accs_last = np.array(acc_dict['acc_last'])
acc_dict['acc_best'] = "{:.2f}^{:.2f}".format(accs_best.mean(), accs_best.std())
acc_dict['acc_last'] = "{:.2f}^{:.2f}".format(accs_last.mean(), accs_last.std())

save_dir = os.path.join(root_dir, 'avg.json')
with open(save_dir, 'w') as f:
    json.dump(acc_dict, f)
