import os
import importlib
import sys
import shutil
import logging
import json
import torch
from time import time, strftime, localtime


def import_class(name):
    components = name.split('.')
    mod = importlib.import_module(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def get_time(total_time):
    s = int(total_time % 60)
    m = int(total_time / 60) % 60
    h = int(total_time / 60 / 60) % 24
    d = int(total_time / 60 / 60 / 24)
    return '{:0>2d}d-{:0>2d}h-{:0>2d}m-{:0>2d}s'.format(d, h, m, s)


def get_current_timestamp():
    ct = time()
    ms = int((ct - int(ct)) * 1000)
    return '[ {},{:0>3d} ] '.format(strftime('%Y-%m-%d %H:%M:%S', localtime(ct)), ms)


def load_checkpoint(work_dir, model_name):
    if model_name == 'debug':
        file_name = '{}/temp/debug.pth.tar'.format(work_dir)
    else:
        file_name = '{}/{}.pth.tar'.format(work_dir, model_name)
    try:
        checkpoint = torch.load(file_name, map_location=torch.device('cpu'))
    except:
        logging.info('')
        logging.error(
            'Error: Wrong in loading this checkpoint: {}!'.format(file_name))
        raise ValueError()
    return checkpoint


def load_model(work_dir, model_name):
    if os.path.exists(work_dir):
        model_file = os.path.join(work_dir, model_name+'.pth.tar')
        if os.path.exists(model_file):
            try:
                checkpoint = torch.load(
                    model_file, map_location=torch.device('cpu'))
            except:
                logging.info('')
                logging.error(
                    'Error: Wrong in loading this checkpoint: {}!'.format(model_file))
                raise ValueError()
        else:
            logging.info('')
            logging.error('Error: File {} does not exist!'.format(model_file))
            raise ValueError()
    else:
        logging.info('')
        logging.error('Error: {} does not exist!'.format(work_dir))
        raise ValueError()
    return checkpoint


def save_checkpoint(model, optimizer, scheduler, epoch, best_state, is_best, work_dir, save_dir, model_name):
    for key in model.keys():
        model[key] = model[key].cpu()
    checkpoint = {
        'model': model, 'optimizer': optimizer, 'scheduler': scheduler,
        'best_state': best_state, 'epoch': epoch,
    }
    cp_name = '{}/checkpoint.pth.tar'.format(work_dir)
    torch.save(checkpoint, cp_name)
    if is_best:
        shutil.copy(cp_name, '{}/{}.pth.tar'.format(save_dir, model_name))
    with open('{}/result.json'.format(save_dir), 'w') as f:
        out = {k: best_state[k] for k in best_state if k != 'cm'}
        json.dump(out, f)

def set_logging(args):
    if args.debug or args.evaluate or args.extract or args.generate_data:
        save_dir = '{}/temp'.format(args.work_dir)
    elif args.resume:
        save_dir = args.work_dir
    else:
        config = args.config.split('/')[1:]
        config[-1] = config[-1].replace('.yaml', '')
        config = '/'.join(config)
        ct = strftime('%Y-%m-%d %H-%M-%S')
        save_dir = '{}/{}/{}'.format(args.work_dir, config, ct)
    os.makedirs(save_dir, exist_ok=True)
    log_format = '[ %(asctime)s ] %(message)s'
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO, format=log_format)
    handler = logging.FileHandler(
        '{}/log.txt'.format(save_dir), mode='w', encoding='UTF-8')
    handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(handler)
    return save_dir
