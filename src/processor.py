import logging
import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from time import time

from . import utils
from .initializer import Initializer


class Processor(Initializer):

    def train(self, epoch):
        self.model.train()
        num_top1, num_sample = 0, 0
        train_iter = tqdm(self.train_loader, dynamic_ncols=True)
        for num, (x, y, _) in enumerate(train_iter):
            self.optimizer.zero_grad()

            # Using GPU
            x = x.float().to(self.device)
            y = y.long().to(self.device)

            # Calculating Output
            out, _ = self.model(x)

            # Updating Weights
            loss = self.loss_func(out, y)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            # Calculating Recognition Accuracies
            num_sample += x.size(0)
            reco_top1 = out.max(1)[1]
            num_top1 += reco_top1.eq(y).sum().item()

            # Showing Progress
            lr = self.optimizer.param_groups[0]['lr']
            if self.scalar_writer:
                self.scalar_writer.add_scalar(
                    'learning_rate', lr, self.global_step)
                self.scalar_writer.add_scalar(
                    'train_loss', loss.item(), self.global_step)
                
            train_iter.set_description(
                'Loss: {:.4f}, LR: {:.4f}'.format(loss.item(), lr))

        # Showing Train Results
        train_acc = num_top1 / num_sample
        if self.scalar_writer:
            self.scalar_writer.add_scalar(
                'train_acc', train_acc, self.global_step)
        logging.info('Epoch: {}/{}, Training accuracy: {:d}/{:d}({:.2%})'.format(
            epoch +
            1, self.max_epoch, num_top1, num_sample, train_acc
        ))
        logging.info('')

    def eval(self, save_score=True):
        self.model.eval()
        start_eval_time = time()
        score = {}
        with torch.no_grad():
            num_top1, num_top5 = 0, 0
            num_sample, eval_loss = 0, []
            cm = np.zeros((self.num_class, self.num_class))
            eval_iter = tqdm(self.eval_loader, dynamic_ncols=True)
            for num, (x, y, name) in enumerate(eval_iter):

                # Using GPU
                x = x.float().to(self.device)
                y = y.long().to(self.device)

                # Calculating Output
                out, _ = self.model(x)

                # Getting Loss
                loss = self.loss_func(out, y)
                eval_loss.append(loss.item())

                if save_score:
                    for n, c in zip(name, out.detach().cpu().numpy()):
                        score[n] = c

                # Calculating Recognition Accuracies
                num_sample += x.size(0)
                reco_top1 = out.max(1)[1]
                num_top1 += reco_top1.eq(y).sum().item()
                reco_top5 = torch.topk(out, min(5, self.num_class))[1]
                num_top5 += sum([y[n] in reco_top5[n, :]
                                for n in range(x.size(0))])

                # Calculating Confusion Matrix
                for i in range(x.size(0)):
                    cm[y[i], reco_top1[i]] += 1


        # Showing Evaluating Results
        acc_top1 = num_top1 / num_sample
        acc_top5 = num_top5 / num_sample
        # MPCA
        acc_perclass = [0] * self.num_class
        for i in range(self.num_class):
            num_c = sum(cm[i])
            acc_perclass[i] = cm[i, i] / num_c
        mpca = sum(acc_perclass) / self.num_class
        eval_loss = sum(eval_loss) / len(eval_loss)
        eval_time = time() - start_eval_time
        eval_speed = len(self.eval_loader) * \
            self.eval_batch_size / eval_time / len(self.args.gpus)
        logging.info('Top-1 accuracy: {:d}/{:d}({:.2%}), MPCA: {:.2%}, Top-5 accuracy: {:d}/{:d}({:.2%}), Mean loss:{:.4f}'.format(
            num_top1, num_sample, acc_top1, mpca, num_top5, num_sample, acc_top5, eval_loss
        ))
        logging.info('Evaluating time: {:.2f}s, Speed: {:.2f} sequnces/(second*GPU)'.format(
            eval_time, eval_speed
        ))
        logging.info('')
        if self.scalar_writer:
            self.scalar_writer.add_scalar(
                'eval_acc', acc_top1, self.global_step)
            self.scalar_writer.add_scalar(
                'eval_loss', eval_loss, self.global_step)

        torch.cuda.empty_cache()

        return acc_top1, acc_top5, cm, score

    def start(self):
        start_time = time()
        if self.args.evaluate:
            if self.args.debug:
                logging.warning('Warning: Using debug setting now!')
                logging.info('')

            # Loading Evaluating Model
            logging.info('Loading evaluating model ...')
            checkpoint = utils.load_checkpoint(
                self.args.work_dir, self.model_name)
            if checkpoint:
                self.model.module.load_state_dict(checkpoint['model'])
            logging.info('Successful!')
            logging.info('')

            # Evaluating
            logging.info('Starting evaluating ...')
            self.eval()
            logging.info('Finish evaluating!')

        else:
            # Resuming
            start_epoch = 0
            best_state = {'acc_top1': 0, 'acc_top5': 0,
                          'acc_top1_last': 0,
                          'cm': 0, 'best_epoch': 0}
            best_score = {}
            if self.args.resume:
                logging.info('Loading checkpoint ...')
                checkpoint = utils.load_checkpoint(self.args.work_dir, self.model_name)
                self.model.module.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch']
                best_state.update(checkpoint['best_state'])
                self.global_step = start_epoch * len(self.train_loader)
                logging.info('Start epoch: {}'.format(start_epoch+1))
                logging.info('Best accuracy: {:.2%}'.format(
                    best_state['acc_top1']))
                logging.info('Successful!')
                logging.info('')

            # Training
            logging.info('Starting training ...')
            for epoch in range(start_epoch, self.max_epoch):

                # Training
                self.train(epoch)

                # Evaluating
                is_best = False
                if (epoch+1) % self.eval_interval(epoch) == 0:
                    logging.info(
                        'Evaluating for epoch {}/{} ...'.format(epoch+1, self.max_epoch))
                    acc_top1, acc_top5, cm, score = self.eval()
                    if acc_top1 > best_state['acc_top1']:
                        is_best = True
                        best_state.update(
                            {'acc_top1': acc_top1, 'acc_top5': acc_top5, 'cm': cm, 'best_epoch': epoch+1})
                        best_score = score
                    best_state.update({'acc_top1_last': acc_top1})

                # Saving Model
                logging.info(
                    'Saving model for epoch {}/{} ...'.format(epoch+1, self.max_epoch))
                utils.save_checkpoint(
                    self.model.module.state_dict(), self.optimizer.state_dict(), self.scheduler.state_dict(),
                    epoch+1, best_state, is_best, self.args.work_dir, self.save_dir, self.model_name
                )
                logging.info('Best top-1 accuracy: {:.2%}@{}th epoch, Total time: {}'.format(
                    best_state['acc_top1'], best_state['best_epoch'], utils.get_time(
                        time()-start_time)
                ))
                logging.info('')
            np.savetxt('{}/cm.csv'.format(self.save_dir),
                       cm, fmt="%s", delimiter=",")
            with open('{}/score.pkl'.format(self.save_dir), 'wb') as f:
                pickle.dump(best_score, f)
            logging.info('Finish training!')
            logging.info('')

    def extract(self):
        logging.info('Starting extracting ...')
        if self.args.debug:
            logging.warning('Warning: Using debug setting now!')
            logging.info('')

        # Loading Model
        logging.info('Loading evaluating model ...')
        checkpoint = utils.load_model(self.args.work_dir, self.model_name)
        if checkpoint:
            self.cm = checkpoint['best_state']['cm']
            self.model.module.load_state_dict(checkpoint['model'])
        logging.info('Successful!')
        logging.info('')

        eval_iter = tqdm(self.eval_loader, dynamic_ncols=True)

        self.model.eval()

        out = []
        names = []
        features = []
        labels = []
        with torch.no_grad():
            for num, (x, y, name) in enumerate(eval_iter):
                names.extend(name)
                labels.extend(y)
                x = x.float().to(self.device)

                dy, feature = self.model(x)
                features.extend(feature.detach().cpu().numpy())
                out.extend(dy.detach().cpu().numpy())

        out = torch.Tensor(out)
        out = torch.nn.functional.softmax(out, dim=1).detach().cpu().numpy()
        weight = self.model.module.fcn.weight.squeeze().detach().cpu().numpy()

        # Saving Data
        if not self.args.debug:
            save_path = os.path.join(self.args.work_dir, 'extraction.npz')
            np.savez(save_path,
                     name=names,
                     label=labels,
                     out=out,
                     feature=features,
                     weight=weight
                     )

        logging.info('Finish extracting!')
        logging.info('')
