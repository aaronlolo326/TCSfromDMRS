import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker

import model.model as module_arch

from pprint import pprint
import time
import os
import json

from torch.distributed.algorithms.join import Join
import torch.distributed as dist

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, world_size, encoder, decoder, pred2ix, pred_func2ix, criterion, metric_ftns,
        # optimizer,
        config, device, ddp, rank,# data_loader, valid_data_loader = None,
        # lr_scheduler = None,
        len_epoch = None
    ):
        super().__init__(
            world_size, encoder, decoder, pred2ix, pred_func2ix, criterion, metric_ftns,
            config, ddp, device, rank
        )
            
        self.log_step = self.grad_accum_step #int(np.sqrt(data_loader.batch_size))
        self.eval_percent = 0.02

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)


    def _train_epoch(self, epoch):

        if True:

            batch_idx = -1
            pred_func_used_accum = set()
            loss_sum = 0
            print ("len_epoch of rank {}: {}".format(self.rank, self.len_epoch))

            self.encoder.train()
            self.decoder.train()
            # for sem_func_ix in range(len(self.decoder.sem_funcs)):
            #     self.decoder.sem_funcs[sem_func_ix].train()
            t0 = time.time()

            for instance_batch in self.data_loader:
                batch_idx += 1
                instance_batch["encoder"] = [tsr.to(self.device) for tsr in instance_batch["encoder"]]
                if self.config['decoder_arch']['args']['use_truth']:
                    instance_batch["decoder"] = [tsr.to(self.device) for tsr in instance_batch["decoder"]]
                batch_log_truth, kl_div, l2_norm_reg, pos_sum, neg_sum, mu_batch, sigma2_batch = self.autoencoder.run(**instance_batch)
                if epoch <= self.autoencoder.end_beta_epoch:
                    beta =  self.autoencoder.start_beta + (self.autoencoder.end_beta - self.autoencoder.start_beta) * (batch_idx / (self.len_epoch - 1)) * (epoch / self.autoencoder.end_beta_epoch)
                else:
                    beta = self.autoencoder.end_beta
                elbo = batch_log_truth - beta * kl_div
                loss = -elbo + l2_norm_reg
                loss_sum += loss

                if any([
                    (batch_idx + 1) % self.grad_accum_step == 0,
                    (batch_idx + 1) == self.len_epoch
                ]):
                    loss_avg = loss_sum / self.grad_accum_step 
                    if 'relpron' not in self.config['name'] and (batch_idx + 1) % (self.grad_accum_step * 10) == 0 or (batch_idx + 1) == self.len_epoch:
                        print ('Train Epoch: {} {} avg Loss: {:.2f} log-T: {:.2f} p: {:.2f} n: {:.2f} u: {:2f} sig: {:2f} KL: {:.2f}'.format(
                            epoch,
                            self._progress(batch_idx),
                            loss_avg.item(),
                            batch_log_truth.item(),
                            pos_sum.item(),
                            neg_sum.item(),
                            torch.max(torch.abs(mu_batch)).item(),
                            torch.mean(sigma2_batch).item(),
                            kl_div.item(),
                            # max([len(instance_batch['decoder'][1][inst_idx]) for inst_idx in range(len(instance_batch['decoder'][1]))])
                        ))
                    loss_avg.backward()
                    self.encoder_opt.step()
                    self.encoder_opt.zero_grad(set_to_none = True)
                    self.decoder_opt.step()
                    self.decoder_opt.zero_grad(set_to_none = True)

                    loss_sum = 0
                
                # evaluate per 5%
                if batch_idx % int(self.len_epoch * self.eval_percent) == 0 or batch_idx + 1 == self.len_epoch:
                    self.encoder.eval()
                    self.decoder.eval()
                    # for sem_func_ix in range(len(self.decoder.sem_funcs)):
                    #     self.decoder.sem_funcs[sem_func_ix].eval()
                    if (not self.ddp or self.rank == 0):
                        if any([
                            # epoch == 1 and batch_idx == 0,
                            self.config['sample_only'] == False and not 'relpron' in self.config['name'],
                            # self.config['sample_only'] == True and batch_idx + 1 == 1,
                            all([
                                'relpron' in self.config['name'],
                                batch_idx + 1 == self.len_epoch,
                                (epoch - 1) % 20 == 0 and epoch > 1
                            ])
                        ]):
                            results_metrics, _ = self.evaluator.eval(epoch, batch_idx, self.len_epoch)
                            pprint (results_metrics)
                            t1 = time.time()
                            print ("Estimated hrs for an epoch: {}".format((t1 - t0) / self.eval_percent / 60 / 60))
                            t0 = t1
                        if self.config['sample_only'] == False and not 'relpron' in self.config['name']:
                            self._save_checkpoint(epoch, int(batch_idx * 100 / self.len_epoch), save_best = False)
                    self.encoder.train()
                    self.decoder.train()
                    # for sem_func_ix in range(len(self.decoder.sem_funcs)):
                    #     self.decoder.sem_funcs[sem_func_ix].train()
                    if self.ddp:
                        dist.barrier()


    def _train_epoch_ddp(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        t0 = time.time()

        self.encoder.train()
        self.decoder.train()
        # for sem_func in self.decoder.sem_funcs:
        #     sem_func.train()
        self.train_metrics.reset()
        self.encoder_opt.zero_grad()
        self.decoder_opt.zero_grad()
        # for opt in self.sem_funcs_opt:
        #     opt.zero_grad()
        
        self._train_epoch(epoch)
            
        # log = None
        log = self.train_metrics.result()

        if self.encoder_lr_scheduler is not None:
            self.encoder_lr_scheduler.step()
        if self.decoder_lr_scheduler is not None:
            self.decoder_lr_scheduler.step()

        t1 = time.time()
        print ("Time used for an epoch: {}s".format(str(t1-t0)))
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)