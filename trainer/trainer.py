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

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, world_size, encoder, decoder, pred2ix, pred_func2ix, criterion, metric_ftns,
        # optimizer,
        config, device, ddp, cpu,# data_loader, valid_data_loader = None,
        # lr_scheduler = None,
        len_epoch = None
    ):
        super().__init__(
            world_size, encoder, decoder, pred2ix, pred_func2ix, criterion, metric_ftns,
            config, ddp, cpu, device
        )
        # if len_epoch is None:
        #     # epoch-based training
        #     self.len_epoch = len(self.data_loader)
        # else:
        #     # iteration-based training
        #     self.data_loader = inf_loop(self.data_loader)
        #     self.len_epoch = len_epoch
            
        self.log_step = self.grad_accum_step #int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)


    def _train_epoch(self, epoch):

        # if True:
        batch_idx = -1
        pred_func_used_accum = set()
        loss_sum = 0
        print ("len_epoch of {}: {}".format(self.device, self.len_epoch))
        for instance_batch in self.data_loader:
            batch_idx += 1
            # print ("batch_idx:", batch_idx)
            # print (self.device, instance_batch)
            pred_func_used_accum.update(instance_batch["pred_func_used_accum"])
            # print ("b")
            instance_batch["encoder"] = [tsr.to(self.device) for tsr in instance_batch["encoder"]]
            # print ("c")



            batch_log_truth, kl_div = self.autoencoder.run(**instance_batch)
            beta =  self.autoencoder.start_beta + (self.autoencoder.end_beta - self.autoencoder.start_beta) * (batch_idx / (self.len_epoch - 1))
            elbo = batch_log_truth - beta * kl_div
            loss = -elbo
            loss_sum += loss / self.grad_accum_step 
            # print ("loss_sum:", loss_sum)

            # print ("pred_func_used_accum:", pred_func_used_accum)

            if any([
                (batch_idx + 1) % self.grad_accum_step == 0,
                (batch_idx + 1) == self.len_epoch
            ]):
                loss_avg = loss_sum
                print ('Train Epoch: {} {} \t avg Loss: {} \t log-T: {} \t KL: {}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss_avg.item(),
                    batch_log_truth.item(),
                    kl_div.item(),
                    # max([len(instance_batch['decoder'][1][inst_idx]) for inst_idx in range(len(instance_batch['decoder'][1]))])
                ))
                # self.optimizer.step()
                # self.optimizer.zero_grad()
                # print ("loss avg", loss_avg,  self.device)
                # print("a", self.device, self.encoder.module.fc_pair.weight.grad)
                loss_avg.backward()
                # print ("backward",  self.device)
                # print("aa", self.device, self.encoder.module.fc_pair.weight.grad)
                self.encoder_opt.step()
                # print ("encoder_opt step", self.device)
                # print("aaa", self.device, self.encoder.module.fc_pair.weisght.grad)
                self.encoder_opt.zero_grad(set_to_none = True)
                # print ("encoder_opt zero",  self.device)
                # print("aaaa", self.device, self.encoder.module.fc_pair.weight.grad)
                for sem_func_idx, sem_func in enumerate(self.decoder.sem_funcs):#sem_func_idx, enumerate(pred_func_used_accum):
                    self.sem_funcs_opt[sem_func_idx].step()
                    # print ("sem_func_idx {}/{} step".format(sem_func_idx, len(pred_func_used_accum)))
                    self.sem_funcs_opt[sem_func_idx].zero_grad(set_to_none = True)
                #     # print ("sem_func_idx {}/{} zero".format(sem_func_idx, len(pred_func_used_accum)))
                # print ("sem_funcs step and zero", self.device)
                # print ("sem_funcs step and zero", self.device)
                # pred_func_used_accum = set()
                loss_sum = 0

                # self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                # self.train_metrics.update('training_loss', loss.item())
                # self.train_metrics.update('log_fuzzy_truthness', log_fuzzy_truthness.item())
                # self.train_metrics.update('kl_div', kl_div.item())
            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(loss))

            # if (batch_idx + 1) % self.log_step == 0:
                # self.logger.debug('Train Epoch: {} {} \t avg Loss: {:.6f} \t T: {:.6f} \t log-T: {:.6f} \t KL: {:.6f} \t max #pf: {}'.format(
                #     epoch,
                #     self._progress(batch_idx),
                #     loss_avg.item(),
                #     torch.exp(batch_log_truth).item(),
                #     batch_log_truth.item(),
                #     kl_div.item(),
                #     max([len(instance_batch['decoder'][1][inst_idx]) for inst_idx in range(len(instance_batch['decoder'][1]))])
                # ))
                # print('_song_n_of' in list(instance_batch[0]['node2pred'].values()))
                # print(self.decoder.sem_funcs['_song_n_of@ARG0'])
                # print(self.decoder.sem_funcs['_song_n_of@ARG0'].fc1.weight.grad)
                # pprint(list(self.decoder.sem_funcs['_song_n_of@ARG0'].parameters()))
                # input()
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            
            # evaluate per 5%
            if not self.ddp or self.device == 0 and batch_idx % int(self.len_epoch / 20) == 0:
                results_metrics = self.evaluator.eval(self.results_dir, epoch, batch_idx, self.len_epoch)
                pprint (results_metrics)
                    # print ("#hyp pairs:", len(results_load['hyp']['hyp_pred_pairs_t1_f0']))

            if batch_idx == self.len_epoch:
                break

            # break

    def _train_epoch_ddp(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        # if 
        t0 = time.time()

        # if self.ddp:
        # TODO: uncomment if using DistributedSampler
        #     self.data_loader.sampler.set_epoch(epoch)

        self.encoder.train()
        # print ("encoder train")
        for sem_func in self.decoder.sem_funcs:
            sem_func.train()
        # print ("sem_funcs train")
        self.train_metrics.reset()
        # print ("len: {}".format(len(self.data_loader)))

        self.encoder_opt.zero_grad()
        # print ("encoder_opt zero_grad")
        for opt in self.sem_funcs_opt:
            opt.zero_grad()
        # print ("sem_funcs zero_grad")
        
        # TODO
        if self.ddp and False:
            with Join([
                self.autoencoder.encoder,
                *self.autoencoder.decoder.sem_funcs
                # self.encoder_opt,
                # *self.sem_funcs_opt
            ]):
                print ("joined")
                self._train_epoch(epoch)
        else:
            self._train_epoch(epoch)
        
            
        # log = None
        log = self.train_metrics.result()

        # if self.do_validation:
        #     val_log = self._valid_epoch(epoch)
        #     log.update(**{'val_'+k : v for k, v in val_log.items()})

        # if self.lr_scheduler is not None:
        #     self.lr_scheduler.step()

        if self.encoder_lr_scheduler is not None:
            self.encoder_lr_scheduler.step()

        for lr_scheduler in self.sem_funcs_lr_scheduler:
            if lr_scheduler is not None:
                lr_scheduler.step()
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

    def eval(self, epoch, batch_idx):

        t0 = time.time()

        self.encoder.eval()
        
        for sem_func in self.decoder.sem_funcs:
            sem_func.eval()

        self.valid_metrics.reset()

        with torch.no_grad():
            for batch_idx, data in enumerate(self.eval_data_loader):
                pass

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)