import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker

import model.model as module_arch

from pprint import pprint


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, encoder, pred2ix, decoders, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(encoder, decoders, criterion, metric_ftns, optimizer, config)
        self.pred2ix = pred2ix
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
            
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.encoder.train()
        [decoder.train() for _, decoder in self.decoders.items()]
        self.train_metrics.reset()
        for batch_idx, instance_batch in enumerate(self.data_loader):
            # Currently only support batch_size = 1
            assert len(instance_batch) == 1
            pprint (instance_batch)
        #     # for instance in batch_data:
        #     #     pprint (instance)
        #     #     instance = {
        #     #         "discarded": False
        #     #         "node2pred": self.node2pred,
        #     #         "encoders": {
        #     #             "pred_func_node": self.pred_func_node,
        #     #             "lexical_preds": self.lexical_preds
        #     #         },
        #     #         "decoders": {
        #     #             "logic_expr": self.logic_expr
        #     #         }
        #     #     }
            node2pred_batch = [instance['node2pred'] for instance in instance_batch]
            encoder_data_batch = [instance['encoders'] for instance in instance_batch]
            decoders_data_batch = [instance['decoders'] for instance in instance_batch]

            # TODO: what if targ in lex?
            targ_preds_ix_batch = torch.LongTensor([
                [self.pred2ix[node2pred_batch[inst_idx][targ_node]] for targ_node in data['pred_func_node']]
            for inst_idx, data in enumerate(encoder_data_batch)]).unsqueeze(dim = 2)

            targ_preds_ix_batch = targ_preds_ix_batch.to(self.device)
            
            lex_preds_ix_batch = torch.LongTensor([
                [self.pred2ix[lex_pred]
                    for lex_pred in data['lexical_preds']]
            for data in encoder_data_batch]).unsqueeze(dim = 1)

            lex_preds_ix_batch = lex_preds_ix_batch.to(self.device)

            mu_batch, log_sigma2_batch = self.encoder(targ_preds_ix_batch, lex_preds_ix_batch)
            
            print (mu_batch)
            print (log_sigma2_batch)
            input ()
                
             
            # sample z ~ q_phi(dmrs_i)
            # mu, sigma = 
            # calculate grad(p_theta(dmrs_i | z)
            # update phi
            # update theta
            
            
#         self.model.train()
#         self.train_metrics.reset()
#         for batch_idx, (data, target) in enumerate(self.data_loader):
#             data, target = data.to(self.device), target.to(self.device)

#             self.optimizer.zero_grad()
#             output = self.model(data)
#             loss = self.criterion(output, target)
#             loss.backward()
#             self.optimizer.step()

#             self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
#             self.train_metrics.update('loss', loss.item())
#             for met in self.metric_ftns:
#                 self.train_metrics.update(met.__name__, met(output, target))

#             if batch_idx % self.log_step == 0:
#                 self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
#                     epoch,
#                     self._progress(batch_idx),
#                     loss.item()))
#                 self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

#             if batch_idx == self.len_epoch:
#                 break
#         log = self.train_metrics.result()

#         if self.do_validation:
#             val_log = self._valid_epoch(epoch)
#             log.update(**{'val_'+k : v for k, v in val_log.items()})

#         if self.lr_scheduler is not None:
#             self.lr_scheduler.step()
#         return log

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