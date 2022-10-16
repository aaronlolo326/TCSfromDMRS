import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter

import data_loader.data_loaders as module_data
import model.model as module_arch

from evaluator.evaluator import Evaluator

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer

import os
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, world_size, encoder, decoder, pred2ix, pred_func2ix, criterion, metric_ftns,
        # optimizer,
        config, ddp, cpu, device):

        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.ddp = ddp
        self.cpu = cpu
        self.world_size = world_size
        self.device = device

        self.encoder = encoder
        self.decoder = decoder
        self.pred2ix = pred2ix
        self.pred_func2ix = pred_func2ix
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        # self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.grad_accum_step = cfg_trainer['grad_accum_step']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch_ddp(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError


    def train(self, rank):
        """
        Full training logic
        """
        # if self.device !torch.cuda.set_device(self.device)

        print ("Initializing training data loader ...")
        if self.ddp:
            # setup data_loader instances
            num_replicas = self.world_size
        else:
            num_replicas = 0

        transformed_dir_suffix = os.path.join("transformed", self.config["name"])
        self.data_loader = self.config.init_obj('data_loader', module_data,
            transformed_dir_suffix = transformed_dir_suffix, pred2ix = self.pred2ix, num_replicas = num_replicas, rank = rank)# lex_pred2cnt = lex_pred2cnt, pred_func2cnt = pred_func2cnt)
        print ("# of training instances: {}".format(len(self.data_loader.dataset)))
        self.len_epoch = len(self.data_loader)


        # transformed_dir  = self.config["data_loader"]["args"]["transformed_dir"]
        # transformed_info_dir = os.path.join(transformed_dir, "info")

        # pred_func2ix_file_path = os.path.join(transformed_info_dir, "pred_func2ix.txt")
        # pred_func2ix = {}
        # with open(pred_func2ix_file_path) as f:
        #     line = f.readline()
        #     while line:
        #         ix, pred_func = line.strip().split("\t")
        #         ix = int(ix)
        #         pred_func2ix[pred_func] = ix
        #         line = f.readline()

        # pred2ix_file_path = os.path.join(transformed_info_dir, "pred2ix.txt")
        # pred2ix = {}
        # with open(pred2ix_file_path) as f:
        #     line = f.readline()
        #     while line:
        #         ix, pred = line.strip().split("\t")
        #         pred2ix[pred] = int(ix)
        #         line = f.readline()
        

        # valid_data_loader = data_loader.split_validation()
        self.valid_data_loader = None
        self.do_validation = self.valid_data_loader is not None

        if self.ddp:
            print ("Sending encoder to DDP device ...")
            encoder = self.encoder.to(rank)
            if self.cpu:
                self.encoder = DDP(encoder)
            else:
                self.encoder = DDP(encoder, device_ids=[rank]) # find_unused_parameters = True)

            print ("Sending semantic functions to DDP device ...")
            for sem_func_ix in range(len(self.decoder.sem_funcs)):
                self.decoder.sem_funcs[sem_func_ix] = self.decoder.sem_funcs[sem_func_ix].to(rank)
                if self.cpu:
                    self.decoder.sem_funcs[sem_func_ix] = DDP(self.decoder.sem_funcs[sem_func_ix])
                else:
                    self.decoder.sem_funcs[sem_func_ix] = DDP(self.decoder.sem_funcs[sem_func_ix], device_ids=[rank]) # find_unused_parameters = True)

        else:
            print ("Sending encoder to device ...")
            self.encoder = self.encoder.to(self.device)
            print ("Sending semantic functions to device ...")
            for sem_func_ix in range(len(self.decoder.sem_funcs)):
                self.decoder.sem_funcs[sem_func_ix] = self.decoder.sem_funcs[sem_func_ix].to(self.device)

        print ("Initializing autoencoder ...")
        if self.ddp:
            self.autoencoder = self.config.init_obj('autoencoder_arch', module_arch, encoder = self.encoder, decoder = self.decoder, device = rank, ddp = self.ddp) 
        else:
            self.autoencoder = self.config.init_obj('autoencoder_arch', module_arch, encoder = self.encoder, decoder = self.decoder, device = self.device, ddp = self.ddp)
        ## each sem_func has its own optimizer (This should be faster)
        print ("Initializing optimizer for the encoder and semantic functions ...")
        if self.ddp and False:
            encoder_opt_args = self.config["encoder_DDPoptimizer"]["args"]
            self.encoder_opt = ZeroRedundancyOptimizer(
                self.encoder.parameters(),
                optimizer_class = getattr(torch.optim, encoder_opt_args["optimizer_class"]),
                lr = encoder_opt_args['lr'], weight_decay = encoder_opt_args["weight_decay"], amsgrad = encoder_opt_args["amsgrad"]
            )
            sem_func_opt_args = self.config["sem_func_DDPoptimizer"]["args"]
            self.sem_funcs_opt = [
                ZeroRedundancyOptimizer(
                sem_func.parameters(),
                optimizer_class = getattr(torch.optim, sem_func_opt_args["optimizer_class"]),
                lr = sem_func_opt_args['lr'], weight_decay = sem_func_opt_args["weight_decay"], amsgrad = sem_func_opt_args["amsgrad"]
                )
                for sem_func in self.decoder.sem_funcs
            ]
        else:
            self.encoder_opt = self.config.init_obj('encoder_optimizer', torch.optim, self.encoder.parameters())
            self.sem_funcs_opt = [self.config.init_obj('sem_func_optimizer', torch.optim, sem_func.parameters()) for sem_func in self.decoder.sem_funcs]
        
        print ("Initializing learning rate scheduler for the optimizer(s) ...")
        self.encoder_lr_scheduler = self.config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.encoder_opt)
        self.sem_funcs_lr_scheduler = [self.config.init_obj('lr_scheduler', torch.optim.lr_scheduler, opt) for opt in self.sem_funcs_opt]


        if not self.ddp or rank == 0:
            print ("Initializing evaluator ...")
            # if self.ddp:
            #     self.evaluator = Evaluator(dataloaders = [eval_hyp_dataloaders], autoencoder = self.autoencoder, config = self.config, device = rank)
            # else:
            print ("Initializing evaluation data loaders ...")
            
            hyp_data_dir = self.config["eval_hyp_dataloader"]["args"]["hyp_data_dir"]
            hyp_pred_pairs_path = [os.path.join(hyp_data_dir, file) for file in os.listdir(hyp_data_dir) if os.path.isfile(os.path.join(hyp_data_dir, file))]
            hyp_pred_pairs_name = [file.rsplit(".", 1)[0] for file in os.listdir(hyp_data_dir) if os.path.isfile(os.path.join(hyp_data_dir, file))]
            eval_hyp_dataloaders = {hyp_pred_pairs_name[idx]: self.config.init_obj(
                'eval_hyp_dataloader',
                module_data,
                hyp_data_path = path, num_replicas = num_replicas, pred_func2ix = self.pred_func2ix, pred2ix = self.pred2ix
                )
                for idx, path in enumerate(hyp_pred_pairs_path)
            }

            self.evaluator = Evaluator(dataloaders = [eval_hyp_dataloaders], autoencoder = self.autoencoder, config = self.config, device = self.device)
            self.results_dir = os.path.join(
                self.config['evaluator']['results_dir'],
                self.config['name']
            )
            os.makedirs(self.results_dir, exist_ok = True)


        print ("Done. Start training")

        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch_ddp(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))


            # evaluation starts here (per epoch)                
            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                results = None
                if not self.ddp or rank == 0:
                    results = self.evaluator.eval(epoch, rank)
                    results_path = os.path.join(self.results_dir, "epoch" + str(epoch) + "_" + str(rank))
                    with open(results_path, "w") as f:
                        json.dump(results, f)
                    with open(results_path, "r") as f:
                        results_load = json.load(f)
                        print ("#hyp pairs:", len(results_load['hyp']['hyp_pred_pairs_t1_f0']))

                    try:
                        # check whether model performance improved or not, according to specified metric(mnt_metric)
                        improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                                (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                    except KeyError:
                        self.logger.warning("Warning: Metric '{}' is not found. "
                                            "Model performance monitoring is disabled.".format(self.mnt_metric))
                        self.mnt_mode = 'off'
                        improved = False

                    if improved:
                        self.mnt_best = log[self.mnt_metric]
                        not_improved_count = 0
                        best = True
                    else:
                        not_improved_count += 1

                    if not_improved_count > self.early_stop:
                        self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                        "Training stops.".format(self.early_stop))
                        break

                    if epoch % self.save_period == 0:
                        self._save_checkpoint(epoch, save_best=best)
        
        print ("Done training!")

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        encoder_arch = type(self.encoder).__name__
        state = {
            'encoder_arch': encoder_arch,
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'sem_funcs_state_dict': [sem_func.state_dict() for sem_func in self.decoder.sem_funcs],
            # 'optimizer': self.optimizer.state_dict(),
            'encoder_opt': self.encoder_opt.state_dict(),
            'sem_funcs_opt': [opt.state_dict() for opt in self.sem_funcs_opt],
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        for sem_func_ix in range(len(self.decoder.sem_funcs)):
            self.decoder.sem_func[sem_func_ix].load_state_dict(checkpoint['decoders_state_dict'][sem_func_ix])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if any([
            checkpoint['config']['encoder_optimizer']['type'] != self.config['encoder_optimizer']['type'],
            checkpoint['config']['sem_func_optimizer']['type'] != self.config['sem_func_optimizer']['type']
        ]):
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.encoder_opt.load_state_dict(checkpoint['encoder_opt'])
            for sem_funcs_ix in range(len(self.sem_funcs_opt)):
                self.sem_funcs_opt[sem_funcs_ix].load_state_dict(checkpoint['sem_funcs_opt'][sem_funcs_ix])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
        