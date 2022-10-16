from filecmp import dircmp
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from collections import defaultdict
from pprint import pprint

import numpy as np

from functools import reduce


# x = F.dropout(x, training=self.training)
# return F.log_softmax(x, dim=1)
class VarAutoencoder(BaseModel):

    def __init__(self, encoder, decoder, start_beta, end_beta, device, ddp):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.device = device
        if ddp:
            self.mu_dim = self.encoder.module.mu_dim
        else:
            self.mu_dim = self.encoder.mu_dim
        self.normal_dist = torch.distributions.Normal(torch.zeros(self.mu_dim, device = self.device), torch.tensor(1.0, device = self.device))

    def sample_from_gauss(self, mu_batch, sigma2_batch, num_samples = 1):
        batch_size, max_num_nodes, mu_dim = mu_batch.size()
        num_nodes_batch = max_num_nodes # num_targ_preds_batch
        # print ("normal_dist", normal_dist.get_device())
        # currently supprot batch_size = 1 only, i.e. max_num_nodes = num_nodes
        if num_samples == 1:
            sample_eps = self.normal_dist.sample(torch.Size([batch_size, num_nodes_batch]))
        else:
            sample_eps = self.normal_dist.sample(torch.Size([batch_size, num_nodes_batch, num_samples]))
            mu_batch = mu_batch.unsqueeze(dim = 2)
            sigma2_batch = sigma2_batch.unsqueeze(dim = 2)
        # print ("sample_eps", sample_eps.get_device())
        # print ("b4:", sigma2_batch.shape)
        # print ("b4:", mu_batch.shape)
        # # if num_samples > 1
        # print ("after:", sigma2_batch.shape)
        # print ("after:", mu_batch.shape)
        # print ("ep:", sample_eps.shape)
        sample_zs = mu_batch + torch.sqrt(sigma2_batch) * sample_eps
        # print ("sample_zs:", sample_zs.shape)
        # return torch.ones(sample_zs.shape)
        return sample_zs

    def run2(self, **kwargs):
        encoder_data = kwargs["encoder"]
        decoder_data = kwargs["decoder"]

        mu_batch, log_sigma2_batch = self.encoder(*encoder_data)
        # print ("log_sigma2_batch", log_sigma2_batch)
        sigma2_batch = torch.exp(log_sigma2_batch)
        print ("log_sigma2_batch", log_sigma2_batch)
        print ("sigma2_batch", sigma2_batch)

        sample_zs = self.sample_from_gauss(mu_batch, sigma2_batch, num_samples = 1)#.squeeze(dim = 2)

        batch_log_truth = self.decoder.decode_batch(sample_zs, *decoder_data, device = self.device)#.to(self.device)
         # print ("sample_zs:", sample_zs.shape)
        # batch_log_truth = self.decoder.decode_batch(sample_zs, *decoder_data, device = self.device)#.to(self.device)
        # print (batch_log_truth)
        # print ("batch_log_truth_run", batch_log_truth.get_device())
        # print ("fuzzy_truthness_samples:", fuzzy_truthness_samples.shape)
        # fuzzy_truthness_samples = fuzzy_truthness_samples.squeeze(dim = 2)
        # # if num_samples > 1:
        # batch_log_truth_avg = torch.mean(batch_log_truth), dim = #)
            
        # standard normal prior
        batch_size, max_num_nodes, mu_dim = mu_batch.size()
        num_nodes_batch = max_num_nodes # num_targ_preds_batch
        kl_div = (1/2) * (torch.sum(sigma2_batch, dim = (1,2)) * mu_dim + torch.sum(torch.square(mu_batch), dim = (1,2)) - num_nodes_batch * mu_dim - torch.sum(torch.log(sigma2_batch), dim = (1,2)) * mu_dim)
        kl_div = kl_div.squeeze()
        # if torch.isinf(kl_div):
        #     print ("KK")
        #     print (num_nodes_batch)
            # print (torch.sum(sigma2_batch, dim = (1,2)) * mu_dim)
            # print (torch.sum(torch.square(mu_batch), dim = (1,2)))
            # print (torch.sum(torch.log(sigma2_batch), dim = (1,2)) * mu_dim)
        #     print (sigma2_batch)
        # if torch.isinf(batch_log_truth):
        #     print ("LL")
        #     print (decoder_data)
        # print ("kl_div", kl_div.get_device())
        # print ("sample_zs", sample_zs.shape)
        # print ("mu_batch", mu_batch.shape)
        # print ("sigma2_batch", sigma2_batch.shape)
        # print ("num_nodes_batch", num_nodes_batch.shape)

        return batch_log_truth , kl_div


    def run(self, **kwargs):

        encoder_data = kwargs["encoder"]
        decoder_data = kwargs["decoder"]

        mu_batch, log_sigma2_batch = self.encoder(*encoder_data)
        # print ("log_sigma2_batch", log_sigma2_batch)
        sigma2_batch = torch.exp(log_sigma2_batch)
        # print ("log_sigma2_batch", log_sigma2_batch)
        # print ("sigma2_batch", sigma2_batch)


        sample_zs = self.sample_from_gauss(mu_batch, sigma2_batch, num_samples = 1)#.squeeze(dim = 2)
        # print ("sample_zs:", sample_zs.shape)
        batch_log_truth = self.decoder.decode_batch(sample_zs, *decoder_data, device = self.device)#.to(self.device)
        # print (batch_log_truth)
        # print ("batch_log_truth_run", batch_log_truth.get_device())
        # print ("fuzzy_truthness_samples:", fuzzy_truthness_samples.shape)
        # fuzzy_truthness_samples = fuzzy_truthness_samples.squeeze(dim = 2)
        # # if num_samples > 1:
        # batch_log_truth_avg = torch.mean(batch_log_truth), dim = #)
            
        # standard normal prior
        batch_size, max_num_nodes, mu_dim = mu_batch.size()
        num_nodes_batch = max_num_nodes # num_targ_preds_batch
        kl_div = (1/2) * (torch.sum(sigma2_batch, dim = (1,2)) * mu_dim + torch.sum(torch.square(mu_batch), dim = (1,2)) - num_nodes_batch * mu_dim - torch.sum(torch.log(sigma2_batch), dim = (1,2)) * mu_dim)
        kl_div = kl_div.squeeze()
        # if torch.isinf(kl_div):
        #     print ("KK")
        #     print (num_nodes_batch)
            # print (torch.sum(sigma2_batch, dim = (1,2)) * mu_dim)
            # print (torch.sum(torch.square(mu_batch), dim = (1,2)))
            # print (torch.sum(torch.log(sigma2_batch), dim = (1,2)) * mu_dim)
        #     print (sigma2_batch)
        # if torch.isinf(batch_log_truth):
        #     print ("LL")
        #     print (decoder_data)
        # print ("kl_div", kl_div.get_device())
        # print ("sample_zs", sample_zs.shape)
        # print ("mu_batch", mu_batch.shape)
        # print ("sigma2_batch", sigma2_batch.shape)
        # print ("num_nodes_batch", num_nodes_batch.shape)
        return batch_log_truth , kl_div

class BSGEncoder(BaseModel):
    
    def __init__(self, hidden_act_func, input_dim, hidden_dim, mu_dim, sigma_dim, num_embs):
        super().__init__()
        self.input_dim = input_dim
        self.mu_dim = mu_dim
        self.embeds = nn.Embedding(num_embs + 1, input_dim, padding_idx = 0).requires_grad_(True)
        self.fc_pair = nn.Linear(input_dim * 2, hidden_dim, bias = False)
        if hidden_act_func == 'ReLU':
            nn.init.kaiming_normal_(self.fc_pair.weight, mode='fan_out', nonlinearity='relu')
        elif hidden_act_func == 'LeakyReLU':
            nn.init.kaiming_normal_(self.fc_pair.weight, a = 0.01, mode='fan_out', nonlinearity='leaky_relu')
        else:
            nn.init.xavier_uniform_(self.fc_pair.weight)
        self.act_hidden = getattr(nn, hidden_act_func)()
        self.fc_mu = nn.Linear(hidden_dim, mu_dim)
        nn.init.xavier_uniform_(self.fc_mu.weight)
        self.fc_sigma = nn.Linear(hidden_dim, sigma_dim)
        nn.init.xavier_uniform_(self.fc_sigma.weight)
    
    def forward(self, targ_preds_ix_batch, content_preds_ix_batch, num_targ_preds_batch, num_content_preds_batch, max_num_targ_preds_batch):
        # print (targ_preds_ix_batch)
        # targ_preds_ix_batch = torch.tensor(targ_preds_ix_batch)
        # content_preds_ix_batch = torch.tensor(content_preds_ix_batch)
        # print ("targ_preds_ix_batch:", targ_preds_ix_batch.shape)
        # print ("content_preds_ix_batch:", content_preds_ix_batch.shape)
        # print ("num_targ_preds_batch:", num_targ_preds_batch.shape)
        content_embeds = self.embeds(content_preds_ix_batch + 1) # + 1 since 0 is padding embedding for batch processing
        targ_embeds = self.embeds(targ_preds_ix_batch + 1)
        # print ("Within targ:", targ_embeds.shape)
        # print ("Within content:", content_embeds.shape)

        targ_embeds_exp = targ_embeds.expand(
            targ_embeds.size(dim=0),
            targ_embeds.size(dim=1),
            content_embeds.size(dim=2),
            self.input_dim
        )
        content_embeds_exp = content_embeds.expand(
            content_embeds.size(dim=0),
            targ_embeds.size(dim=1),
            content_embeds.size(dim=2),
            self.input_dim
        )
        # print ("content_embeds_exp", content_embeds_exp.shape)
        # print ("targ_embeds_exp", targ_embeds_exp.shape)

        paired = torch.cat((content_embeds_exp, targ_embeds_exp), dim=3)
        paired_masked = paired[:, :num_targ_preds_batch, :num_content_preds_batch,:]
        # print ("paired_masked:", paired_masked.shape)
        paired_lin = self.fc_pair(paired_masked)
        a = self.act_hidden(paired_lin)
        sum_a = torch.mean(a, dim = 2)
        mu_batch = self.fc_mu(sum_a)
        mu_pad = (0, 0, 0, (max_num_targ_preds_batch - mu_batch.size(dim = 1)))
        mu_batch_padded = F.pad(mu_batch, mu_pad, "constant", 0)
        log_sigma_batch = self.fc_sigma(sum_a)
        log_sigma_pad = (0, 0, 0, (max_num_targ_preds_batch - mu_batch.size(dim = 1)))
        log_sigma_batch_padded = F.pad(log_sigma_batch, log_sigma_pad, "constant", 0)
        # print ("paired_masked", paired_masked)
        # print ("paired_lin", paired_lin)
        # print ("sum_a", sum_a)
        # print ("self.fc_pair.weight", self.fc_pair.weight)
        # print ("self.fc_sigma.weight", self.fc_sigma.weight)

        return mu_batch_padded, log_sigma_batch_padded
    
class OnePlaceSemFunc(BaseModel):

    def __init__(self, input_dim, hidden_dim, hidden_act_func, output_act_func):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if hidden_act_func == 'ReLU':
            # fc1_normal_dist = torch.distributions.Normal(torch.zeros(hidden_dim), torch.sqrt(2/hidden_dim))
            # fc1_weights = fc2_normal_dist.sample(torch.Size([1, hidden_act_func]))
            # self.fc1 = torch.nn.Parameter(fc2_weights)
            nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        elif hidden_act_func == 'LeakyReLU':
            nn.init.kaiming_normal_(self.fc1.weight, a = 0.01, mode='fan_out', nonlinearity='leaky_relu')
        else:
            pass
        self.act1 = getattr(nn, hidden_act_func)()
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.act2 = getattr(nn, output_act_func)()
        self.log_act2 = getattr(nn, "LogSigmoid")()
        
    def forward(self, x, take_log = False):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        if take_log:
            x = self.log_act2(x)
        else:
            x = self.act2(x)

        return x
    
class TwoPlaceSemFunc(BaseModel):
    
    def __init__(self, input_dim, hidden_dim, hidden_act_func, output_act_func):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if hidden_act_func == 'ReLU':
            # fc1_normal_dist = torch.distributions.Normal(torch.zeros(hidden_dim), torch.sqrt(2/hidden_dim))
            # fc1_weights = fc2_normal_dist.sample(torch.Size([1, hidden_act_func]))
            # self.fc1 = torch.nn.Parameter(fc2_weights)
            nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        elif hidden_act_func == 'LeakyReLU':
            nn.init.kaiming_normal_(self.fc1.weight, a = 0.01, mode='fan_out', nonlinearity='leaky_relu')
        else:
            pass
        self.act1 = getattr(nn, hidden_act_func)()
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.act2 = getattr(nn, output_act_func)()
        self.log_act2 = getattr(nn, "LogSigmoid")()
        
    def forward(self, x, take_log = False):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        if take_log:
            x = self.log_act2(x)
        else:
            x = self.act2(x)
        return x


# DDP hangs when using this code

# class FLDecoder(BaseModel):
    
#     # def a_AND_b(self, args): return self.fl_and(*args)
#     # def a_OR_b(self, args): return self.fl_or(*args)
#     # def NOT_a(self, args): return self.fl_not(*args)

#     # def a_AND_NOT_b(self, args): return self.op2f[0]([args[0], self.op2f[2](args[1])])
#     # def a_IMPLIES_b(self, args): return self.op2f[1]([self.op2f[2](args[0]), args[1]])
#     # def a_EQ_b(self, args): return self.op2f[1]([self.op2f[0](args), self.op2f[2](self.op2f[1](args))])
#     # def a(self, args): return args

#     def a_AND_b(self, args): return self.fl_and(*args) # args[:,0], args[:,1])
#     def a_OR_b(self, args): return self.fl_or(*args) #args[:,0], args[:,1])
#     def NOT_a(self, args): return self.fl_not(*args)

#     def a_AND_NOT_b(self, args): return self.a_AND_b([args[0], self.NOT_a([args[1]])])
#     def a_IMPLIES_b(self, args): return self.a_OR_b([self.NOT_a([args[0]]), args[1]])
#     def a_EQ_b(self, args): return self.a_OR_b([self.a_AND_b(args), self.NOT_a([self.a_OR_b(args)])])
#     def a(self, args): return args

#     def __init__(self, t_norm, s_norm, take_log, freq_sampling, num_negative_samples, sem_funcs, pred_func2cnt, pred_funcs):
        
#         super().__init__()

#         # self.one_place_decoder = OnePlaceSemFunc
#         # self.two_place_decoder = TwoPlaceSemFunc
#         self.fl_not = self.neg
#         self.take_log = take_log
#         if take_log:
#             self.add_or_mul = torch.add
#             self.log_or_id = torch.log
#         else:
#             self.add_or_mul = torch.mul
#             self.log_or_id = torch.nn.Identity()

#         if t_norm == "product":
#             self.fl_and = self.product_and
#         elif t_norm == "min":
#             self.fl_and = self.min_and

#         if s_norm == "product":
#             self.fl_or = self.product_or
#         elif s_norm == "max":
#             self.fl_or = self.max_or

#         self.op2f = [
#             self.a_AND_b,
#             self.a_OR_b,
#             self.NOT_a
#         ]
#         self.op2f += [
#             self.a_OR_b,
#             self.a_AND_NOT_b,
#             self.a_IMPLIES_b,
#             self.a_EQ_b,
#             self.a
#         ]

#         self.freq_sampling = freq_sampling
#         self.num_negative_samples = num_negative_samples
#         self.sem_funcs = sem_funcs
#         self.pred_funcs = pred_funcs

#         # pred_func_names = list(pred_func2cnt.keys())
#         one_place_pred_func2cnt = {pred_func_ix: cnt for pred_func_ix, cnt in pred_func2cnt.items() if self.pred_funcs[pred_func_ix].endswith("@ARG0")}
#         self.one_place_pred_func_names = list(one_place_pred_func2cnt.keys())
#         two_place_pred_func2cnt = {pred_func_ix: cnt for pred_func_ix, cnt in pred_func2cnt.items() if not self.pred_funcs[pred_func_ix].endswith("@ARG0")}
#         self.two_place_pred_func_names = list(two_place_pred_func2cnt.keys())
#         self.pred_func_probs = None
#         if self.freq_sampling:
#             one_place_freq_sum = sum(one_place_pred_func2cnt.values())
#             two_place_freq_sum = sum(two_place_pred_func2cnt.values())
#             self.one_place_pred_func_probs = [pred_func2cnt[pred_func_name]/one_place_freq_sum for pred_func_name in self.one_place_pred_func_names]
#             self.two_place_pred_func_probs = [pred_func2cnt[pred_func_name]/two_place_freq_sum for pred_func_name in self.two_place_pred_func_names]
#         else:
#             self.one_place_pred_func_probs = None
#             self.two_place_pred_func_probs = None
#         self.num_neg_sampled = 999999
#         self.regen_one_place_negative_samples()
#         self.regen_two_place_negative_samples()

#     def regen_one_place_negative_samples(self):
#         self.neg_samp_one_place_pred_funcs = np.random.choice(self.one_place_pred_func_names, self.num_neg_sampled, replace = True, p = self.one_place_pred_func_probs)
#         self.neg_samp_one_place_idx = 0
#     def regen_two_place_negative_samples(self):
#         self.neg_samp_two_place_pred_funcs = np.random.choice(self.two_place_pred_func_names, self.num_neg_sampled, replace = True, p = self.two_place_pred_func_probs)
#         self.neg_samp_two_place_idx = 0

#     @staticmethod
#     def get_op(op_str):
#         op = op_str.split("-")[1]
#         return op

#     def product_and(self, a, b):
#         # print (a, b)
#         return self.add_or_mul(a, b)
            
#     def product_or(self, a, b):
#         return a + b - a * b
#     def min_and(self, a, b):
#         return torch.minimum(a, b)
#     def max_or(self, a, b):
#         return torch.maximum(a, b)
#     def neg(self, a):
#         return 1 - a

#     # def forward(self, sample_zs, logic_expr, pred_func_nodes):
#     #     # currently support batch_size = 1
#     #     return [
#     #         self.decode(sample_zs[inst_idx], logic_expr[inst_idx], pred_func_nodes[inst_idx])
#     #         for inst_idx in range(len(sample_zs))
#     #     ]

#     def decode_batch(self, sample_zs, logic_expr, pred_func_nodes, device):
#         batch_truth = [
#             self.decode(sample_zs[inst_idx], logic_expr[inst_idx], pred_func_nodes[inst_idx], device).squeeze()
#             for inst_idx in range(len(sample_zs))
#         ]
#         # currently support batch_size = 1
#         batch_truth = batch_truth[0]
#         # print ("batch_truth", batch_truth)
#         return batch_truth
#         # return torch.stack([
#         #     torch.sum(sample_zs[inst_idx]).unsqueeze(dim = 0).unsqueeze(dim = 0) for inst_idx in range(len(sample_zs))
#         # ])

#     def _neg_sem_funcs(self, sem_func_name, concat_z):
#         return self.fl_not(self.sem_funcs[sem_func_name](concat_z))

#     def decode(self, sample_z, logic_expr, pred_func_nodes, device):
#         truth = None
#         # print (pred_func_nodes, sample_z)
#         node2z = dict(zip(pred_func_nodes, sample_z))
#         # print (node2z)
#         ## if dict is used:
#         # if isinstance(logic_expr, dict):
#         if len(logic_expr) == 2 and all([isinstance(logic_expr[1][i], int) for i in range(len(logic_expr[1]))]):
#             ## if dict is used:
#             # sem_func_name, args = logic_expr['pf'], logic_expr['args']
#             sem_func_name, args = logic_expr
#             # try:
#             concat_z = torch.cat([node2z[arg] for arg in args], dim = 1)
#             print ("concat_z", concat_z.shape)
#             # except Exception as e:
#             #     print (e)
#             #     print (encoder_data)
#             #     print (pred_func_name, args)
#             #     input()
#             pos_sample_truth = self.log_or_id(self.sem_funcs[sem_func_name](concat_z))
#             # negative samples
#             neg_sem_func_names = self.get_negative_samples(self.pred_funcs[sem_func_name])
#             neg_samples_truths =  torch.stack([self._neg_sem_funcs(sem_func_name, concat_z) for sem_func_name in neg_sem_func_names])
#             print ("neg_samples_truths:", neg_samples_truths.shape)
#             # neg_samples_truths =  self.fl_not(self.sem_funcs[x](concat_z)), neg_pred_func_names
#             # print ("negative samples:\t", neg_pred_func_names, neg_samples_truths)
#             # neg_samples_truths_T = torch.transpose(neg_samples_truths, 0, 1)
#             if self.fl_and == self.product_and:
#             # neg_samples_truth =  self.fl_and(neg_samples_truths)
#                 neg_samples_truth =  torch.prod(neg_samples_truths, dim = 0)
#             elif self.fl_and == self.min_and:
#                 neg_samples_truth =  torch.min(neg_samples_truths, dim = 0)
#             print ("neg_samples_truth", neg_samples_truth.shape)
#             print ("pos_sample_truth", pos_sample_truth.shape)
#             truth = self.fl_and(pos_sample_truth, neg_samples_truth)
#             print ("pf truth", truth.shape)
#         elif logic_expr:
#             root, *dgtrs = logic_expr
#             # op = self.get_op(root)
#             op = root
#             # try:
#             decoded_dgtrs = [self.decode(sample_z, dgtr, pred_func_nodes, device) for dgtr in dgtrs]
#             # if op == 2:
#             #     print (op, decoded_dgtrs)
#             # print (decoded_dgtrs.shape)
#             truth = self.op2f[op](decoded_dgtrs)
#             # except:
#             # print (op)
#             #     input()
#             if len(decoded_dgtrs) == 2:
#                 print (op, decoded_dgtrs[0].shape, decoded_dgtrs[1].shape, truth.shape)
#             elif len(decoded_dgtrs) == 1:
#                 print (op, decoded_dgtrs[0].shape, truth.shape)
#         else:
#             truth = 1.0
#             print ("err:", logic_expr, " :logic_expr")
        
#         # print (logic_expr, "\t", truth)
#         # try:
#         #     if truth.shape == torch.Size([1,8,1]):
#         #         pass
#         # except:
#         #     print (op, truth, decoded_dgtrs)
#         return truth

#     def get_negative_samples(self, pos_pred_func_name):
#         if pos_pred_func_name.endswith("@ARG0"):
#             self.neg_samp_one_place_idx = self.neg_samp_one_place_idx + self.num_negative_samples
#             if self.neg_samp_one_place_idx >= self.num_neg_sampled:
#                 self.regen_one_place_negative_samples()
#             return self.neg_samp_one_place_pred_funcs[self.neg_samp_one_place_idx - self.num_negative_samples : self.neg_samp_one_place_idx]
#         else:
#             self.neg_samp_two_place_idx = self.neg_samp_two_place_idx + self.num_negative_samples
#             if self.neg_samp_two_place_idx >= self.num_neg_sampled:
#                 self.regen_two_place_negative_samples()
#             return self.neg_samp_two_place_pred_funcs[self.neg_samp_two_place_idx - self.num_negative_samples : self.neg_samp_two_place_idx]

#         return sampled_pred_funcs


class FLDecoder(BaseModel):
    
    # def a_AND_b(self, args): return self.fl_and(*args)
    # def a_OR_b(self, args): return self.fl_or(*args)
    # def NOT_a(self, args): return self.fl_not(*args)

    # def a_AND_NOT_b(self, args): return self.op2f[0]([args[0], self.op2f[2](args[1])])
    # def a_IMPLIES_b(self, args): return self.op2f[1]([self.op2f[2](args[0]), args[1]])
    # def a_EQ_b(self, args): return self.op2f[1]([self.op2f[0](args), self.op2f[2](self.op2f[1](args))])
    # def a(self, args): return args

    def a_AND_b(self, args): return self.fl_and(*args) # args[:,0], args[:,1])
    def a_OR_b(self, args): return self.fl_or(*args) #args[:,0], args[:,1])
    def NOT_a(self, args): return self.fl_not(*args)

    def a_AND_NOT_b(self, args): return self.a_AND_b([args[0], self.NOT_a(args[1])])
    def a_IMPLIES_b(self, args): return self.a_OR_b([self.NOT_a(args[0]), args[1]])
    def a_EQ_b(self, args): return self.a_OR_b([self.a_AND_b(args), self.NOT_a(self.a_OR_b(args))])
    def a(self, args): return args

    def __init__(self, t_norm, s_norm, take_log, freq_sampling, num_negative_samples, neg_cost, sem_funcs, pred_func2cnt, pred_funcs):
        
        super().__init__()

        # self.one_place_decoder = OnePlaceSemFunc
        # self.two_place_decoder = TwoPlaceSemFunc
        self.fl_not = self.neg
        self.take_log = take_log
        self.neg_cost = neg_cost
        if take_log:
            self.add_or_mul = torch.add
            self.log_or_id = torch.log
        else:
            self.add_or_mul = torch.mul
            self.log_or_id = torch.nn.Identity()

        if t_norm == "product":
            self.fl_and = self.product_and
        elif t_norm == "min":
            self.fl_and = self.min_and

        if s_norm == "product":
            self.fl_or = self.product_or
        elif s_norm == "max":
            self.fl_or = self.max_or

        self.op2f = [
            self.a_AND_b,
            self.a_OR_b,
            self.NOT_a
        ]
        self.op2f += [
            self.a_OR_b,
            self.a_AND_NOT_b,
            self.a_IMPLIES_b,
            self.a_EQ_b,
            self.a
        ]

        self.freq_sampling = freq_sampling
        self.num_negative_samples = num_negative_samples
        self.sem_funcs = sem_funcs
        self.pred_funcs = pred_funcs

        # pred_func_names = list(pred_func2cnt.keys())
        one_place_pred_func2cnt = {pred_func_ix: cnt for pred_func_ix, cnt in pred_func2cnt.items() if self.pred_funcs[pred_func_ix].endswith("@ARG0")}
        self.one_place_pred_func_names = list(one_place_pred_func2cnt.keys())
        two_place_pred_func2cnt = {pred_func_ix: cnt for pred_func_ix, cnt in pred_func2cnt.items() if not self.pred_funcs[pred_func_ix].endswith("@ARG0")}
        self.two_place_pred_func_names = list(two_place_pred_func2cnt.keys())
        self.pred_func_probs = None
        if self.freq_sampling:
            one_place_freq_sum = sum(one_place_pred_func2cnt.values())
            two_place_freq_sum = sum(two_place_pred_func2cnt.values())
            self.one_place_pred_func_probs = [pred_func2cnt[pred_func_name]/one_place_freq_sum for pred_func_name in self.one_place_pred_func_names]
            self.two_place_pred_func_probs = [pred_func2cnt[pred_func_name]/two_place_freq_sum for pred_func_name in self.two_place_pred_func_names]
        else:
            self.one_place_pred_func_probs = None
            self.two_place_pred_func_probs = None
        self.num_neg_sampled = 99
        self.regen_one_place_negative_samples()
        self.regen_two_place_negative_samples()

    def regen_one_place_negative_samples(self):
        self.neg_samp_one_place_pred_funcs = np.random.choice(self.one_place_pred_func_names, self.num_neg_sampled, replace = True, p = self.one_place_pred_func_probs)
        self.neg_samp_one_place_idx = 0
    def regen_two_place_negative_samples(self):
        self.neg_samp_two_place_pred_funcs = np.random.choice(self.two_place_pred_func_names, self.num_neg_sampled, replace = True, p = self.two_place_pred_func_probs)
        self.neg_samp_two_place_idx = 0

    @staticmethod
    def get_op(op_str):
        op = op_str.split("-")[1]
        return op

    def product_and(self, a, b):
        return self.add_or_mul(a, b)
            
    def product_or(self, a, b):
        return a + b - a * b
    def min_and(self, a, b):
        return torch.minimum(a, b)
    def max_or(self, a, b):
        return torch.maximum(a, b)
    def neg(self, a):
        return 1 - a

    # def forward(self, sample_zs, logic_expr, pred_func_nodes):
    #     # currently support batch_size = 1
    #     return [
    #         self.decode(sample_zs[inst_idx], logic_expr[inst_idx], pred_func_nodes[inst_idx])
    #         for inst_idx in range(len(sample_zs))
    #     ]
    def decode_batch2(self, sample_zs, logic_expr, pred_func_nodes, device):
        # currently support batch_size = 1

        batch_log_truth = torch.tensor([
            torch.sum(sample_zs)
            for inst_idx in range(sample_zs.size(0))
        ], device = device)
        # print ("batch_log_truth_i", batch_log_truth.get_device())
        return batch_log_truth

    def decode_batch(self, sample_zs, logic_expr, pred_func_nodes, device):
        # currently support batch_size = 1

        batch_log_truth = torch.tensor([
            self.decode(sample_zs[inst_idx], logic_expr[inst_idx], pred_func_nodes[inst_idx], device, agg = True)[0]
            for inst_idx in range(sample_zs.size(0))
        ], device = device)
        # print ("batch_log_truth_i", batch_log_truth.get_device())
        return batch_log_truth
        # return torch.stack([
        #     torch.sum(sample_zs[inst_idx]).unsqueeze(dim = 0).unsqueeze(dim = 0) for inst_idx in range(len(sample_zs))
        # ])

    def _neg_sem_funcs(self, sem_func_name, concat_z):
        truth = self.sem_funcs[sem_func_name](concat_z)

        neg_truth = self.fl_not(truth)
        # log_neg_truth = torch.log(neg_truth)
        # if torch.isinf(log_neg_truth):
        #     log_neg_truth = torch.clamp(log_neg_truth, min = -100)
            # print (log_neg_truth, concat_z, self.sem_funcs[sem_func_name].module.fc1, self.sem_funcs[sem_func_name].module.fc2)
        return neg_truth

    def get_neg_samps_truths(self, sem_func_name, concat_z):
        neg_sem_func_names = self.get_negative_samples(self.pred_funcs[sem_func_name])
        neg_samp_truths =  torch.tensor([torch.clamp(torch.log(self._neg_sem_funcs(sem_func_name, concat_z)), min = -100) for sem_func_name in neg_sem_func_names])
        # print ("neg_samp_truths)
        # print ("neg_samp_truths.get_device()", torch.log(self._neg_sem_funcs(neg_sem_func_names[0], concat_z)).get_device())
        return neg_samp_truths

    def decode(self, sample_z, logic_expr, pred_func_nodes, device, agg = False):
        pos_truth = None
        # print (pred_func_nodes, sample_z)
        node2z = dict(zip(pred_func_nodes, sample_z))
        ## if dict is used:
        # if isinstance(logic_expr, dict):
        if len(logic_expr) == 2 and all([isinstance(logic_expr[1][i], int) for i in range(len(logic_expr[1]))]):
            ## if dict is used:
            # sem_func_name, args = logic_expr['pf'], logic_expr['args']
            sem_func_name, args = logic_expr
            # if num_samples > 1 
            # concat_z = torch.cat([node2z[arg] for arg in args], dim = 1)
            concat_z = torch.cat([node2z[arg] for arg in args], dim = 0)
            # print ("concat_z.get_device()", concat_z.get_device())
            pos_truth = self.log_or_id(self.sem_funcs[sem_func_name](concat_z))
            # print ("pos_truth1.get_device()", pos_truth.get_device())
            neg_samp_truths = self.get_neg_samps_truths(sem_func_name, concat_z)
            # print ("neg_samp_truths.get_device()", neg_samp_truths.get_device())
            # print ("neg_samp_truths", neg_samp_truths.shape)
            # negative samples

        elif logic_expr:
            root, *dgtrs = logic_expr
            # op = self.get_op(root)
            op = root
            decoded_dgtrs = list(zip(*[self.decode(sample_z, dgtr, pred_func_nodes, device) for dgtr in dgtrs]))
            pos_truth = self.op2f[op](decoded_dgtrs[0])
            # print ("pos_truth2.get_device()", pos_truth.get_device())
            neg_samp_truths = torch.cat(decoded_dgtrs[1])
            # print ("neg_samp_truths2.get_device()", neg_samp_truths.get_device())
        else:
            pos_truth = 1.0
            print ("err:", logic_expr, " :logic_expr")
            input ()
        # print ("pos_truth", pos_truth.shape)
        if agg:
            log_truth = torch.sum(neg_samp_truths) + torch.clamp(torch.log(pos_truth), min = -100)
            # print ("pos_truth3.get_device()", pos_truth.get_device())
            # print ("neg_samp_truths3.get_device()", neg_samp_truths.get_device())
            # print ("log_truth.get_device()", log_truth.get_device())
            # print ("neg_samp_truths", neg_samp_truths.shape)
            # print ("truth", truth.shape)
            return log_truth, None
        else:
            return pos_truth, neg_samp_truths

    def get_negative_samples(self, pos_pred_func_name):
        if pos_pred_func_name.endswith("@ARG0"):
            if self.neg_samp_one_place_idx + self.num_negative_samples >= self.num_neg_sampled:
                self.regen_one_place_negative_samples()
            sampled_pred_funcs = self.neg_samp_one_place_pred_funcs[self.neg_samp_one_place_idx : self.neg_samp_one_place_idx + self.num_negative_samples]
            self.neg_samp_one_place_idx += self.num_negative_samples 
        else:
            if self.neg_samp_two_place_idx + self.num_negative_samples >= self.num_neg_sampled:
                self.regen_two_place_negative_samples()
            sampled_pred_funcs = self.neg_samp_two_place_pred_funcs[self.neg_samp_two_place_idx : self.neg_samp_two_place_idx + self.num_negative_samples]
            self.neg_samp_two_place_idx += self.num_negative_samples
        return sampled_pred_funcs