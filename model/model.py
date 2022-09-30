from filecmp import dircmp
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from collections import defaultdict
from pprint import pprint


# x = F.dropout(x, training=self.training)
# return F.log_softmax(x, dim=1)
    
class BSGEncoder(BaseModel):
    
    def __init__(self, hidden_act_func, input_dim, hidden_dim, mu_dim, sigma_dim, num_embs):
        super().__init__()
        self.input_dim = input_dim
        self.embeds = nn.Embedding(num_embs, input_dim)
        self.fc_pair = nn.Linear(input_dim * 2, hidden_dim, bias = False)
        self.act_hidden = getattr(nn, hidden_act_func)()
        self.fc_mu = nn.Linear(hidden_dim, mu_dim)
        self.fc_sigma = nn.Linear(hidden_dim, sigma_dim)
    
    def forward(self, targ_preds_ix_batch, lex_preds_ix_batch):

        lex_embeds = self.embeds(lex_preds_ix_batch)
        targ_embeds = self.embeds(targ_preds_ix_batch)

        targ_embeds_exp = targ_embeds.expand(
            targ_embeds.size(dim=0),
            targ_embeds.size(dim=1),
            lex_embeds.size(dim=2),
            self.input_dim)
        lex_embeds_exp = lex_embeds.expand(
            lex_embeds.size(dim=0),
            targ_embeds.size(dim=1),
            lex_embeds.size(dim=2),
            self.input_dim)

        paired = torch.cat((lex_embeds_exp, targ_embeds_exp), dim=3)
        z = self.fc_pair(paired)
        a = self.act_hidden(z)
        sum_a = torch.sum(a, dim = 2)
        mu_batch = self.fc_mu(sum_a)
        log_sigma_batch = self.fc_sigma(sum_a)

        return mu_batch, log_sigma_batch
    
class OnePlaceSemFunc(BaseModel):
    
    def __init__(self, input_dim, hidden_dim, hidden_act_func, output_act_func):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.act1 = getattr(nn, hidden_act_func)()
        self.act2 = getattr(nn, output_act_func)()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x
    
class TwoPlaceSemFunc(BaseModel):
    
    def __init__(self, input_dim, hidden_dim, hidden_act_func, output_act_func):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.act1 = getattr(nn, hidden_act_func)()
        self.act2 = getattr(nn, output_act_func)()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x

class FLDecoder(BaseModel):
    
    def __init__(self, fuzzy_logic, num_negative_samples, sem_funcs):
        
        super().__init__()

        self.op2func = {
            "aANDb": lambda args: self.fl_and(*args),
            "aORb": lambda args: self.fl_or(*args),
            "!a": lambda args: self.fl_not(*args)
        }
        f = self.op2func
        self.op2func.update({
            "!a=>b": lambda args: f["aORb"](*args),
            "aAND!b": lambda args: f["aANDb"](args[0], f["!a"](args[1])),
            "a=>b": lambda args: f["aORb"](f["!a"](args[0]), args[1]),
            "a<=>b": lambda args: f["aORb"](f["aANDb"](*args), f["!a"](f["aORb"](*args))),
            "a": lambda args: args
        })
        pprint (self.op2func)
        # self.one_place_decoder = OnePlaceSemFunc
        # self.two_place_decoder = TwoPlaceSemFunc
        self.fuzzy_logic = fuzzy_logic
        self.fl_not = self.fl_not
        if fuzzy_logic == "product":
            self.fl_and = self.product_and
            self.fl_or = self.product_or
            self.log_or_id = torch.nn.Identity
            self.add_or_mul = torch.mul
        elif fuzzy_logic == "max-product":
            self.fl_and = self.product_and
            self.fl_or = self.max_or
            self.log_or_id = torch.log
            self.add_or_mul = torch.add
        self.num_negative_samples = num_negative_samples
        self.sem_funcs = sem_funcs

    @staticmethod
    def get_op(op_str):
        op = op_str.split("-")[1]
        return op

    def product_and(self, a, b):
        return self.add_or_mul(a, b)
    def product_or(self, a, b):
        return a + b - a * b
    def max_or(self, a, b):
        return torch.max(a, b)
    def fl_not(self, a):
        return 1 - a
    
    def decode_batch(self, decoders_data_batch, encoder_data_batch, sample_zs):
        # currently support batch_size = 1
        # sample z ~ q_phi(dmrs_i) (reparametrization)
        return [
            self.decode(data["logic_expr"], encoder_data_batch[inst_idx], sample_zs[inst_idx])
            for inst_idx, data in enumerate(decoders_data_batch)
        ]

    def decode(self, logic_expr, encoder_data, sample_z):
        fuzzy_truthness = None

        node2z = dict(zip(encoder_data["pred_func_nodes"], sample_z))
        if isinstance(logic_expr, dict):
            pred_func_name, args = logic_expr['pred_func_name'], logic_expr['args']
            concat_z = torch.cat([node2z[arg] for arg in args])
            fuzzy_truthness = self.log_or_id(self.sem_funcs[pred_func_name](concat_z))
            
        elif logic_expr:
            root, *dgtrs = logic_expr
            op = self.get_op(root)
            decoded_dgtrs = [self.decode(dgtr, encoder_data, sample_z) for dgtr in dgtrs]
            fuzzy_truthness = self.op2func[op](decoded_dgtrs)
        else:
            fuzzy_truthness = 1.0
            print ("err:", logic_expr, " :logic_expr")
        # print (logic_expr, "\t", fuzzy_truthness)
        return fuzzy_truthness

    def get_negative_sample(self):
        self.num_negative_samples


    # def _decode(logic_expr)
        