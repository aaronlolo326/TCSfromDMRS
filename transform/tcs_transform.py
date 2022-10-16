from os import preadv
from readline import set_auto_history
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

from collections import defaultdict, Counter
from pprint import pprint
from varname import nameof
import time
import os

from disjoint_set import DisjointSet
from toposort import toposort, toposort_flatten

from src import util, dg_util
from utils import get_transformed_info, draw_logic_expr

from functools import reduce

import copy

TRUE = "<True>"

EVENTUALITIES = 'e'
INSTANCES = 'x'

LOGIC_PRED_TYPE_C = 'C'
LOGIC_PRED_TYPE_H = 'H'
LOGIC_PRED_TYPE_S = 'S'
LOGIC_PRED_TYPE_X = 'X'

NEG = 'S-!a'
UNARY_SCOPAL_OP = ['S-a', 'S-!a']
BINARY_SCOPAL_OP = ["S-aAND!b", "S-a<=>b", "S-b=>a", "S-a=>b", "S-aANDb"]

NEQ = 'NEQ'
EQ = 'EQ'
HEQ = 'HEQ'
H = 'H'
ARG0 = 'ARG0'
ARG1 = 'ARG1'

OP2IX = {
    "aANDb": 0,
    "aORb": 1,
    "!a": 2,
    "!a=>b": 3,
    "aAND!b": 4,
    "a=>b": 5,
    "a<=>b": 6,
    "a": 7
}

schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "snt_id": {
                "type": "number"
            },
            "decoders": {
                "type": "object",
                "properties": {
                    "logic_expr": {
                        "type": "array",
                    },
                    "pred_func_used":  {
                        "type": "array",
                    }
                },
                "required": ["logic_expr", "pred_func_used"]
            },
            "encoders": {
                "type": "object",
                "properties": {
                    "pred_func_nodes": {
                        "type": "array"
                    },
                    "content_preds": {
                        "type": "array"
                    }
                },
                "required": ["pred_func_nodes", "content_preds"]
            }
        },
        "required": ["decoders", "encoders", "snt_id"]
    }
}

class TruthConditions(object):
    """Transform DMRS to a logical expression of a reading.

    Args:
        min_pred_func_freq (int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
        min_content_pred_freq (int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
        content_pred2cnt (dict): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
        pred_func2cnt (dict): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
        filter_min_freq (bool): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, config, to_ix, min_pred_func_freq, min_content_pred_freq, content_pred2cnt, pred_func2cnt, filter_min_freq, pred2ix, pred_func2ix):

        self.config = config
        self.filter_min_freq = filter_min_freq
        self.min_pred_func_freq = min_pred_func_freq
        self.min_content_pred_freq = min_content_pred_freq
        self.content_pred2cnt = content_pred2cnt
        self.pred_func2cnt = pred_func2cnt
        self.pred2ix = pred2ix
        self.pred_func2ix = pred_func2ix
        self.to_ix = to_ix

        self.discarded = False
        self.node2pred = defaultdict()
        # for decoders
        self.logic_expr = None
        self.pred_func_used = set()
        # for encoder
        self.content_preds = []
        self.pred_func_nodes = set()

    def _pred2ix(self, pred):
        if self.to_ix and self.pred2ix:
            if pred in self.pred2ix:
                # print (pred, self.pred2ix[pred], self.content_pred2cnt[self.pred2ix[pred]])
                return self.pred2ix[pred]
            else:
                return None
        else:
            return pred

    def _pred_func2ix(self, pred_func):
        if self.to_ix and self.pred_func2ix:
            if pred_func not in self.pred_func2ix:
                # infrequent
                return None
            else:
                return self.pred_func2ix[pred_func]
        else:
            return pred_func

    def _check_discard(self):
        if self.discarded:
            return
        # self.discarded = False
        if not self.logic_expr:
            self.discarded_reason = "no logic expr"
            self.discarded = True
        elif len(self.pred_func_nodes) < 2 or len(self.content_preds) < 2:
            self.discarded_reason = "too few pred func node/content preds"
            self.discarded = True
        else:
            pass
        
    def _get_node2pred(self):
        for node, node_prop in self.dmrs_nxDG.nodes(data = True):
            # if not node_prop['predicate'] in self.config['ignore']:
            self.node2pred[node] = node_prop['predicate']
    
    def _get_content_pred(self):
        for node, node_prop in self.dmrs_nxDG.nodes(data = True):
            if 'pos' in node_prop:
                if all([
                    self._is_content_node(node),
                    self._is_frequent(self._pred2ix(node_prop['predicate']), self.content_pred2cnt, self.min_content_pred_freq)
                ]):
                    self.content_preds.append(node_prop['predicate'])

    def _get_topmost_scope(self, curr_scope, original_scope):
        # print ("curr_scope:", curr_scope)
        topmost_scope = self.scope2topmost_scope.get(curr_scope)
        if topmost_scope != None:
            # print ("topmost_scope:", topmost_scope)
            return topmost_scope
        par_scope = self.scope2par_scope.get(curr_scope)
        if par_scope == original_scope:
            return False
        if par_scope != None:
            # print ("par_scope:", par_scope)
            return self._get_topmost_scope(par_scope, original_scope)
        else:
            return curr_scope

    def _is_coord_conj(self, node):
        return any([
            self._is_logic_pred_of_type(node, LOGIC_PRED_TYPE_X),
            self._is_logic_pred_of_type(node, LOGIC_PRED_TYPE_H),
            self._is_logic_pred_of_type(node, LOGIC_PRED_TYPE_C)
        ])
            
    def _get_coord_conj_obj_nodes(self):
        self.node2is_cc_obj = defaultdict(bool)
        self.top_scope2cc_obj_scopes = defaultdict(set)
        self.be_arg2_node2arg1 = defaultdict(bool)
        self.node2is_be_arg2 = defaultdict(bool)
        self.be_node2arg1_node = defaultdict()
        for node in self.dmrs_nxDG.nodes():
            if self.node2pred[node] == '_be_v_id':
                # find out arg1 of _be_v_id (because currently only non-conjuncts are supported)
                be_arg1_node = list(filter(
                    lambda x: self.get_edge_arg_lbl(x[3]) == 'ARG1',
                    self.get_out_edges_arg(node, edge_types = [NEQ, EQ])
                ))
                be_arg2_node = list(filter(
                    lambda x: self.get_edge_arg_lbl(x[3]) == 'ARG2',
                    self.get_out_edges_arg(node, edge_types = [NEQ, EQ])
                ))
                if be_arg1_node != []:
                    be_arg1_node = be_arg1_node[0][1]
                    if not self._is_coord_conj(be_arg1_node):
                        self.be_node2arg1_node[node] = be_arg1_node
                if be_arg2_node != []:
                    be_arg2_node = be_arg2_node[0][1]
                    if not self._is_coord_conj(be_arg2_node):
                        self.node2is_be_arg2[be_arg2_node] = True
                        if be_arg1_node != []:
                            self.be_arg2_node2arg1[be_arg2_node] = be_arg1_node

        for node in self.dmrs_nxDG.nodes():
            if self._is_coord_conj(node):
                # if has NEQ and not with HNDL
                in_h_edges = self.get_in_edges_arg(node, edge_types = [HEQ, H])
                in_x_edges = self.get_in_edges_arg(node, edge_types = [NEQ, EQ])
                if in_x_edges and not [e for e in in_h_edges if self.get_edge_arg_lbl(e[3]) in ["L-HNDL", "R-HNDL"]]:
                    self.node2is_cc_obj[node] = True
                    for src, targ, key, lbl in in_x_edges:
                        self.top_scope2cc_obj_scopes[self.scope2topmost_scope[self.node2scope[src]]].add(self.node2scope[node])
                        # _be_v_id
                        if self.node2pred[src] == "_be_v_id" and self.get_edge_arg_lbl(lbl) == "ARG2":
                            if src in self.be_node2arg1_node:
                                self.be_arg2_node2arg1[node] = self.be_node2arg1_node[src]
        
        cc_nodes_new_temp = self.node2is_cc_obj | self.node2is_trsprt_obj
        while cc_nodes_new_temp:
            cc_nodes_new = cc_nodes_new_temp.copy()
            cc_nodes_new_temp = set()
            for node in cc_nodes_new:
                out_edges_arg = self.get_out_edges_arg(node, edge_types = [NEQ])
                for src, targ, key, lbl in out_edges_arg:
                    if self._is_coord_conj(targ):
                        self.node2is_cc_obj[targ] = True
                        cc_nodes_new_temp.add(targ)
                        if self.be_arg2_node2arg1.get(node) != None:
                            self.be_arg2_node2arg1[targ] = self.be_arg2_node2arg1.get(node)
                    elif self.be_arg2_node2arg1.get(node) != None:
                        self.node2is_be_arg2[targ] = True
                        self.be_arg2_node2arg1[targ] = self.be_arg2_node2arg1.get(node)

    def _get_transparent_obj_nodes(self):
        self.node2is_trsprt_obj = defaultdict(bool)
        for node in self.dmrs_nxDG.nodes():
            if node not in self.node2pred:
                print (self.node2pred)
                print (node)
            if self.node2pred[node] in self.config['transparent_preds']:
                # if has NEQ and not with HNDL
                in_x_edges = self.get_in_edges_arg(node, edge_types = [NEQ, EQ])
                if in_x_edges:
                    self.node2is_trsprt_obj[node] = True

    def _get_scopes(self):
        
        self.scope2nodes = defaultdict()
        self.node2scope = defaultdict()
        self.node2qtfr = defaultdict()
        self.node2outscoped = defaultdict(bool)
        self.scope2par_scope = defaultdict()
        self.scope2topmost_scope = defaultdict()
        self.functor_scope2cc_scopes = defaultdict(set)
        self.functor_node2cc_nodes = defaultdict(set)
        self.scope2functor_node2cc_nodes = defaultdict(lambda: defaultdict(set))
        scopes = DisjointSet()
        coord_conj_deps = DisjointSet()
        self.outscoped = DisjointSet()
        self.coord_conj_deps = DisjointSet()
        self.node2cc_dep = defaultdict()
        num_scope = 0
        
        for src, targ, lbl in self.dmrs_nxDG.edges(data = 'label'):
            if lbl == 'RSTR/H':
                # print (self.dmrs_nxDG.nodes[src])
                # assert all([
                #             self.dmrs_nxDG.nodes[src]['pos'] == 'q'
                #         ])
                self.node2qtfr[targ] = self.dmrs_nxDG.nodes[src]['predicate']
                continue
            elif lbl.endswith("/EQ"):
                scopes.union(src, targ)
                self.outscoped.union(src, targ)
            else:
                scopes.find(src)
                scopes.find(targ)
                if self._is_edge_scopal(lbl):
                    self.node2outscoped[targ] = True
                    self.outscoped.union(src, targ)

        self.scope2nodes = {idx: list(d_set) for idx, d_set in enumerate(scopes.itersets())}

        for idx, d_set in self.scope2nodes.items():
            for node in d_set:
                self.node2scope[node] = idx


        for src, targ, lbl in self.dmrs_nxDG.edges(data = 'label'):
            if lbl in ['RSTR/H']:
                continue
            if self._is_coord_conj(targ) and not self._is_edge_scopal(lbl):
                coord_conj_deps.find(targ)
                self.scope2functor_node2cc_nodes[self.node2scope[src]][src].add(targ)
                if not self.outscoped.connected(src, targ):
                    self.functor_node2cc_nodes[src].add(targ)
                    if self._is_coord_conj(src):
                        coord_conj_deps.union(src, targ)
                # if lbl.endswith("/EQ"):
                    # self.scope2functor_node2cc_nodes[self.node2scope[src]][src].add(targ)
                if not lbl.endswith("/EQ"):
                    self.functor_scope2cc_scopes[self.node2scope[src]].add(
                        self.node2scope[targ]
                    )
            if self._is_edge_scopal(lbl):
                src_scope = self.node2scope[src]
                targ_scope = self.node2scope[targ]
                if any([
                    src_scope == targ_scope
                ]):
                    self.discarded = True
                    self.discarded_reason = "scopal edge connects same scopes"
                    return
                else:
                    # print (self.dmrs_nxDG.edges(data = 'label'))
                    self.scope2par_scope[targ_scope] = src_scope

        try:
            for scope in self.scope2nodes:
                topmost_scope = self._get_topmost_scope(scope, scope)
                if not topmost_scope is False:
                    self.scope2topmost_scope[scope] = topmost_scope
                else:
                    self.discarded = True
                    self.discarded_reason = "qeq cycle"
                    return
        except:
            self.discarded = True
            self.discarded_reason = "error in get_topmost_scope"
            for src, targ, lbl in self.dmrs_nxDG.edges(data = 'label'):
                if lbl in ['RSTR/H']:
                    continue
                if self._is_edge_scopal(lbl):
                    # print (src, targ, lbl)
                    self.scope2par_scope[self.node2scope[targ]] = self.node2scope[src]  

        topo_sorted_nodes = list(toposort(self.functor_node2cc_nodes))
        self.cc_dep_idx2nodes = {idx: list(d_set) for idx, d_set in enumerate(coord_conj_deps.itersets())}

        self.cc_dep_idx2sorted_nodes = defaultdict(list)
        for cc_dep_idx, nodes in self.cc_dep_idx2nodes.items():
            cc_dep = {node: self.functor_node2cc_nodes[node] for node in nodes}
            topo_sorted_nodes = list(toposort(cc_dep))
            topo_sorted_nodes_flattened = [node for nodes in topo_sorted_nodes for node in nodes]
            self.cc_dep_idx2sorted_nodes[cc_dep_idx] = topo_sorted_nodes_flattened
            self.cc_dep_idx2sorted_nodes[cc_dep_idx].reverse()
        self.node2cc_dep_idx = {node: idx for idx, nodes in self.cc_dep_idx2sorted_nodes.items() for node in nodes}

        return True

    def _is_content_node(self, node):
        node_prop = self.dmrs_nxDG.nodes[node]
        pred, pred_pos = node_prop['predicate'], node_prop['pos']
        is_content_node = all([
            pred_pos in self.config["content_pos"] or pred in self.config['abs_pred_func']['carg'],
            not any([
                self._is_logic_pred_of_type(node, LOGIC_PRED_TYPE_X),
                self._is_logic_pred_of_type(node, LOGIC_PRED_TYPE_H),
                self._is_logic_pred_of_type(node, LOGIC_PRED_TYPE_C),
                self._is_logic_pred_of_type(node, LOGIC_PRED_TYPE_S)
            ]),
            not pred in self.config['ignore'],
            not pred == "_be_v_id",
            not pred in self.config['modals'],
        ])
        if is_content_node:
            if not self._has_pred_func(node):
                print ("content but not pf:", self.node2pred[node])
        return is_content_node

    def _has_pred_func(self, node):
        pred, pred_pos = self.dmrs_nxDG.nodes[node]['predicate'], self.dmrs_nxDG.nodes[node]['pos']
        return any([
            all([
                # not pred_pos in ["S", "q"],
                pred_pos in ["n", "a", "v"], # "p"],
                not any([
                    self._is_logic_pred_of_type(node, LOGIC_PRED_TYPE_X),
                    self._is_logic_pred_of_type(node, LOGIC_PRED_TYPE_H),
                    self._is_logic_pred_of_type(node, LOGIC_PRED_TYPE_C),
                    self._is_logic_pred_of_type(node, LOGIC_PRED_TYPE_S)
                ]),
                # not self._get_scopal_adv_out_edge(node)[0] and self.config["ignore_scopal_adv"],
                not pred in self.config['ignore'],
                not pred == "_be_v_id"
            ]),
            # pred in self.config['abs_pred_func']['sem'],
            pred in self.config['abs_pred_func']['carg']
            # pred in self.config['abs_pred_func']['cpd']
            # pred in self.config['abs_pred_func']['neg']
        ])

    def _has_intr_var(self, node):
        pred, pred_pos = self.dmrs_nxDG.nodes[node]['predicate'], self.dmrs_nxDG.nodes[node]['pos']
        return any([
            all([
                # not pred_pos in ["S", "q"],
                pred_pos in ["n", "a", "v"],# "p"],
                not self._is_logic_pred_of_type(node, LOGIC_PRED_TYPE_S),
                # self._get_scopal_adv_out_edge(node)[1] and self.config["ignore_scopal_adv"],
                not pred in self.config['ignore'],
                not pred == "_be_v_id"
            ]),
            # pred in self.config['abs_pred_func']['sem'],
            pred in self.config['abs_pred_func']['carg'],
            pred in self.config['transparent_preds']
            # pred in self.config['abs_pred_func']['neg']
        ])

    def _is_logic_pred_of_type(self, node, op_type):
        # check only the predicate
        pred = self.node2pred[node]
        return pred in self.config['logical_preds'][op_type]
        
    def _get_coord_conj_obj2targ(self, node):
        # check also the arguments of node; return the set of outedges
        arg2targ, op = None, None
        pred = self.node2pred[node]
        for op_type in [LOGIC_PRED_TYPE_C, LOGIC_PRED_TYPE_H, LOGIC_PRED_TYPE_X]:
            if self._is_logic_pred_of_type(node, op_type):
                req_args, op = self.config['logical_preds'][op_type][pred]
                # args_wo_scope = [arg.split("/")[0] for arg in args]
                out_edges = [e for e in self.dmrs_nxDG.out_edges(node, keys = True, data = 'label') if self.get_edge_arg_lbl(e[3]) in req_args]
                arg2targ = {self.get_edge_arg_lbl(e[3]): e[1] for e in out_edges}
                if any([
                    set(arg2targ.keys()) == set(req_args),
                    arg2targ and op_type in [LOGIC_PRED_TYPE_H, LOGIC_PRED_TYPE_X],
                    any([
                        'L-INDEX' in arg2targ and 'L-HNDL' in arg2targ,
                        'R-INDEX' in arg2targ and 'R-HNDL' in arg2targ
                    ]) and op_type == LOGIC_PRED_TYPE_C
                ]):
                    break
        return arg2targ, op

    def _get_logic_node_out_edges(self, node):
        # check also the arguments of node; return the set of outedges
        pred = self.node2pred[node]
        # if self._is_logic_pred_of_type(node, LOGIC_PRED_TYPE_S):# and not pred in self.config['ignore']:
        req_args, op = self.config['logical_preds'][LOGIC_PRED_TYPE_S][pred]
        out_edges = [e for e in self.get_out_edges_arg(node) if e[3] in req_args]
        out_edges_lbl = [e[3] for e in out_edges]
        if out_edges_lbl == req_args:
            return out_edges, op
        if out_edges_lbl[::-1] == req_args:
            return out_edges[::-1], op
        elif op in ["{}-aANDb".format(LOGIC_PRED_TYPE_S), "{}-aAND!b".format(LOGIC_PRED_TYPE_S)] and len(out_edges) == 1:
            if out_edges_lbl[0] == req_args[0]:
                return [out_edges[0], None], op
            elif out_edges_lbl[0] == req_args[1]:
                return [None, out_edges[0]], op
        return [None, None], op

    def _get_scopal_node_out_edge(self, node):
        # currently handle also scopal verbs, nouns, adjectives ...
        # want(w, x, h) => keep want(w, x)? and repeated want(w, x) if I want apples and oranges? (problematic if using product fuzzy logic)
        pred = self.node2pred[node]
        out_scopal_edges = [e for e in self.dmrs_nxDG.out_edges(node, keys = True, data = 'label') if self._is_edge_scopal(e[3])]
        out_nonscopal_edges = [e for e in self.dmrs_nxDG.out_edges(node, keys = True, data = 'label') if not self._is_edge_scopal(e[3])]
        return out_scopal_edges, out_nonscopal_edges
    
    def _get_pred_func_name(self, pred, arg, rm_sec_lbl = True):
        if rm_sec_lbl:
            arg = self.get_edge_arg_lbl(arg)
        pred_func_name = pred + "@" + arg
        return self._pred_func2ix(pred_func_name)

    @staticmethod
    def _is_edge_scopal(edge_lbl):
        return edge_lbl.endswith("/H") or edge_lbl.endswith("/HEQ")

    def _is_frequent(self, key, counter, min_freq):
        if not counter:
            return True
        if not self.filter_min_freq:
            return True
        if key == None:
            return False
        else:
            return counter[key] >= min_freq

    def _get_pred_func_ix(self, pred_func_name, args):
        if not self.to_ix:
            pred_func = {
                # "pred_func_name": pred_func_name,
                "pf": pred_func_name,
                "args": args
            }
        else:
            pred_func = [pred_func_name, args]
        return pred_func

    def _op2ix(self, op):
        if self.to_ix:
            return OP2IX[op]
        else:
            return op

    def get_op(self, op_str):
        if self.to_ix:
            if op_str:
                op_str = op_str.split("-")[1]
        return self._op2ix(op_str)

    def _compose_expr_unary(self, op, expr):
        composed_expr = None
        if op and expr:
            composed_expr = [self.get_op(NEG), expr]
        else:
            composed_expr = expr
        return composed_expr

    def _compose_expr_binary(self, op, left_expr, right_expr):
        op = self.get_op(op)
        composed_expr = None
        compose_left, compose_right = False, False
        if left_expr:
            compose_left = True 
            # if isinstance(left_expr, dict):
            #     if not self._is_frequent(left_expr['pred_func'], self.pred_func2cnt, self.min_pred_func_freq):
            #         compose_left = False 
        if right_expr:
            compose_right = True 
            # if isinstance(right_expr, dict):
            #     if not self._is_frequent(right_expr['pred_func'], self.pred_func2cnt, self.min_pred_func_freq):
            #         compose_right = False

        if compose_left and compose_right:
            composed_expr = [op, left_expr, right_expr]
        elif compose_left:
            composed_expr = self._compose_expr_unary(
                False,
                left_expr
            )
        elif compose_right:
            composed_expr = self._compose_expr_unary(
                op in ["aAND!b"],#["S-aAND!b"],
                right_expr
            )
        return composed_expr

    def _count_outscoping_neg_qtfr(self):
        pass
    
    @staticmethod
    def get_edge_arg_lbl(edge_lbl):
        return edge_lbl.split("/")[0]

    @staticmethod
    def get_edge_sec_lbl(edge_lbl):
        return edge_lbl.split("/")[1]

    def get_out_edges_arg(self, node, edge_types = None):
        out_edges = [e for e in self.dmrs_nxDG.out_edges(node, keys = True, data = 'label')]
        if edge_types:
            out_edges_arg = [e for e in out_edges
                if e[3] != "MOD/EQ" and self.get_edge_sec_lbl(e[3]) in edge_types
            ]
        else:
            out_edges_arg = [e for e in out_edges
                if e[3] != "MOD/EQ"
            ]
        return out_edges_arg

    def get_in_edges_arg(self, node, edge_types = None):
        in_edges = [e for e in self.dmrs_nxDG.in_edges(node, keys = True, data = 'label')]
        if edge_types:
            in_edges_arg = [e for e in in_edges
                if e[3] != "MOD/EQ" and self.get_edge_sec_lbl(e[3]) in edge_types
            ]
        else:
            in_edges_arg = [e for e in in_edges
                if e[3] != "MOD/EQ"
            ]
        return in_edges_arg

    def _build_sub_arg0_logic_expr(self, curr_node):
        sub_arg0_logic_expr = None
        if self._has_pred_func(curr_node):
            curr_pred = self.node2pred[curr_node]
            arg0pred_func_name = self._get_pred_func_name(curr_pred, ARG0)
            if self._is_frequent(arg0pred_func_name, self.pred_func2cnt, self.min_pred_func_freq):
                # arg0pred_func = {"pred_func_name": arg0pred_func_name, "args": [curr_node]}
                arg0pred_func = self._get_pred_func_ix(arg0pred_func_name, [curr_node])
                self.pred_func_nodes.add(curr_node)
                self.pred_func_used.add(arg0pred_func_name)
            # also instantiate args of nominals, if any
                pred_funcs = None
            # if self.dmrs_nxDG.nodes[curr_node]['cvarsort'] == LOGIC_PRED_TYPE_X:
            #     out_edges_arg = self.get_out_edges_arg(curr_node)
            #     if out_edges_arg:
            #         # e.g., I heard the claim that I do not run and swim
            #         pred_funcs = self._get_pred_funcs(self, curr_node, out_edges_arg, curr_node, [{}])
            #         # pred_funcs = [{"pred_func": self._get_pred_func_name(self.node2pred[e[0]], e[3]), "args": [node, e[1]]} for e in out_edges]
            #         # sub_arg0_logic_expr = reduce(lambda e1, e2: self._compose_expr_binary("aANDb", e2, e1), pred_funcs)
                sub_arg0_logic_expr = self._compose_expr_binary("arg0-aANDb", arg0pred_func, pred_funcs)
        return sub_arg0_logic_expr

    def _get_pred_func(self, remote_edge, conj2node = {}, be_arg1_node = None):

        def _get_conj_targ_node(targ, conj2node):
            if conj2node and targ:
                conj_targ = None
                conj_targ_temp = conj2node.get(targ)
                while conj_targ_temp:
                    conj_targ = conj_targ_temp
                    conj_targ_temp = conj2node.get(conj_targ)
                if conj_targ != None:
                    targ = conj_targ
            return targ
                    
        pred_func = None
        src, targ, key, lbl = remote_edge
        pred = self.node2pred[src]
        pred_func_name = self._get_pred_func_name(pred, lbl)
        # arg0pred_func_name = self._get_pred_func_name(pred, ARG0)
        if pred == '_be_v_id' or all([
            self._is_frequent(pred_func_name, self.pred_func2cnt, self.min_pred_func_freq)#,
            # self._is_frequent(arg0pred_func_name, self.pred_func2cnt, self.min_pred_func_freq)
        ]):
            targ = _get_conj_targ_node(targ, conj2node)
            if pred == '_be_v_id' and self._has_pred_func(targ):
                # ignore if it is arg1
                if self.get_edge_arg_lbl(lbl) == "ARG2":
                    be_arg1_node = _get_conj_targ_node(self.be_node2arg1_node.get(src), conj2node)
                    if all([
                        be_arg1_node != None
                    ]):
                        if self._has_intr_var(be_arg1_node):
                            be_arg2_pred = self.node2pred[targ]
                            be_arg2_pred_arg0pred_func_name = self._get_pred_func_name(be_arg2_pred, ARG0)
                            be_arg1_pred = self.node2pred[be_arg1_node]
                            be_arg1_pred_arg0pred_func_name = self._get_pred_func_name(be_arg1_pred, ARG0)
                            if all([
                                self._is_frequent(be_arg2_pred_arg0pred_func_name, self.pred_func2cnt, self.min_pred_func_freq),
                                self._is_frequent(be_arg1_pred_arg0pred_func_name, self.pred_func2cnt, self.min_pred_func_freq)
                            ]):
                                pred_func = self._get_pred_func_ix(be_arg2_pred_arg0pred_func_name, [be_arg1_node])
                                self.pred_func_used.add(be_arg2_pred_arg0pred_func_name)
                                self.pred_func_nodes.add(be_arg1_node)
                                # if self.snt_id == "1000070100040":
                                #     print (self.pred_func_nodes, 1)
            elif self._has_pred_func(targ):
                targ_pred = self.node2pred[targ]
                targ_arg0pred_func_name = self._get_pred_func_name(targ_pred, ARG0)
                if self._is_frequent(targ_arg0pred_func_name, self.pred_func2cnt, self.min_pred_func_freq):
                    arg0_node = _get_conj_targ_node(self.be_arg2_node2arg1.get(src, src), conj2node)
                    if self._has_intr_var(arg0_node):
                        arg0pred_func_name = self._get_pred_func_name(self.node2pred[arg0_node], ARG0)
                        if self._is_frequent(arg0pred_func_name, self.pred_func2cnt, self.min_pred_func_freq):
                            pred_func = self._get_pred_func_ix(pred_func_name, [arg0_node, targ])
                            self.pred_func_used.add(pred_func_name)
                            self.pred_func_nodes.add(arg0_node)
                            # if self.snt_id == "1000070100040":
                            #     print (self.pred_func_nodes, 3)
                            self.pred_func_nodes.add(targ)
                            # if self.snt_id == "1000070100040":
                            #     print (self.pred_func_nodes, 2)
        return pred_func

    def _get_pred_funcs(self, curr_node, curr_scope, curr_scope_nodes, top_scopes, conj2node, node_edge_idx_tbd, be_arg1_node):

        full_pred_func = None

        remote_node, remote_edge_idx = None, None
        start = True
        if node_edge_idx_tbd != None:
            remote_node, remote_edge_idx = node_edge_idx_tbd
            if remote_node == curr_node:
                start = False

        cc_arg_found = False
        for edge_idx, curr_edge in enumerate(self.node2pred_func_order[curr_node]):
            # print ("getting pf", curr_edge, conj2node, node_edge_idx_tbd)
            pred_func = None
            op = None
            src, targ, key, lbl = curr_edge
            if start == False and remote_node == curr_node:
                if edge_idx == remote_edge_idx:
                    start = True
            if not start:
                continue
            if self.node2is_trsprt_obj[targ]:
                for targ_src, targ_targ, targ_key, targ_lbl in self.get_out_edges_arg(targ):
                    if self.get_edge_arg_lbl(targ_lbl) == ARG1:
                        targ = targ_targ
                        break
            if self._is_edge_scopal(lbl):
                # include want(w) -arg1/h-> x as want(w, x)?
                h_targ = targ
                next_scope = self.node2scope[h_targ]
                next_scope_nodes = self.scope2nodes[next_scope].copy()
                # print (curr_node, "F")
                pred_func = self._build_partial_logic_expr(next_scope, next_scope_nodes, [], conj2node, node_edge_idx_tbd, {}, be_arg1_node)
                op = "pred_func_/H-aANDb"
            elif self.node2is_cc_obj[targ]:
                if targ in conj2node:
                    pred_func = self._get_pred_func(curr_edge, conj2node, be_arg1_node)
                    op = "pred_func_expanded_cc-aANDb"
                else:
                    # cc_dep_idx = self.node2cc_dep_idx[targ]
                    # cc_dep_nodes = self.cc_dep_idx2sorted_nodes[cc_dep_idx].copy()
                    # next_node = cc_dep_nodes.pop(0)
                    next_scope = self.node2scope[targ]
                    next_scope_nodes = self.scope2nodes[next_scope].copy()
                    next_scope_nodes.remove(targ)
                    curr_scope_nodes.insert(0, curr_node)
                    top_scopes_new = [curr_scope]# + top_scopes
                    remote_scope2nodes = {curr_scope: curr_scope_nodes}
                    # print ("top_scopes_new:", top_scopes_new)
                    pred_func = self._expand_coord_conj(targ, next_scope, next_scope_nodes, top_scopes_new, conj2node, curr_scope, curr_scope_nodes, (curr_node, edge_idx), remote_scope2nodes, be_arg1_node)
                    op = "pred_func_cc-aANDb"
                    cc_arg_found = True
            elif self._has_intr_var(targ):# or self._is_coord_conj(targ):
                # print ("has:", targ)
                pred_func = self._get_pred_func(curr_edge, conj2node, be_arg1_node)
                op = "pred_func-aANDb"
            # print ("pred_func:", pred_func)
            if op and pred_func:
                full_pred_func = self._compose_expr_binary(
                    op,
                    pred_func,
                    full_pred_func
                )
            if cc_arg_found:
                break

        return full_pred_func
   

    def _expand_coord_conj(self, curr_node, curr_scope, curr_scope_nodes, top_scopes, conj2node, remote_scope, remote_scope_nodes, node_edge_idx_tbd, remote_scope2nodes = {}, be_arg1_node = None):
        # ref: https://github.com/delph-in/docs/wiki/SynSem_Problems_ScopalNonScopal
        if self.node2is_cc_obj[curr_node]:
            # cc: coordination conjunction
            cc_or_eq_exprs = [None, None] # if has handle arguments, proceed to dgtr scope; otherwise, compute EQ's expr
            cc_arg2targ, op = self._get_coord_conj_obj2targ(curr_node)
            # print (curr_node, cc_arg2targ)
            for idx, arg in enumerate(['L', 'R']):
                h_targ, e_targ = cc_arg2targ.get('{}-HNDL'.format(arg)), cc_arg2targ.get('{}-INDEX'.format(arg))
                if h_targ != None or e_targ != None:
                    if e_targ != None:
                        conj2node[curr_node] = e_targ
                        # print (curr_node, "-->", e_targ)
                    if h_targ != None:
                        has_dgtr_scope = True
                        next_scope = self.node2scope[h_targ]
                        next_scope_nodes = self.scope2nodes[next_scope].copy()
                        # print (curr_node, "A")
                        cc_expr = None
                        if not self.node2is_cc_obj[h_targ]:
                            cc_expr = self._build_partial_logic_expr(next_scope, next_scope_nodes, [], conj2node, None, remote_scope2nodes.copy(), be_arg1_node)
                        # print (curr_node, "B")
                        # eq_expr = self._build_partial_logic_expr(remote_scope, remote_scope_nodes.copy(), top_scopes.copy(), conj2node, node_edge_idx_tbd, remote_scope2nodes.copy())
                        eq_expr = self._expand_coord_conj(h_targ, curr_node, curr_scope, top_scopes.copy(), conj2node, remote_scope, remote_scope_nodes.copy(), node_edge_idx_tbd, remote_scope2nodes.copy(), be_arg1_node)
                        cc_or_eq_exprs[idx] = self._compose_expr_binary('cc^eq-aANDb', cc_expr, eq_expr)
                    elif e_targ != None:
                        # print ("ECC:", e_targ, top_scopes, arg)
                        # cc_or_eq_exprs[idx] = self._build_partial_logic_expr(curr_scope, curr_scope_nodes, top_scopes, conj2node, scope2nodes_tbd.copy())
                        next_scope = self.node2scope[e_targ]
                        next_scope_nodes = [e_targ]
                        top_scopes_new = [curr_scope] + top_scopes
                        cc_or_eq_exprs[idx] = self._expand_coord_conj(e_targ, next_scope, next_scope_nodes, top_scopes_new.copy(), conj2node, curr_scope, curr_scope_nodes, node_edge_idx_tbd, remote_scope2nodes.copy(), be_arg1_node)
                    if curr_node in conj2node:
                        del conj2node[curr_node]
                    # cc_eq_exprs[idx] = self._compose_expr_binary('aANDb-cc^eq', cc_expr[idx], eq_expr[idx])
            cc_or_eq_expr = self._compose_expr_binary(op, cc_or_eq_exprs[0], cc_or_eq_exprs[1])
            # if curr_node in conj2node:
            #     del conj2node[curr_node]
            return cc_or_eq_expr
        else:
            return self._build_partial_logic_expr(None, [], top_scopes.copy(), conj2node, node_edge_idx_tbd, remote_scope2nodes, be_arg1_node)


    def _build_partial_logic_expr(self, curr_scope, curr_scope_nodes, top_scopes, conj2node, node_edge_idx_tbd = None, remote_scope2nodes = {}, be_arg1_node = None):
        
        agg_partial_logic_expr = None

        partial_logic_expr = None
        # print (curr_scope, curr_scope_nodes, top_scopes, conj2node, node_edge_idx_tbd, remote_scope2nodes)

        if curr_scope in remote_scope2nodes:
            curr_scope_nodes = remote_scope2nodes[curr_scope]
            del remote_scope2nodes[curr_scope]
            # print ("curr_scope_nodes:", curr_scope_nodes)

        if curr_scope_nodes:

            curr_node = curr_scope_nodes.pop(0)
            curr_pred = self.node2pred[curr_node]

            if self.node2is_cc_obj[curr_node] and not node_edge_idx_tbd and curr_node not in conj2node:
                top_scopes = [curr_scope] + top_scopes
                # print (curr_node, "C")
                return self._expand_coord_conj(curr_node, curr_scope, curr_scope_nodes, top_scopes, conj2node, None, [], node_edge_idx_tbd, remote_scope2nodes, be_arg1_node)
            elif not self.node2is_cc_obj[curr_node] and not self.node2is_trsprt_obj[curr_node]:
                pred_funcs_logic_expr = None
                scopal_expr = None
                # either a predicate with predicate functions ...
                if self._has_pred_func(curr_node) or curr_pred == '_be_v_id':
                    pred_funcs_logic_expr = self._get_pred_funcs(curr_node, curr_scope, curr_scope_nodes.copy(), top_scopes, conj2node, node_edge_idx_tbd, be_arg1_node)
                    # print (curr_pred, "full_pf:", pred_funcs_logic_expr)
                # or a scopal logical operator ...
                elif self._is_logic_pred_of_type(curr_node, LOGIC_PRED_TYPE_S) or self._is_coord_conj(curr_node):
                    scopal_exprs = [None, None]
                    op = None
                    if self._is_logic_pred_of_type(curr_node, LOGIC_PRED_TYPE_S):
                        scopal_out_edges, op = self._get_logic_node_out_edges(curr_node)
                        # print (curr_node, scopal_out_edges)
                        if op in UNARY_SCOPAL_OP:
                            # assert len(scopal_out_edges) == 1
                            pass
                        for idx, e in enumerate(scopal_out_edges):
                            if e != None:
                                h_targ = e[1]
                                next_scope = self.node2scope[h_targ]
                                next_scope_nodes = self.scope2nodes[next_scope].copy()
                                # print (curr_node, "D")
                                scopal_exprs[idx] = self._build_partial_logic_expr(next_scope, next_scope_nodes, [], conj2node, node_edge_idx_tbd, remote_scope2nodes.copy(), be_arg1_node)
                            else:
                                scopal_exprs[idx] = None
                    else:
                        cc_arg2targ, op = self._get_coord_conj_obj2targ(curr_node)
                        for idx, arg in enumerate(['L', 'R']):
                            h_targ, e_targ = cc_arg2targ.get('{}-HNDL'.format(arg)), cc_arg2targ.get('{}-INDEX'.format(arg))
                            if h_targ != None:
                                has_dgtr_scope = True
                                next_scope = self.node2scope[h_targ]
                                next_scope_nodes = self.scope2nodes[next_scope].copy()
                                # print (curr_node, "E")
                                scopal_exprs[idx] = self._build_partial_logic_expr(next_scope, next_scope_nodes, [], conj2node, node_edge_idx_tbd, remote_scope2nodes.copy(), be_arg1_node)
                            else:
                                scopal_exprs[idx] = None
                    if op in BINARY_SCOPAL_OP or self._is_coord_conj(curr_node):
                        if op:
                            scopal_expr = self._compose_expr_binary(op, scopal_exprs[0], scopal_exprs[1])
                    elif op in UNARY_SCOPAL_OP:
                        scopal_expr = self._compose_expr_unary(op == NEG, scopal_exprs[0])
                # or not on the scopal operator list, but contains scopal argments
                # else:
                #     scopal_out_edges, out_nonscopal_edges = self._get_scopal_node_out_edge(curr_node)
                #     for idx, e in enumerate(scopal_out_edges):
                #         if e:
                #             h_targ = e[1]
                #             next_scope = self.node2scope[h_targ]
                #             next_scope_nodes = self.scope2nodes[next_scope].copy()
                #             print (curr_node, "F")
                #             next_scope_logic_expr = self._build_partial_logic_expr(next_scope, next_scope_nodes, top_scopes.copy(), conj2node, node_edge_idx_tbd, remote_scope2nodes.copy())
                #             scopal_expr = self._compose_expr_binary("aANDb-/H", next_scope_logic_expr, scopal_expr)
                pf_scopal_expr = self._compose_expr_binary('2in1-aANDb', pred_funcs_logic_expr, scopal_expr)
                # print (curr_node, "G")
                eq_expr = self._build_partial_logic_expr(curr_scope, curr_scope_nodes, top_scopes, conj2node, node_edge_idx_tbd, remote_scope2nodes, be_arg1_node)
                partial_logic_expr = self._compose_expr_binary('pf^eq-aANDb', pf_scopal_expr, eq_expr) 
            else:
                partial_logic_expr = self._build_partial_logic_expr(curr_scope, curr_scope_nodes, top_scopes, conj2node, node_edge_idx_tbd, remote_scope2nodes, be_arg1_node)
            curr_scope_nodes.insert(0, curr_node)
            return partial_logic_expr
            
        else:
            next_partial_logic_expr = None
            if top_scopes:
                next_partial_top_scope = top_scopes.pop(0)
                next_partial_top_scope_nodes = self.scope2nodes[next_partial_top_scope].copy()
                # print (curr_scope, "H")
                next_partial_logic_expr = self._build_partial_logic_expr(next_partial_top_scope, next_partial_top_scope_nodes, top_scopes, conj2node, node_edge_idx_tbd, remote_scope2nodes, be_arg1_node)
                top_scopes.insert(0, next_partial_top_scope)

            return next_partial_logic_expr


    def _build_logic_expr(self):

        # e.g., I run or1(EQ) swim happily or2(EQ) sadly loudly(EQ)
        # => or1(o1, run(r), swim(s)) and or2(o2, happy(h, o1), sad(s, o1)) and loudly(o1)
        # for-loop (recursive) over coordinations? (within same scope?)

        # a) right or first
        # (happy(h, r) andEQ run(r) andEQ loudly(r) or1 happy(h, s) and swim(s) andEQ loudly(s))
        # or2
        # (sad(s, r) andEQ run(r) andEQ loudly(r) or1 sad(s, s) andEQ swim(s) andEQ loudly(s))

        # b) left or first
        # (run(r) andEQ (happy(h, r) or2 sad(s, r)) andEQ loudly(r))
        # or1
        # (swim(s) andEQ (happy(h, s) or2 sad(s, s)) andEQ loudly(s))

        # c) loudly first
        # loudly(r) andEQ run(r) or loudly(s) and swim(s)

        # what about e.g., I run,(imp) walk or1(EQ) swim happily or2(EQ) sadly loudly(EQ)
        # no dogs or(EQ) cats that eat(EQ) run(NEQ)
        # Not(eat(d) and run(d) or1 eat(c) and run(c))
        # https://delph-in.github.io/delphin-viz/demo/#input=I%20run,%20walk%20or%20swim%20happily%20or%20sadly%20loudly&count=5&grammar=erg1214-uw&tree=true&mrs=true&dmrs=true
        # (happy(h, r) andEQ run(r) andEQ loudly(r) imp (happy(h, w) andEQ walk(w) andEQ loudly(w) or1 happy(h, s) and swim(s) andEQ loudly(s))
        # or2
        # (sad(s, r) andEQ run(r) andEQ loudly(r) imp (sad(s, w) andEQ walk(w) andEQ loudly(w) or1 sad(s, s) andEQ swim(s) andEQ loudly(s))

        # print ("\n\n\n\n")
        
        full_logic_expr = None

        # TODO: _be_v_id is 'absorbed' to co-indexing predicates; _be_v_id should be removed from has_pred_func
        # discard infrequent arg0_pred_func
        arg0_logic_expr = None
        for node, node_prop in self.dmrs_nxDG.nodes(data = True):
            if self.node2is_be_arg2[node] != True and not self.node2pred[node] == "_be_v_id":
                sub_arg0_logic_expr = self._build_sub_arg0_logic_expr(node)
                arg0_logic_expr = self._compose_expr_binary("arg0-aANDb", sub_arg0_logic_expr, arg0_logic_expr)
        
        agg_partial_logic_expr = None
        node2visited = {node: False for node in self.dmrs_nxDG.nodes}
        top_scopes = [scope for scope, nodes in self.scope2nodes.items() if all([not self.node2outscoped[node] for node in nodes])]
            # par_scope = self.scope2par_scope.get(functor_scope)

            # while par_scope != None:
            #     scope2desc_cc_scopes[par_scope].update(list(cc_scopes))
            #     par_scope = self.scope2par_scope.get(par_scope)

            # functor_top_scope = self._get_topmost_scope(functor_scope)
            # scope2desc_cc_scopes[functor_top_scope].update(list(cc_scopes))
        
        #remove any top scope that contains cc_arg from outside
        # print ("dep2sorted:", self.cc_dep_idx2sorted_nodes)
        # print ("node2dep:", self.node2cc_dep_idx)
        # print ("scope2topmost_scope:", self.scope2topmost_scope)

        # print (self.scope2nodes)
        # Top scope order
        functor_top2cc_top = defaultdict(set)
        top_scopes_with_cc_arg = set()

        for functor_scope, cc_scopes in self.functor_scope2cc_scopes.items():
            for cc_scope in cc_scopes:
                cc_top = self.scope2topmost_scope[cc_scope]
                functor_top = self.scope2topmost_scope[functor_scope]
                if cc_top != functor_top:
                    top_scopes_with_cc_arg.add(cc_top)
                    functor_top2cc_top[functor_top].add(cc_top)

        # print ("functor_top2cc_top:", functor_top2cc_top)
        topo_sorted_functor_cc_tops = list(toposort(functor_top2cc_top))
        topo_sorted_functor_cc_tops_flat = [scope for scopes in topo_sorted_functor_cc_tops for scope in scopes]
        top_scopes_ordered = [
            scope for scope in top_scopes if scope not in top_scopes_with_cc_arg and scope not in self.top_scope2cc_obj_scopes # topo_sorted_top_scopes_flattened
        ] + [scope for scope in self.top_scope2cc_obj_scopes if scope not in top_scopes_with_cc_arg] #+ topo_sorted_functor_cc_tops_flat
        # print ("top_scopes:", top_scopes)
        # print ("top_scope2cc_obj_scopes:", self.top_scope2cc_obj_scopes)
        # print ("top_scopes_ordered:", top_scopes_ordered)
        
        # EQ node order
        for scope in self.scope2nodes.copy():
            topo_sorted_nodes_flattened = []
            if self.scope2functor_node2cc_nodes[scope]:
                topo_sorted_nodes = list(toposort(self.scope2functor_node2cc_nodes[scope]))
                topo_sorted_nodes_flattened = [node for nodes in topo_sorted_nodes for node in nodes if node in self.scope2nodes[scope]]
            self.scope2nodes[scope] = [
                node for node in self.scope2nodes[scope] if node not in topo_sorted_nodes_flattened
            ] + topo_sorted_nodes_flattened#[::-1]
        # print (self.scope2nodes)

        # predicate function order
        self.node2pred_func_order = defaultdict(list)
        for node, node_prop in self.dmrs_nxDG.nodes(data = True):
            if self._has_pred_func(node) or self.node2pred[node] == '_be_v_id':
                out_edges_arg = self.get_out_edges_arg(node)
                arg_type2edges = defaultdict(list)
                for src, targ, key, lbl in out_edges_arg:
                    if self.node2is_cc_obj[targ]:
                        arg_type2edges['cc'].append((src, targ, key, lbl))
                    else:
                        arg_type2edges['non-cc'].append((src, targ, key, lbl))
                self.node2pred_func_order[node] = arg_type2edges['non-cc'] +  arg_type2edges['cc']
        # print ("node2pred_func_order:", self.node2pred_func_order)
        
        # mark conj2node for nominalization and eventualities
        conj2node = defaultdict()
        for node, node_prop in self.dmrs_nxDG.nodes(data = True):
            if node_prop['predicate'] in self.config['transparent_preds']:
                for src, targ, key, lbl in self.get_out_edges_arg(node):
                    if self.get_edge_arg_lbl(lbl) == ARG1:
                        conj2node[node] = targ
                        break

        be_arg1_node = None


        # cc_arg_scopes = set()
        # if self.functor_scope2cc_scopes:
        #     cc_arg_scopes = set(reduce(lambda x, y: x.union(y), self.functor_scope2cc_scopes.values()))

        # top_scopes_wo_cc_arg = [scope for scope in top_scopes if scope not in cc_arg_scopes]
        # # sort the expansion order (factorized is better)
        # # scope with coordination as dgtr scope first
        # # within each scope, put coordination first
        # # partial order for both?
        # cc_functor_scopes_ordered = []
        # topo_sorted_scopes = list(toposort(self.functor_scope2cc_scopes))
        # print (self.scope2nodes)
        # print (top_scopes)
        # print (self.functor_scope2cc_scopes)
        # print (top_scopes_wo_cc_arg)
        # print (topo_sorted_scopes)
        # for scopes in topo_sorted_scopes:
        #     for scope in scopes:
        #         topmost_scope = self._get_topmost_scope(scope)
        #         cc_functor_scopes_ordered.append(topmost_scope)
        # cc_functor_scopes_ordered = list(dict.fromkeys(cc_functor_scopes_ordered))
        # top_scopes_ordered = [
        #     scope for scope in top_scopes if scope not in cc_functor_scopes_ordered
        # ] + cc_functor_scopes_ordered
        # print (cc_functor_scopes_ordered)
        # print (top_scopes_ordered)
        # top_scopes_ordered vs. top_scopes_wo_cc_arg
        if top_scopes_ordered:
            curr_scope = top_scopes_ordered.pop(0)
            curr_scope_nodes = self.scope2nodes[curr_scope].copy()
            # scope_sources = [node for node in self.scope2nodes[curr_scope] if not self.node2outscoped[node]]
            remote_scope2nodes = defaultdict(list)
            agg_partial_logic_expr = self._build_partial_logic_expr(curr_scope, curr_scope_nodes, top_scopes_ordered, conj2node, None, remote_scope2nodes, be_arg1_node)
        full_logic_expr = self._compose_expr_binary("full-aANDb", arg0_logic_expr, agg_partial_logic_expr)
        self.logic_expr = full_logic_expr
        # # join argo here?
        # # TODO: if predicate only has arg0 and negation is on the predicate, negate the <pred>@arg0?
        # if self._has_pred_func(curr_node):
        #   curr_arg0func_name = self._get_pred_func_name(pred, "ARG0")
        #   curr_arg0_func = {"pred_func": curr_arg0func_name, "args": [curr_node]}
        # # for encoder
        # # TODO: how to handle infrequent pred_func? e.g., u_fwfwe@arg2(e, x)
        #   if self._is_frequent(curr_pred_func, self.pred_func2cnt, self.min_pred_func_freq):
        #       self.pred_func_nodes.add(curr_node)


    def __call__(self, sample):
        
        transformed = defaultdict()
        
        # for decoder
        pred2vars = defaultdict()
        
        snt_id = sample['id']
        self.snt_id = snt_id
        # print (snt_id)
        dmrs_nodelink_dict = sample['dmrs']
        self.dmrs_nxDG = nx.readwrite.json_graph.node_link_graph(dmrs_nodelink_dict)
        mapping = {node: ix for ix, node in enumerate(list(self.dmrs_nxDG.nodes())) }
        # print (mapping)
        self.dmrs_nxDG = nx.relabel_nodes(self.dmrs_nxDG, mapping)
        # print (self.dmrs_nxDG.nodes)
        
        self.discarded_reason = None
        self._get_node2pred()

        self._get_scopes()
            
        if not self.discarded:
            self._get_transparent_obj_nodes()


        if not self.discarded:
            # print ("cc nodes:", self.node2is_cc_obj)
            try:
                self._get_coord_conj_obj_nodes()
            except Exception as e:
                self.discarded = True
                self.discarded_reason = str(e)
                print (e)
                # erg_digraph = dg_util.Erg_DiGraphs()
                # erg_digraph.init_dmrs_from_nxDG(self.dmrs_nxDG)
                # erg_digraph.draw_dmrs(name = snt_id)
                # print ()

        if not self.discarded:
            # if self.to_ix:
                # self._build_logic_expr()
            if True: #not self.to_ix:
                try:
                    self._build_logic_expr()
                except Exception as e:
                    self.discarded = True
                    self.discarded_reason = str(e)
                    if not "recursion" in self.discarded_reason:
                        print (self.snt_id, e)
                    # erg_digraph = dg_util.Erg_DiGraphs()
                    # erg_digraph.init_dmrs_from_nxDG(self.dmrs_nxDG)
                    # erg_digraph.draw_dmrs(name = snt_id)
                    # print ()

        if not self.discarded:
            self._get_content_pred()

        self._check_discard()
        
        # convert to ix if ix map is available
        self.node2pred = {node: self._pred2ix(pred) for node, pred in self.node2pred.items() if self._pred2ix(pred) != None}
        self.content_preds = [self._pred2ix(pred) for pred in self.content_preds]

        if not self.discarded and not set(self.pred_func_nodes).issubset(set(self.node2pred.keys())):
            print (snt_id, self.node2pred, self.pred_func_nodes)
            erg_digraph = dg_util.Erg_DiGraphs()
            erg_digraph.init_dmrs_from_nxDG(self.dmrs_nxDG)
            erg_digraph.draw_dmrs(name = snt_id)
            logic_expr_save_path = os.path.join("figures", "logic_expr_{}.png".format(snt_id))
            # draw_logic_expr(self.logic_expr, save_path = logic_expr_save_path)
            print ()
            input()

        transformed = {
            "discarded": self.discarded,
            "discarded_reason": self.discarded_reason,
            "node2pred": self.node2pred,
            "pred_func_nodes": list(self.pred_func_nodes),
            "content_preds": self.content_preds,
            "logic_expr": self.logic_expr,
            "pred_func_used": list(self.pred_func_used)
        }

        return transformed
       
        
#         for node, node_prop in self.dmrs_nxDG.nodes(data = True):
            
#             pred = node_prop['predicate']
#             pred_lemma, pred_pos = node_prop['lemma'], node_prop['pos']
            
#             if pred_pos in self.config["content_pos"]:
#                 self.content_preds.append(pred)
#             if pred_pos in "q":
#                 if pred in self.config["neg_quantifier"]:
#                     pass
#                 continue
#             # non-quantifier or logical connectives
#             self.entity_preds.append(pred)
            
#             if not pred_pos in "S" or pred in self.config['abs_pred_func']['sem']:
#                 if pred in self.config['logical_preds'] and pred not in self.config['logical_preds']['!a']:
#                     self._factorize_conj(node, None, node, self.config['logical_preds'][node])
#                 else:
#                     self.node2tcf[node].append(((pred + ';' + "ARG0", pred), 1))
#                     self._factorize_conj(node, None, node, None)
                
                
#         pprint (self.node2tcf)
#         print ()
#         pprint (self.neg_scope)
        

        
#         return transformed


# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""

#     def __call__(self, sample):
#         image, landmarks = sample['image'], sample['landmarks']

#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C x H x W
#         image = image.transpose((2, 0, 1))
#         return {'image': torch.from_numpy(image),
#                 'landmarks': torch.from_numpy(landmarks)}
    