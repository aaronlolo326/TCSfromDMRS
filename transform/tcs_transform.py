import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

from collections import defaultdict, Counter
from pprint import pprint
from varname import nameof
import time

from disjoint_set import DisjointSet

from src import util, dg_util

TRUE = "<True>"

class TruthConditions(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, config,
        min_pred_func_freq, min_lex_pred_freq, lex_pred2cnt, pred_func2cnt,
        filter_min_freq):

        self.config = config
        self.filter_min_freq = filter_min_freq
        self.min_pred_func_freq = min_pred_func_freq
        self.min_lex_pred_freq = min_lex_pred_freq
        self.lex_pred2cnt = lex_pred2cnt
        self.pred_func2cnt = pred_func2cnt

        self.op2truth_v = {
            "aANDb": (1,1),
            "aORb": (1,1),
            "!a->b": (1,1),
            "a->b": (1,1),
            "b->a": (1,1),
            "a<->b": (1,1),
            "aAND!b": (1,0),
            "!a": (0,)
        }

        self.discarded = False

    def _check_discard(self):
        # TODO
        discarded = False
        if not self.logic_expr:
            discarded = True
        elif not self.pred_func_node or not self.lexical_preds:
            discarded = True
        else:
            pass
        self.discarded = discarded
        
    def _get_node2pred(self):
        self.node2pred = defaultdict()
        for node, node_prop in self.dmrs_nxDG.nodes(data = True):
            self.node2pred[node] = node_prop['predicate']
    
    def _get_lexical_pred(self):
        self.lexical_preds = []
        for node, node_prop in self.dmrs_nxDG.nodes(data = True):
            if 'pos' in node_prop:
                if self._is_lexical_pred(node_prop['predicate'], node_prop['pos']) and self._is_frequent(node_prop['predicate'], self.lex_pred2cnt, self.min_lex_pred_freq):
                    self.lexical_preds.append(node_prop['predicate'])

    def _get_scopes(self):
        
        self.node2scope = defaultdict()
        
        scopes = DisjointSet()
        num_scope = 0
        
        for src, targ, lbl in self.dmrs_nxDG.edges(data = 'label'):
            if lbl.endswith("/EQ"):
                scopes.union(src, targ)
            else:
                scopes.find(src)
                scopes.find(targ)
                
        self.scope2nodes = {idx: d_set for idx, d_set in enumerate(scopes.itersets())}

        for idx, d_set in self.scope2nodes.items():
            for e in d_set:
                self.node2scope[e] = idx

    def _is_lexical_pred(self, pred, pred_pos):
        return pred_pos in self.config["lexical_pos"] and not pred in self.config['ignore']

    def _has_pred_func(self, pred, pred_pos):
        return any([
            not pred_pos in ["S", "q"] and not self._is_logic_pred(pred) and not pred in self.config['ignore'],
            pred in self.config['abs_pred_func']['sem'],
            pred in self.config['abs_pred_func']['carg'],
            pred in self.config['abs_pred_func']['neg']
        ])

    def _has_intr_var(self, pred, pred_pos):
        return any([
            not pred_pos in ["S", "q"] and not self._is_logic_pred(pred) and not pred in self.config['ignore'],
            pred in self.config['abs_pred_func']['sem'], # include this?
            pred in self.config['abs_pred_func']['carg'],
            pred in self.config['abs_pred_func']['neg']
        ])
    def _is_logic_pred(self, pred):
        return pred in self.config['logical_preds'] and not pred in self.config['ignore']
    
    @staticmethod
    def _get_pred_func_name(pred, arg, rm_sec_lbl = True):
        if rm_sec_lbl:
            arg = arg.split("/")[0]
        return pred + "@" + arg

    def _is_frequent(self, key, counter, min_freq):
        if not self.filter_min_freq:
            return True
        else:
            return counter[key] >= min_freq

    def _compose_expr(self, op, left_expr, right_expr):
        composed_expr = None
        if isinstance(left_expr, dict):
            if not self._is_frequent(left_expr['pred_func'], self.pred_func2cnt, self.min_pred_func_freq):
                return composed_expr
        if isinstance(right_expr, dict):
            if not self._is_frequent(right_expr['pred_func'], self.pred_func2cnt, self.min_pred_func_freq):
                return composed_expr
        if left_expr and right_expr:
            composed_expr = [op, left_expr, right_expr]
        elif left_expr:
            composed_expr = left_expr
        elif right_expr:
            composed_expr = right_expr
        return composed_expr

    def _dfs(self, curr_node, remote_node, remote_edge, par_is_logic_pred, node2visited):
        
        sub_logic_expr = None
        
        node_prop = self.dmrs_nxDG.nodes[curr_node]
        pred, pred_lemma, pred_pos, cvarsort = [node_prop.get(k) for k in ['predicate', 'lemma', 'pos', 'cvarsort']]
                    
        if pred_pos in "q":
            if pred in self.config["neg_quantifier"]:
                pass
                # self.logic_expr.append(["AND", ["!", _dfs()]])
        
        if self._is_logic_pred(pred):

            args_op = self.config['logical_preds'][pred]
            args = args_op["args"]
            args_wo_scope = [arg.split("/")[0] for arg in args]
            op = args_op["op"]
            # print (pred, op)
            if op in self.op2truth_v:

                # out_edges = list(e for e in self.dmrs_nxDG.out_edges(curr_node, data = 'label') if e[2] in args)
                out_edges = list(e for e in self.dmrs_nxDG.out_edges(curr_node, data = 'label') if e[2].split("/")[0] in args_wo_scope)

                # To be done: neg
                if op == '!a':
                    pass
                # To be done: partial conj (without, say, L-INDEX)
                else:
                    if len(out_edges) == 2:
                    #     print (pred, out_edges, self.dmrs_nxDG.out_edges(curr_node, data = 'label'), args_wo_scope)
                    # print ([out_edges[0], out_edges[1]], "\t", args)
                        if [out_edges[0][2], out_edges[1][2]] == args:
                            sub_logic_expr = self._compose_expr(op,
                                                        self._dfs(out_edges[0][1], remote_node, remote_edge, True, node2visited),
                                                        self._dfs(out_edges[1][1], remote_node, remote_edge, True, node2visited))
                        elif [out_edges[1][2], out_edges[0][2]] == args:
                            sub_logic_expr = self._compose_expr(op,
                                                        self._dfs(out_edges[1][1], remote_node, remote_edge, True, node2visited),
                                                        self._dfs(out_edges[0][1], remote_node, remote_edge, True, node2visited))
                    elif len(out_edges) == 1:
                        sub_logic_expr = self._compose_expr("aANDb", # TBD
                                                       self._dfs(out_edges[0][1], remote_node, remote_edge, True, node2visited),
                                                       sub_logic_expr)

        if self._has_pred_func(pred, pred_pos):
            if par_is_logic_pred and remote_node and remote_edge:
                remote_edge_src, remote_edge_targ, remote_edge_key = remote_edge
                remote_pred = self.node2pred[remote_node]
                remote_edge_lbl = self.dmrs_nxDG.edges[remote_edge_src, remote_edge_targ, remote_edge_key]['label']
                # print ("remote:", remote_pred, remote_edge_lbl)
                sub_logic_expr = self._compose_expr("aANDb",
                                               {"pred_func": self._get_pred_func_name(remote_pred, remote_edge_lbl), "args": [remote_node, curr_node]},
                                               sub_logic_expr)
        
            

               
        if not node2visited[curr_node]:     
            
            node2visited[curr_node] = True
            
            if self._has_pred_func(pred, pred_pos):
                curr_pred_func = self._get_pred_func_name(pred, "ARG0")
                sub_logic_expr = self._compose_expr("aANDb",
                                               {"pred_func": curr_pred_func, "args": [curr_node]},
                                               sub_logic_expr)
                if self._is_frequent(curr_pred_func, self.pred_func2cnt, self.min_pred_func_freq):
                    self.pred_func_node.add(curr_node)
            
            out_edges = list(e for e in self.dmrs_nxDG.out_edges(curr_node, keys = True, data = 'label'))
            for src, targ, key, lbl in out_edges:
                targ_node_prop = self.dmrs_nxDG.nodes[targ]
                targ_pred, targ_pred_lemma, targ_pred_pos, targ_cvarsort = [targ_node_prop.get(k) for k in ['predicate', 'lemma', 'pos', 'cvarsort']]
                # if curr pred is a function AND targ pred has intrinsic variable; and edge label is not MOD/EQ
                if self._has_pred_func(pred, pred_pos) and self._has_intr_var(targ_pred, targ_pred_pos):
                    if lbl != "MOD/EQ":
                        sub_logic_expr = self._compose_expr("aANDb",
                                                       {"pred_func": self._get_pred_func_name(pred, lbl), "args": [curr_node, targ]},
                                                       sub_logic_expr)
                new_remote_node, new_remote_edge = None, None
                if self._has_pred_func(pred, pred_pos):
                    new_remote_node = curr_node
                    new_remote_edge = (src, targ, key)
                l_sub_logic_expr = self._dfs(targ, new_remote_node, new_remote_edge, False, node2visited)
                sub_logic_expr = self._compose_expr("aANDb",
                                               l_sub_logic_expr,
                                               sub_logic_expr)

                                  
        return sub_logic_expr        
    
    
    def _build_logic_expr(self):
        
        # for decoders
        self.logic_expr = None
        sub_logic_expr = None

        # for encoder
        self.pred_func_node = set()

        sources = [node for node in self.dmrs_nxDG.nodes if self.dmrs_nxDG.in_degree(node) == 0]
        node2visited = {node: False for node in self.dmrs_nxDG.nodes}
        
        for node in sources:                          
            sub_logic_expr = self._dfs(node, None, None, False, node2visited)
            self.logic_expr = self._compose_expr("aANDb", self.logic_expr, sub_logic_expr)
    
    def _draw_logic_expr(self, timestamp = False, name = "err"):
        
        def _build_tree(logic_expr_tree, sub_logic_expr, curr_node, par_node, edge_lbl):
            if isinstance(sub_logic_expr, str):
                logic_expr_tree.add_node(curr_node, label = sub_logic_expr)
            elif isinstance(sub_logic_expr, dict):
                logic_expr_tree.add_node(curr_node, label = "{} {}".format(sub_logic_expr['pred_func'], str(sub_logic_expr['args'])))
            elif sub_logic_expr:
                root, left, right = sub_logic_expr
                logic_expr_tree.add_node(curr_node, label = root)
                _build_tree(logic_expr_tree, left, curr_node*2, curr_node, 'a')
                _build_tree(logic_expr_tree, right, curr_node*2+1, curr_node, 'b')
            if par_node:
                logic_expr_tree.add_edge(par_node, curr_node, label = edge_lbl)

        
        logic_expr_tree = nx.DiGraph()
        _build_tree(logic_expr_tree, self.logic_expr, 1, None, None)
                
        time_str = "_" + time.asctime( time.localtime(time.time()) ).replace(" ", "-") if timestamp else ""
        save_path = "./figures/logic_expr_{}".format(name) + time_str + ".png"
        ag = to_agraph(logic_expr_tree)
        ag.layout('dot')
        ag.draw(save_path)
        # print ("logic expression tree drawn:", save_path)


    def __call__(self, sample):
        
        transformed = defaultdict()
        
        # for decoder
        pred2vars = defaultdict()
        
        snt_id = sample['id']
        # print (snt_id)
        dmrs_nodelink_dict = sample['dmrs']
        self.dmrs_nxDG = nx.readwrite.json_graph.node_link_graph(dmrs_nodelink_dict)
                                    # self.node2tcf = defaultdict(list)
                                    # self.neg_scope = Counter()
                                    # self.node2conj_args = defaultdict(list)
        
        # erg_digraph = dg_util.Erg_DiGraphs()
        # erg_digraph.init_dmrs_from_nxDG(self.dmrs_nxDG)
        # erg_digraph.draw_dmrs(name = snt_id)
        
        self._get_node2pred()

        self._get_lexical_pred()

        self._get_scopes()
        
        self._build_logic_expr()

        self._check_discard()
        
        # self._draw_logic_expr(name = snt_id)
        
        transformed = {
            "discarded": self.discarded,
            "node2pred": self.node2pred,
            "encoders": {
                "pred_func_node": self.pred_func_node,
                "lexical_preds": self.lexical_preds
            },
            "decoders": {
                "logic_expr": self.logic_expr
            }
        }

        return transformed
       
        
#         for node, node_prop in self.dmrs_nxDG.nodes(data = True):
            
#             pred = node_prop['predicate']
#             pred_lemma, pred_pos = node_prop['lemma'], node_prop['pos']
            
#             if pred_pos in self.config["lexical_pos"]:
#                 self.lexical_preds.append(pred)
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
    