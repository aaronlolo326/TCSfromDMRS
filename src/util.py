
def get_lemma_pos(predicate):
    # extract lemma pos
    pred_lemma, pred_pos = None, None

    if predicate == '_':
        pred_lemma, pred_pos = None, None
    # unknown predicate starts with 'u_'
    elif predicate.startswith('u_'):
        pred_lemma, pred_pos = predicate[2:].rsplit("_",1)
    elif predicate[0] == '_':
        
        try:
            assert predicate not in ['_']
        except:
            pass
            
        if "_u_unknown" in predicate and predicate not in ['_unknown_a_1', '_unknown_n_1']:
            pred_lemma, pred_unk_pos = predicate.rsplit("/",1)
            pred_lemma = pred_lemma[1:]
            pred_unk_pos_split = pred_unk_pos.rsplit("_", 2)
            assert pred_unk_pos_split[1:3] == ['u', 'unknown']
            pred_pos = pred_unk_pos_split[0]
            #pred_lemma = pred_lemma.replace('+',' ')[1:]
            
        elif predicate.count('_') not in [2,3]:
            pred_lemma, pred_pos, *_ = predicate.rsplit("_",2)
            pred_lemma = pred_lemma[1:]
            try:
                assert predicate in ['_only_child_n_1', '_nowhere_near_x_deg']
            except:
                pass
                # print (predicate)
            # if not predicate in ['_only_child_n_1', '_nowhere_near_x_deg']:
            #     print (predicate)
                
        else:
            _, pred_lemma, pred_pos, *_ = predicate.split("_")
            #pred_lemma = pred_lemma.replace('+',' ')
            if pred_pos == 'dir':
                pred_pos = 'p'
                print ("dir:", predicate)
            if pred_pos == 'state':
                pred_pos = 'p'
                print ("state:", predicate)
        
        try:
            assert pred_pos in 'a v n q x p c'.split(" ") or "_u_unknown" in predicate
        except:
            print (predicate)
        
        if [pred_lemma, pred_pos] == [None, None]:
            pred_lemma, pred_pos, *_ = predicate.rsplit("_", 2)
            #pred_lemma = pred_lemma.replace('+',' ')[1:]
            print ("fallback:", predicate, pred_lemma, pred_pos)
            
    else:
        if "_" in predicate:
            pred_lemma_S, pred_pos_S, *_ = predicate.split("_")
            if pred_pos_S == 'q':
                pred_lemma, pred_pos = pred_lemma_S, pred_pos_S
        else:
            pred_lemma, pred_pos = predicate, "S"

    return pred_lemma, pred_pos

def is_data_json(file):
    return file.endswith(".json") and not file.endswith('-checkpoint.json')

    
