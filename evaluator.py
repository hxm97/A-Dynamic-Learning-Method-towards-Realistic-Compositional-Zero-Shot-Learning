import torch
import numpy as np
import copy
from scipy.stats import hmean

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_mask(device, mask):
    if device == '1080Ti':
        return torch.ByteTensor(mask)
    else:
        return torch.BoolTensor(mask)

class Evaluator:

    def __init__(self, dset, model, cfg):

        self.dset = dset
        self.cfg = cfg
        pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.all_pairs]      #1962
        self.train_pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.tr_pairs]        #1398
        self.te_uc_pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.te_uc_pairs]     #193
        self.te_up_pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.te_up_pairs]     #189
        self.pairs = torch.LongTensor(pairs)
        
        if dset.phase == 'train':
            print('Evaluating with train pairs')
            test_pair_set = set(dset.tr_pairs)
        elif dset.phase == 'val':
            print('Evaluating with validation pairs')
            test_pair_set = set(dset.vl_uc_pairs + dset.vl_up_pairs + dset.tr_pairs)
            seen_concept_set = set(dset.tr_pairs+dset.vl_uc_pairs)
        else:
            print('Evaluating with test pairs')
            test_pair_set = set(dset.te_uc_pairs + dset.te_up_pairs + dset.tr_pairs)
            seen_concept_set = set(dset.tr_pairs+dset.te_uc_pairs)

        self.test_pair_dict = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in test_pair_set]
        self.test_pair_dict = dict.fromkeys(self.test_pair_dict, 0)
       
        for attr, obj in test_pair_set:
            pair_val = dset.val_pair2idx[(attr,obj)]
            key = (dset.attr2idx[attr], dset.obj2idx[obj])
            self.test_pair_dict[key] = [pair_val, 0, 0]

        if dset.open_world:
            masks = [1 for _ in dset.all_pairs]
        else:
            masks = [1 if pair in test_pair_set else 0 for pair in dset.val_pairs]
        self.closed_mask = generate_mask(cfg.device, masks)
        
        seen_pair_set = set(dset.tr_pairs)
        mask = [1 if pair in seen_pair_set  else 0 for pair in dset.val_pairs]
        self.seen_mask = generate_mask(cfg.device, mask)        #[True, True, True, ..., False]
        
        seen_concept_mask = [1 if pair in seen_concept_set else 0 for pair in dset.val_pairs]
        self.seen_comp_mask = generate_mask(cfg.device, seen_concept_mask)       #[True, True, True, ..., False]
        
        self.score_model = self.score_manifold_model
    
    def evaluate(self, scores, pair_label, attr_label, obj_label, topk=3):
        
        scores = {k: v.to('cpu') for k, v in scores.items()}
        scores = torch.stack([scores[(attr,obj)] for attr, obj in self.dset.val_pairs], 1) #21670*1780, test_images*(train_pairs+test_unseen_comps+test_unseen_pairs)
        batch_num = scores.shape[0]
        
        attr_label, obj_label = attr_label.to('cpu'), obj_label.to('cpu')
        pairs = list(zip(list(attr_label.numpy()), list(obj_label.numpy())))
        
        best_hm, hm_seen, best_seen, best_unseen_comp, best_unseen_pair, hm_unseen_comp, hm_unseen_pair, hm_bias1, hm_bias2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        pair_label = pair_label.unsqueeze(1)
        pair_label = pair_label.repeat(1, topk)        #test_images*topk
        mask1 = self.seen_mask.repeat(scores.shape[0],1) # Repeat mask along pairs dimension
        mask2 = self.seen_comp_mask.repeat(scores.shape[0], 1)
        
        #best seen acc
        score_test = scores.clone().detach()
        score_test[~mask1] -= 10.0 # Add bias to test uc and up pairs
        _, pred = score_test.topk(topk, dim=1)     #test_images*topk
        correct = torch.eq(pair_label, pred)        #test_images*topk
        correct = correct.any(1).float()        #test_images
        seen_corr, seen_false= 0, 0
        for i in range(batch_num):
            if pairs[i] in self.train_pairs:
                if correct[i] == 1:
                    seen_corr += 1     #10107
                else:
                    seen_false += 1 
        best_seen_acc = float(seen_corr/(seen_corr+seen_false))
        
        #best unseen comp acc
        score_test = scores.clone().detach()
        score_test[~mask1] += 10.0 # Add bias to test uc and up pairs
        score_test[~mask2] -= 10.0   #Add bias to test up pairs
        _, pred = score_test.topk(topk, dim=1)     #test_images*topk
        correct = torch.eq(pair_label, pred)        #test_images*topk
        correct = correct.any(1).float()        #test_images
        unseen_comp_corr, unseen_comp_false= 0, 0
        for i in range(batch_num):
            if pairs[i] in self.te_uc_pairs:
                if correct[i] == 1:
                    unseen_comp_corr += 1    #5615
                else:
                    unseen_comp_false += 1  
        best_unseen_comp_acc = float(unseen_comp_corr/(unseen_comp_corr+unseen_comp_false))
        
        #best unseen pair acc
        score_test = scores.clone().detach()
        score_test[~mask1] -= 10.0 # Add bias to test uc and up pairs
        score_test[~mask2] += 20.0   #Add bias to test up pairs
        _, pred = score_test.topk(topk, dim=1)     #test_images*topk
        correct = torch.eq(pair_label, pred)        #test_images*topk
        correct = correct.any(1).float()        #test_images
        unseen_pair_corr, unseen_pair_false= 0, 0
        for i in range(batch_num):
            if pairs[i] in self.te_up_pairs:
                if correct[i] == 1:
                    unseen_pair_corr += 1     #5948
                else:
                    unseen_pair_false += 1   
        best_unseen_pair_acc = float(unseen_pair_corr/(unseen_pair_corr+unseen_pair_false))
        
        for bias1 in range(10,20):
            score_test = scores.clone().detach()
            score_test[~mask1] += bias1/200.0 # Add bias to test uc and up pairs
            score_test[~mask2] += 4.0/200.0    #Add bias to test up pairs
            for bias2 in range(10,20):
                score_test[~mask2] += 1.0/200.0    #Add bias to test up pairs
                _, pred = score_test.topk(topk, dim=1)     #test_images*topk
                #print(pred)
                #pred = torch.argmax(scores, dim=1)     
                correct = torch.eq(pair_label, pred)        #test_images*topk
                #print(correct.shape)
                correct = correct.any(1).float()        #test_images
                #print(correct.shape)
                #correct = torch.eq(label, pair_label)
                #attr_match = (attr_truth.unsqueeze(1).repeat(1, topk) == _scores[0][:, :topk])
                #obj_match = (obj_truth.unsqueeze(1).repeat(1, topk) == _scores[1][:, :topk])
                #match = (attr_match * obj_match).any(1).float()
                
                seen_corr, unseen_comp_corr, unseen_pair_corr = 0, 0, 0
                seen_false, unseen_comp_false, unseen_pair_false = 0, 0, 0
                
                for i in range(batch_num):
                    #print(pairs[i])
                    if pairs[i] in self.train_pairs:
                        if correct[i] == 1:
                            seen_corr += 1     #10107
                        else:
                            seen_false += 1 
                    elif pairs[i] in self.te_uc_pairs:
                        if correct[i] == 1:
                            unseen_comp_corr += 1    #5615
                        else:
                            unseen_comp_false += 1  
                    elif pairs[i] in self.te_up_pairs:
                        if correct[i] == 1:
                            unseen_pair_corr += 1     #5948
                        else:
                            unseen_pair_false += 1   
                #seen_acc, unseen_comp_acc, unseen_pair_acc = 0.5, 0.5, 0.5
                seen_acc = float(seen_corr/(seen_corr+seen_false))
                unseen_comp_acc = float(unseen_comp_corr/(unseen_comp_corr+unseen_comp_false))
                unseen_pair_acc = float(unseen_pair_corr/(unseen_pair_corr+unseen_pair_false))
                hm = 3/(1/(seen_acc+1e-4)+1/(unseen_comp_acc+1e-4)+1/(unseen_pair_acc+1e-4))
                if hm >= best_hm:
                    best_hm = hm
                    hm_seen = seen_acc
                    hm_unseen_comp = unseen_comp_acc
                    hm_unseen_pair = unseen_pair_acc
                    hm_bias1 = bias1
                    hm_bias2 = bias2
        return hm_bias1, hm_bias2, hm_seen, hm_unseen_comp, hm_unseen_pair, best_hm, best_seen_acc, best_unseen_comp_acc, best_unseen_pair_acc
                
    def generate_predictions(self, scores, bias1 = 0.0, bias2 =0.0, topk = 5): # (Batch, #pairs)
        def get_pred_from_scores(_scores, topk):
            _scores = _scores.float()
            _, pair_pred = _scores.topk(topk, dim = 1) #sort returns indices of k largest values
            pair_pred = pair_pred.contiguous().view(-1)
            attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(-1, topk), \
                self.pairs[pair_pred][:, 1].view(-1, topk)
            return (attr_pred, obj_pred)

        results = {}
        mask = self.seen_mask.repeat(scores.shape[0],1) # Repeat mask along pairs dimension
        scores[~mask] += bias1 # Add bias to test uc and up pairs
        mask = self.seen_comp_mask.repeat(scores.shape[0], 1)
        scores[~mask] += bias2    #Add bias to test up pairs
        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e4
        results.update({'closed': get_pred_from_scores(closed_scores, topk)})
        return results

    def score_manifold_model(self, scores, bias1 = 0.0, bias2=0.0, topk = 5):
       
        scores = {k: v.to('cpu') for k, v in scores.items()}
        scores = torch.stack(
            [scores[(attr,obj)] for attr, obj in self.dset.val_pairs], 1
        ) # (Batch, #pairs)
        orig_scores = scores.clone()
        results = self.generate_predictions(scores, bias1, bias2, topk)
        results['scores'] = orig_scores
        return results

    def score_fast_model(self, scores, bias1 = 0.0, bias2 = 0.0, topk = 5):
        
        results = {}
        mask = self.seen_mask.repeat(scores.shape[0],1) # Repeat mask along pairs dimension
        scores[~mask] += bias1 # Add bias to test pairs
        
        mask = self.seen_comp_mask.repeat(scores.shape[0], 1)
        scores[~mask] += bias2    #Add bias to test up pairs
        
        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e4

        closed_scores = closed_scores.float()
        _, pair_pred = closed_scores.topk(topk, dim = 1) #sort returns indices of k largest values
        pair_pred = pair_pred.contiguous().view(-1)
        attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(-1, topk), \
            self.pairs[pair_pred][:, 1].view(-1, topk)

        results.update({'closed': (attr_pred, obj_pred)})
        return results

    def evaluate_predictions(self, predictions, attr_truth, obj_truth, pair_truth, allpred, topk = 1):
        attr_truth, obj_truth, pair_truth = attr_truth.to('cpu'), obj_truth.to('cpu'), pair_truth.to('cpu')
        pairs = list(zip(list(attr_truth.numpy()), list(obj_truth.numpy())))
        
        seen_ind, unseen_comp_ind, unseen_pair_ind = [], [], []
        for i in range(len(attr_truth)):
            if pairs[i] in self.train_pairs:
                seen_ind.append(i)      #10107
            elif pairs[i] in self.te_uc_pairs:
                unseen_comp_ind.append(i)       #5615
            elif pairs[i] in self.te_up_pairs:
                unseen_pair_ind.append(i)       #5948
        
        seen_ind, unseen_comp_ind, unseen_pair_ind = torch.LongTensor(seen_ind), torch.LongTensor(unseen_comp_ind), torch.LongTensor(unseen_pair_ind)
        
        
        def _process(_scores):
            # Top k pair accuracy
            # Attribute, object and pair
            attr_match = (attr_truth.unsqueeze(1).repeat(1, topk) == _scores[0][:, :topk])
            obj_match = (obj_truth.unsqueeze(1).repeat(1, topk) == _scores[1][:, :topk])
            
            # Match of object pair
            match = (attr_match * obj_match).any(1).float()
            # Match of seen and unseen pairs
            seen_match = match[seen_ind]
            unseen_comp_match = match[unseen_comp_ind]
            unseen_pair_match = match[unseen_pair_ind]
            
            local_score_dict = copy.deepcopy(self.test_pair_dict)
            num = 0
            corr = 0
            for pair_gt, pair_pred in zip(pairs, match):        #pair_gt:(attr_index, obj_index), pair_pred: 0 or 1
                local_score_dict[pair_gt][2] += 1.0
                num+=1
                if int(pair_pred) == 1:
                    local_score_dict[pair_gt][1] += 1.0
                    corr += 1
            #print(local_score_dict)
            #  Now we have hits and totals for classes in evaluation set
            seen_score, unseen_comp_score, unseen_pair_score = [], [], []
            for key, (idx, hits, total) in local_score_dict.items():        #(atr, obj): [pair_index, correct predictions, testing number]
                if total == 0:
                    continue
                score = hits/total
                if bool(self.seen_mask[idx]) == True:
                    seen_score.append(score)        #18
                elif bool(self.seen_comp_mask[idx]) == True:
                    unseen_comp_score.append(score)      #18, length of seen_score+unseen_score==36
                elif bool(self.closed_mask[idx]) == True:
                    unseen_pair_score.append(score)
            
            return match, seen_match, unseen_comp_match, unseen_pair_match, torch.Tensor(seen_score+unseen_comp_score+unseen_pair_score), torch.Tensor(seen_score), torch.Tensor(unseen_comp_score), torch.Tensor(unseen_pair_score)

        def _add_to_dict(_scores, type_name, stats):
            base = ['_match', '_seen_match', '_unseen_comp_match', '_unseen_pair_match', '_ca', '_seen_ca', '_unseen_comp_ca', '_unseen_pair_ca']
            for val, name in zip(_scores, base):
                stats[type_name + name] = val
        stats = {}
        closed_scores = _process(predictions['closed'])
        _add_to_dict(closed_scores, 'closed', stats)

        scores = predictions['scores']
        magic_binsize = 7
        seen_accuracy, unseen_comp_accuracy, unseen_pair_accuracy, match_accuracy, ca_accuracy, seen_ca_accuracy, unseen_comp_ca_accuracy, unseen_pair_ca_accuracy = [], [], [], [], [], [], [], []
        base_scores = {k: v.to('cpu') for k, v in allpred.items()}
        base_scores = torch.stack(
            [allpred[(attr,obj)] for attr, obj in self.dset.val_pairs], 1
        ) # (Batch, #pairs)

        biaslist1, biaslist2 = range(4,9), range(4,9)
        for bias1 in biaslist1:
            for bias2 in biaslist2:
                bias1, bias2 = bias1/5.0, bias2/5.0
                scores = base_scores.clone()
                results = self.score_fast_model(scores, bias1 = bias1, bias2 = bias2, topk = topk)
                results = results['closed'] # we only need biased
                results = _process(results)
                
                seen_match = float(results[1].mean()+1e-4)
                unseen_comp_match = float(results[2].mean()+1e-4)
                unseen_pair_match = float(results[3].mean()+1e-4)
                
                match = float(results[0].mean()+1e-4)
                ca = float(results[4].mean()+1e-4)
                seen_ca = float(results[5].mean()+1e-4)
                unseen_comp_ca = float(results[6].mean()+1e-4)
                unseen_pair_ca = float(results[7].mean()+1e-4)
                
                seen_accuracy.append(seen_match)
                unseen_comp_accuracy.append(unseen_comp_match)
                unseen_pair_accuracy.append(unseen_pair_match)
                
                match_accuracy.append(match)
                ca_accuracy.append(ca)
                seen_ca_accuracy.append(seen_ca)
                unseen_comp_ca_accuracy.append(unseen_comp_ca)
                unseen_pair_ca_accuracy.append(unseen_pair_ca)

        seen_accuracy, unseen_comp_accuracy, unseen_pair_accuracy = np.array(seen_accuracy), np.array(unseen_comp_accuracy), np.array(unseen_pair_accuracy)
        mean_accuracy = (seen_accuracy + unseen_comp_accuracy + unseen_pair_accuracy)/3.
        idx = np.argmax(mean_accuracy)
        
        bias1 = biaslist1[idx//magic_binsize]
        bias2 = biaslist2[idx%magic_binsize]
        for key in stats:
            stats[key] = float(stats[key].mean())
        
        stats['bias1'] = float(bias1)
        stats['bias2'] = float(bias2)
        stats['best_mean'] = np.max(mean_accuracy)
        stats['best_unseen_comp'] = np.max(unseen_comp_accuracy)
        stats['best_unseen_pair'] = np.max(unseen_pair_accuracy)
        stats['best_seen'] = np.max(seen_accuracy)
        
        stats['hm_unseen_comp'] = unseen_comp_accuracy[idx]
        stats['hm_unseen_pair'] = unseen_pair_accuracy[idx]
        stats['hm_seen'] = seen_accuracy[idx]
        
        stats['closed_match'] = match_accuracy[idx]
        stats['closed_ca'] = ca_accuracy[idx]
        stats['closed_seen_ca'] = seen_ca_accuracy[idx]
        stats['closed_unseen_comp_ca'] = unseen_comp_ca_accuracy[idx]
        stats['closed_unseen_pair_ca'] = unseen_pair_ca_accuracy[idx]
        return stats