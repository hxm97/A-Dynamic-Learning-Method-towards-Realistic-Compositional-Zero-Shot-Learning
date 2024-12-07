import numpy as np
import os
import random
from os.path import join as ospj
import torch
import collections
import random

dataset = 'mitstates'    #cgqa|mitstates
root = '/gdata/huxm/rzsl_data/'
img_root = root + dataset + '/photo/images'
split_root = '/ghome/huxm/rzsl/data/' + dataset + '_pair_split'
labels = 'metadata_compositional-split-natural.t7'

data = torch.load(ospj(split_root, labels))
train_data, val_data, test_data = [], [], []

def parse_pairs(pair_list):
    with open(pair_list, 'r') as f:
        pairs = f.read().strip().split('\n')
        pairs = [t.split() for t in pairs]
        #pairs = list(map(tuple, pairs))
    #attrs, objs = zip(*pairs)
    return pairs

seen_pairs = parse_pairs(ospj(split_root, 'seen_pairs.txt'))
test_unseen_comps = parse_pairs(ospj(split_root, 'test_unseen_comps.txt'))
test_unseen_pairs = parse_pairs(ospj(split_root, 'test_unseen_pairs.txt'))
val_unseen_comps = parse_pairs(ospj(split_root, 'val_unseen_comps.txt'))
val_unseen_pairs = parse_pairs(ospj(split_root, 'val_unseen_pairs.txt'))

f1 = open(ospj('/ghome/huxm/rzsl/data/'+dataset+'_img_split', 'train_imgs.txt'), 'w')
f2 = open(ospj('/ghome/huxm/rzsl/data/'+dataset+'_img_split', 'val_imgs.txt'), 'w')
f3 = open(ospj('/ghome/huxm/rzsl/data/'+dataset+'_img_split', 'test_imgs.txt'), 'w')

num_train_img = 0
num_val_seen_img = 0
num_val_unseen_comp_img = 0
num_val_unseen_pair_img = 0
num_test_seen_img = 0
num_test_unseen_comp_img = 0
num_test_unseen_pair_img = 0

#count = collections.defaultdict(int)
count = dict()
for pair in seen_pairs:
    pair_name = pair[0]+'_'+pair[1]
    count[pair_name] = 0

dict = seen_pairs

folders = os.listdir('/gdata/huxm/rzsl_data/mitstates/art/images')
for pair_name in folders:
    loc = pair_name.find('_')
    attr = pair_name[:loc]
    obj = pair_name[loc+1:]
    pair = [attr, obj]
    file_path = os.path.join('/gdata/huxm/rzsl_data/mitstates/art/images', pair_name)
    imgs = os.listdir(file_path)
    
    for image in imgs:
        if pair in seen_pairs:
            count[pair_name] += 1
        elif pair in test_unseen_comps:
            f3.write('%s/%s %s %s\n'%(pair_name, image, attr, obj))
            num_test_unseen_comp_img += 1
        elif pair in test_unseen_pairs:
            f3.write('%s/%s %s %s\n'%(pair_name, image, attr, obj))
            num_test_unseen_pair_img += 1
        elif pair in val_unseen_comps:
            f2.write('%s/%s %s %s\n'%(pair_name, image, attr, obj))
            num_val_unseen_comp_img += 1
        elif pair in val_unseen_pairs:
            f2.write('%s/%s %s %s\n'%(pair_name, image, attr, obj))
            num_val_unseen_pair_img += 1
    
    #for instance in data:
        #image, attr, obj = instance['image'], instance['attr'], instance['obj']
        #pair = [attr, obj]
        #pair_name = pair[0]+'_'+pair[1]
    
        
    
#for instance in data:
    
    #image, attr, obj = instance['image'], instance['attr'], instance['obj']
    #pair = [attr, obj]
    #pair_name = pair[0]+'_'+pair[1]
for pair_name in folders:
    loc = pair_name.find('_')
    attr = pair_name[:loc]
    obj = pair_name[loc+1:]
    pair = [attr, obj]
    file_path = os.path.join('/gdata/huxm/rzsl_data/mitstates/art/images', pair_name)
    imgs = os.listdir(file_path)
    for image in imgs:
        if pair in seen_pairs:
            if count[pair_name] > 5:
                rand_num_1 = random.random()
                if rand_num_1 < 0.4:
                    count[pair_name] -= 1
                    rand_num_2 = random.random()
                    if rand_num_2 < 0.3:
                        f2.write('%s/%s %s %s\n'%(pair_name, image, attr, obj))
                        num_val_seen_img += 1
                    else:
                        f3.write('%s/%s %s %s\n'%(pair_name, image, attr, obj))
                        num_test_seen_img += 1
                else:
                    f1.write('%s/%s %s %s\n'%(pair_name, image, attr, obj))
                    num_train_img += 1
            else:
                f1.write('%s/%s %s %s\n'%(pair_name, image, attr, obj))
                num_train_img += 1

f1.close()
f2.close()
f3.close()

print(num_train_img)
print(num_val_seen_img)
print(num_val_unseen_comp_img)
print(num_val_unseen_pair_img)
print(num_test_seen_img)
print(num_test_unseen_comp_img)
print(num_test_unseen_pair_img)