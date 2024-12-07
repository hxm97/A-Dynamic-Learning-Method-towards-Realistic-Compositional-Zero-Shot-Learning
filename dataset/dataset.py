import os
import shutil

def parse_pairs(pair_list):
    with open(pair_list, 'r') as f:
        pairs = f.read().strip().split('\n')
        pairs = [t.split() for t in pairs]
        pairs = list(map(tuple, pairs))
    attrs, objs = zip(*pairs)
    return attrs, objs, pairs

path='/gdata/huxm/rzsl_data/mitstates'
path1='/ghome/huxm/rzsl/data/mitstates_img_split'
path2 = '/ghome/huxm/rzsl/data/mitstates_pair_split'


tr_attrs, tr_objs, tr_pairs = parse_pairs(os.path.join(path2,  'seen_pairs.txt'))
vl_uc_attrs, vl_uc_objs, vl_uc_pairs = parse_pairs(os.path.join(path2, 'val_unseen_comps.txt'))
vl_up_attrs, vl_up_objs, vl_up_pairs = parse_pairs(os.path.join(path2, 'val_unseen_pairs.txt'))
te_uc_attrs, te_uc_objs, te_uc_pairs = parse_pairs(os.path.join(path2, 'test_unseen_comps.txt'))
te_up_attrs, te_up_objs, te_up_pairs = parse_pairs(os.path.join(path2, 'test_unseen_pairs.txt'))
'''
train_file = os.path.join(path1,  'train_imgs.txt')
with open(train_file, 'r') as f:
    train_image_label = [l.strip() for l in f.readlines()]
val_file = os.path.join(path1,  'val_imgs.txt')
with open(val_file, 'r') as f:
    val_image_label = [l.strip() for l in f.readlines()]
test_file = os.path.join(path1,  'test_imgs.txt')
with open(test_file, 'r') as f:
    test_image_label = [l.strip() for l in f.readlines()]
'''
to_be_fix = '/ghome/huxm/rzsl/to_be_fix.txt'
with open(to_be_fix, 'r') as f:
    fix_file = [l.strip() for l in f.readlines()]
for lines in fix_file:
    items = lines.split()
    img_name = items[0]
    #loc = img_name.find('/')
    #img_name = img_name[loc+1:]
    ori_att = items[1]
    ori_obj = items[2]
    aim_att = items[3]
    aim_obj = items[3]

    ori_path_1 = os.path.join(path, 'art/images', ori_att+'_'+ori_obj, img_name)
    aim_path_1 = os.path.join(path, 'art/images', aim_att+'_'+aim_obj, img_name)
    
    ori_path_2 = os.path.join(path, 'cartoon/images', ori_att+'_'+ori_obj, img_name)
    aim_path_2 = os.path.join(path, 'cartoon/images', aim_att+'_'+aim_obj, img_name)
    
    ori_path_3 = os.path.join(path, 'photo/images', ori_att+'_'+ori_obj, img_name)
    aim_path_3 = os.path.join(path, 'photo/images', aim_att+'_'+aim_obj, img_name)
    
    shutil.move(ori_path_1, aim_path_1)
    shutil.move(ori_path_2, aim_path_2)
    shutil.move(ori_path_3, aim_path_3)