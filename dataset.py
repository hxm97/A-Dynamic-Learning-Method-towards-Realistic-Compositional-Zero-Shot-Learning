import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tdata
import torchvision.models as tmodels
import torchvision.transforms as transforms
import tqdm

from PIL import Image


class ImageLoader:
    def __init__(self, root, phase, train_domain_1, train_domain_2, test_domain):
        self.img_dir = root
        self.phase = phase
        self.train_domain_1 = train_domain_1
        self.train_domain_2 = train_domain_2
        self.test_domain = test_domain

    def __call__(self, img):
        if self.phase == 'train' :
            file_1 = os.path.join(self.img_dir, self.train_domain_1, 'images', img)
            img_1 = Image.open(file_1).convert('RGB')
            file_2 = os.path.join(self.img_dir, self.train_domain_2, 'images', img)
            img_2 = Image.open(file_2).convert('RGB')
            return img_1, img_2
        else:
            file_1 = os.path.join(self.img_dir, self.train_domain_1, 'images', img)
            img_1 = Image.open(file_1).convert('RGB')
            file_2 = os.path.join(self.img_dir, self.train_domain_2, 'images', img)
            img_2 = Image.open(file_2).convert('RGB')
            file_3 = os.path.join(self.img_dir, self.test_domain, 'images', img)
            img_3 = Image.open(file_3).convert('RGB')
            return img_1, img_2, img_3


def imagenet_transform(phase):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if phase == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
       
    elif phase == 'test' or phase == 'val':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    return transform


class CompositionDataset(tdata.Dataset):
    def __init__(
        self,
        phase,
        open_world=False,
        cfg=None
    ):
        self.phase = phase
        self.cfg = cfg
        self.open_world = open_world
        self.transform = imagenet_transform(phase)
        self.loader = ImageLoader(cfg.root_dir, self.phase, cfg.train_domain_1, cfg.train_domain_2, cfg.test_domain)
        self.all_attrs, self.all_objs, self.all_pairs, self.tr_attrs, self.tr_objs, self.tr_pairs, self.vl_uc_pairs, self.vl_up_pairs, self.te_uc_pairs, self.te_up_pairs = self.parse_split()
        self.val_pairs = self.tr_pairs+self.te_uc_pairs+self.te_up_pairs
        
        att_class_map, obj_class_map = {}, {}
        att_file = self.cfg.pair_split+'all_attrs.txt'
        with open(att_file, 'r') as f:
            att_names = [l.strip() for l in f.readlines()]
        for i,l in enumerate(att_names):
            items = l.split()
            att_class_map[items[-1]] = i
        
        obj_file = self.cfg.pair_split + 'all_objs.txt'
        with open(obj_file, 'r') as f:
            obj_names = [l.strip() for l in f.readlines()]
        for i,l in enumerate(obj_names):
            items = l.split()
            obj_class_map[items[-1]] = i
        
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.all_objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.all_attrs)}
        self.train_pair2idx = {pair: idx for idx, pair in enumerate(self.tr_pairs)}
        self.val_pair2idx = {pair: idx for idx, pair in enumerate(self.val_pairs)}
        
        self.train_data, self.train_atts, self.train_objs, self.val_data, self.val_atts, self.val_objs, self.test_data, self.test_atts, self.test_objs = self.get_split_info()
        
        print('# train pairs: %d | # val unseen comps: %d | # val unseen pairs: %d | # test unseen comps: %d | # test unseen pairs: %d' % (len(self.tr_pairs), len(self.vl_uc_pairs), len(self.vl_up_pairs), len(self.te_uc_pairs), len(self.te_up_pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %(len(self.train_data), len(self.val_data), len(self.test_data)))
        
        if self.phase == 'train':
            self.data = self.train_data
            self.atts = self.train_atts
            self.objs = self.train_objs
            if cfg.subset:
                ind = np.arange(len(self.data))
                ind = ind[::len(ind) // 1000]
                self.data = [self.data[i] for i in ind]
        elif self.phase == 'val':
            self.data = self.val_data
            self.atts = self.val_atts
            self.objs = self.val_objs
        else:
            self.data = self.test_data
            self.atts = self.test_atts
            self.objs = self.test_objs
        
        
        
    def get_split_info(self):
        train_data, val_data, test_data = [], [], []
        train_atts, val_atts, test_atts = [], [], []
        train_objs, val_objs, test_objs = [], [], []
        
        train_file = os.path.join(self.cfg.img_split,  'train_imgs.txt')
        with open(train_file, 'r') as f:
            train_image_label = [l.strip() for l in f.readlines()]
        for lines in train_image_label:
            items = lines.split()
            train_data.append(items[0])
            train_atts.append(items[1])
            train_objs.append(items[2])
        
        val_file = os.path.join(self.cfg.img_split,  'val_imgs.txt')
        with open(val_file, 'r') as f:
            val_image_label = [l.strip() for l in f.readlines()]
        for lines in val_image_label:
            items = lines.split()
            val_data.append(items[0])
            val_atts.append(items[1])
            val_objs.append(items[2])
            
        test_file = os.path.join(self.cfg.img_split,  'test_imgs.txt')
        with open(test_file, 'r') as f:
            test_image_label = [l.strip() for l in f.readlines()]
        for lines in test_image_label:
            items = lines.split()
            test_data.append(items[0])
            test_atts.append(items[1])
            test_objs.append(items[2])
        return train_data, train_atts, train_objs, val_data, val_atts, val_objs, test_data, test_atts, test_objs

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs
        
        tr_attrs, tr_objs, tr_pairs = parse_pairs(os.path.join(self.cfg.pair_split,  'seen_pairs.txt'))
        vl_uc_attrs, vl_uc_objs, vl_uc_pairs = parse_pairs(os.path.join(self.cfg.pair_split, 'val_unseen_comps.txt'))
        vl_up_attrs, vl_up_objs, vl_up_pairs = parse_pairs(os.path.join(self.cfg.pair_split, 'val_unseen_pairs.txt'))
        te_uc_attrs, te_uc_objs, te_uc_pairs = parse_pairs(os.path.join(self.cfg.pair_split, 'test_unseen_comps.txt'))
        te_up_attrs, te_up_objs, te_up_pairs = parse_pairs(os.path.join(self.cfg.pair_split, 'test_unseen_pairs.txt'))
        #all_attrs, all_objs, all_pairs = parse_pairs(os.path.join(self.cfg.pair_split, 'all_pairs.txt'))
        
        all_attrs, all_objs = sorted(list(set(tr_attrs + vl_uc_attrs + vl_up_attrs + te_uc_attrs + te_up_attrs))), sorted(list(set(tr_objs + vl_uc_objs + vl_up_objs + te_uc_objs + te_up_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_uc_pairs + vl_up_pairs + te_uc_pairs +te_up_pairs)))
        
        return all_attrs, all_objs, all_pairs, tr_attrs, tr_objs, tr_pairs, vl_uc_pairs, vl_up_pairs, te_uc_pairs, te_up_pairs

    def __getitem__(self, index):
        #image, attr, obj = self.data[index]
        image = self.data[index]
        attr = self.atts[index]
        obj = self.objs[index]
        
        if self.phase == 'train':
            img1, img2 = self.loader(image)
            img1, img2 = self.transform(img1), self.transform(img2)
            data = {'img1': img1, 'img2': img2, 'attr': self.attr2idx[attr], 'obj': self.obj2idx[obj], 'pair': self.train_pair2idx[(attr,obj)], 'attr_name': attr,'obj_name': obj, 'img_name': image}
        else:
            img1, img2, img3 = self.loader(image)
            img1, img2, img3 = self.transform(img1), self.transform(img2), self.transform(img3)
            data = {'img1': img1, 'img2': img2, 'img3': img3, 'attr': self.attr2idx[attr], 'obj': self.obj2idx[obj], 'pair': self.val_pair2idx[(attr,obj)], 'img_name': image}
        return data

    def __len__(self):
        return len(self.data)

