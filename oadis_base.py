import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import Backbone
from basic_layers import MLP
from model import convert_weights
from hubconf import dino_vits16, dino_vitb16, dino_vitb8, dino_vits8, dino_resnet50
from  torch.cuda.amp import autocast
import copy
device = 'cuda:0'

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class OADIS(nn.Module):
    """Object-Attribute Compositional Learning from Image Pair.
    """
    def __init__(self, dset, cfg):
        super(OADIS, self).__init__()
        self.cfg = cfg

        self.num_attrs = len(dset.all_attrs)
        self.num_objs = len(dset.all_objs)
        self.pair2idx = dset.val_pair2idx
        self.attr2idx = dset.attr2idx
        self.obj2idx = dset.obj2idx
        
        self.tr_pairs = dset.tr_pairs
        
        tr_attrs, tr_objs = zip(*dset.tr_pairs)
        tr_attrs = [dset.attr2idx[attr] for attr in tr_attrs]
        tr_objs = [dset.obj2idx[obj] for obj in tr_objs]
        self.tr_attrs = torch.LongTensor(tr_attrs).to(device)
        self.tr_objs = torch.LongTensor(tr_objs).to(device)
        
        self.val_pairs = dset.tr_pairs+dset.te_uc_pairs+dset.te_up_pairs
        val_attrs, val_objs = zip(*self.val_pairs)
        val_attrs = [dset.attr2idx[attr] for attr in val_attrs]
        val_objs = [dset.obj2idx[obj] for obj in val_objs]
        self.val_attrs = torch.LongTensor(val_attrs).to(device)
        self.val_objs = torch.LongTensor(val_objs).to(device)
        
        if cfg.feat_extractor == 'resnet18':
            feat_dim = 512
            self.feat_extractor = Backbone(cfg)
        elif cfg.feat_extractor == 'resnet50':
            feat_dim = 2048
            self.feat_extractor = Backbone(cfg)
        elif cfg.feat_extractor == 'resnet101':
            feat_dim = 2048
            self.feat_extractor = Backbone(cfg)
        elif cfg.feat_extractor == 'vit-b_16':
            feat_dim = 384
            self.feat_extractor = dino_vitb16(model_root = cfg.model_root)
        elif cfg.feat_extractor == 'vit-b_8':
            feat_dim = 768
            self.feat_extractor = dino_vitb8(model_root = cfg.model_root)
        elif cfg.feat_extractor == 'vit-s_16':
            feat_dim = 384
            self.feat_extractor = dino_vits16(model_root = cfg.model_root)
        elif cfg.feat_extractor == 'vit-s_8':
            feat_dim = 768
            self.feat_extractor = dino_vits8(model_root = cfg.model_root)
        
        if cfg.use_fp16 == True:
            convert_weights(self.feat_extractor)
        
        self.dy_conv = Dynamic_conv2d(feat_dim, feat_dim, kernel_size=1)
        self.batchnorm = nn.BatchNorm2d(feat_dim)
        self.relu = nn.ReLU()
        '''
        img_emb_modules = [
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU()
        ]
        '''
        
        if '+' in cfg.wordembs:
            self.emb_dim = cfg.emb_dim*2
        else:
            self.emb_dim = cfg.emb_dim
        
        self.zsl_prt_dec = _get_clones(PrtAttLayer(dim=self.emb_dim, nhead=6), 2)
        self.concept_mod = nn.Linear(feat_dim, self.emb_dim)
        
        self.conv = nn.Conv2d(feat_dim, self.emb_dim, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        
        if cfg.wordembs == 'prompt':
            self.all_text_feats = torch.load(cfg.word_vector+'all_prompt_pair.pt')
            self.train_text_feats = torch.load(cfg.word_vector+'train_prompt_pair.pt')
            self.emb_dim = 1024
        
        else:
            self._setup_word_composer(dset, cfg)
        
        #self.img_embedder = nn.Sequential(*img_emb_modules)
        self.img_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_final = nn.Linear(feat_dim, self.emb_dim)
        self.classifier = CosineClassifier(dataset_name = cfg.dataset_name, temp=cfg.cosine_cls_temp)
        
    def _setup_word_composer(self, dset, cfg):
        self.attr_wordemb = torch.from_numpy(np.loadtxt(os.path.join(cfg.word_vector, cfg.wordembs+'_attr')))
        self.obj_wordemb = torch.from_numpy(np.loadtxt(os.path.join(cfg.word_vector, cfg.wordembs+'_obj')))
        self.word_dim = self.attr_wordemb.shape[1]
        
        self.attr_embedder = nn.Embedding(self.num_attrs, self.word_dim)
        self.obj_embedder = nn.Embedding(self.num_objs, self.word_dim)
        self.attr_embedder.weight.data.copy_(self.attr_wordemb)
        self.obj_embedder.weight.data.copy_(self.obj_wordemb)
        if cfg.use_fp16 == True:
            self.attr_embedder.weight.data = self.attr_embedder.weight.data.half()
            self.obj_embedder.weight.data = self.obj_embedder.weight.data.half()
        
        if '+' in cfg.wordembs:
            emb_dim = cfg.emb_dim*2
        else:
            emb_dim = cfg.emb_dim
        
        self.wordemb_compose = cfg.wordemb_compose
        if cfg.wordemb_compose == 'linear':
            self.compose = nn.Sequential(
                nn.Dropout(cfg.wordemb_compose_dropout),
                nn.Linear(self.word_dim*2, self.emb_dim)
            )
        elif cfg.wordemb_compose == 'obj-conditioned':
            self.object_code = nn.Sequential(
                nn.Linear(self.word_dim, 600),
                nn.ReLU(True)
            )
            self.attribute_code = nn.Sequential(
                nn.Linear(self.word_dim, 600),
                nn.ReLU(True)
            )
            self.attribute_code_fc = nn.Sequential(
                nn.Linear(600, 600),
                nn.ReLU(True),
            )
            self.compose = MLP(
                self.word_dim + 600, 600, self.emb_dim, 2, batchnorm=False,
                drop_input=cfg.wordemb_compose_dropout
            )
        elif cfg.wordemb_compose == 'obj-conditioned-vaw':
            self.object_code = nn.Sequential(
                nn.Linear(self.word_dim, 300),
                nn.ReLU(True)
            )
            self.attribute_code = nn.Sequential(
                nn.Linear(self.word_dim, 300),
                nn.ReLU(True)
            )
            self.compose = nn.Sequential(
                nn.Dropout(cfg.wordemb_compose_dropout),
                nn.Linear(self.word_dim + 300, 300)
            )


    def compose_word_embeddings(self, mode='train'):
        if mode == 'train':
            concepts = []
            for pairs in self.tr_pairs:
                attrs, objs = pairs[0], pairs[1]
                attr_id, obj_id = self.attr2idx[attrs], self.obj2idx[objs]
                attr_emb, obj_emb = self.attr_wordemb[attr_id, :].to(device), self.obj_wordemb[obj_id, :].to(device)
                if self.cfg.use_fp16 == True:
                    attr_emb, obj_emb = attr_emb.half().to(device), obj_emb.half().to(device)
                attr_emb, obj_emb = attr_emb.float(), obj_emb.float()  #.cuda()
                
                if 'obj-conditioned' in self.cfg.wordemb_compose:
                    object_c = self.object_code(obj_emb) # [n_pairs, 1024].
                    attribute_c = self.attribute_code(attr_emb) # [n_pairs, 1024].
                    if 'vaw' in self.cfg.wordemb_compose:
                        attribute_c = object_c * attribute_c
                    else:
                        attribute_c = self.attribute_code_fc(object_c * attribute_c)
                    concept_emb = torch.cat((obj_emb, attribute_c), dim=-1) # [n_pairs, word_dim + 1024].
                else:
                    concept_emb = torch.cat((obj_emb, attr_emb), dim=-1)
                concept_emb = self.compose(concept_emb) # [n_pairs, emb_dim].
                concept_emb = concept_emb.unsqueeze(0)
                concepts.append(concept_emb)
                
        elif mode == 'val':
            concepts = []
            for pairs in self.val_pairs:
                attrs, objs = pairs[0], pairs[1]
                attr_id, obj_id = self.attr2idx[attrs], self.obj2idx[objs]
                attr_emb, obj_emb = self.attr_wordemb[attr_id, :].to(device), self.obj_wordemb[obj_id, :].to(device)
                if self.cfg.use_fp16 == True:
                    attr_emb, obj_emb = attr_emb.half().to(device), obj_emb.half().to(device)
                attr_emb, obj_emb = attr_emb.float(), obj_emb.float()  #.cuda()
                if 'obj-conditioned' in self.cfg.wordemb_compose:
                    obj_emb = obj_emb.float()  #.cuda()
                    attr_emb = attr_emb.float()  #.cuda()
                    object_c = self.object_code(obj_emb) # [n_pairs, 1024].
                    attribute_c = self.attribute_code(attr_emb) # [n_pairs, 1024].
                    if 'vaw' in self.cfg.wordemb_compose:
                        attribute_c = object_c * attribute_c
                    else:
                        attribute_c = self.attribute_code_fc(object_c * attribute_c)
                    concept_emb = torch.cat((obj_emb, attribute_c), dim=-1) # [n_pairs, word_dim + 1024].
                else:
                    concept_emb = torch.cat((obj_emb, attr_emb), dim=-1)
                concept_emb = self.compose(concept_emb) # [n_pairs, emb_dim].
                concept_emb = concept_emb.unsqueeze(0)
                concepts.append(concept_emb)
        
        concepts = torch.cat(concepts, dim=0)
        return concepts


    def train_forward(self, batch):
        img1 = batch['img1']
        img2 = batch['img2']

        attr_labels = batch['attr']
        obj_labels = batch['obj']
        pair_labels = batch['pair']
        bs = img1.shape[0]
        
        if self.cfg.wordembs == 'prompt':
            concept = self.train_text_feats
            concept = concept.half().to(device)
        else:
            concept = self.compose_word_embeddings(mode='train') # (n_pairs, emb_dim)
        
        concept = concept.unsqueeze(1).repeat(1, bs, 1).cuda()      #83*batch_size*300
        concept_res_1, concept_res_2 = concept.clone().detach(), concept.clone().detach()
        
        with autocast():
            img1, style_stat_1, style_0_1, style_1_1, style_2_1, style_3_1= self.feat_extractor(img1)
            img2, style_stat_2, style_0_2, style_1_2, style_2_2, style_3_2= self.feat_extractor(img2)

        if 'vit' in self.cfg.feat_extractor:
            img1 = img1[:,0,:].squeeze(1)
            img2 = img2[:,0,:].squeeze(1)
        
        if self.cfg.use_fp16 == False:
            img1, style_stat_1, style_0_1, style_1_1, style_2_1, style_3_1 = img1.type(torch.float32).cuda(), style_stat_1.type(torch.float32).cuda(), style_0_1.type(torch.float32).cuda(), style_1_1.type(torch.float32).cuda(), style_2_1.type(torch.float32).cuda(), style_3_1.type(torch.float32).cuda()
            img2, style_stat_2, style_0_2, style_1_2, style_2_2, style_3_2 = img2.type(torch.float32).cuda(), style_stat_2.type(torch.float32).cuda(), style_0_2.type(torch.float32).cuda(), style_1_2.type(torch.float32).cuda(), style_2_2.type(torch.float32).cuda(), style_3_2.type(torch.float32).cuda()
        
        style_0_1, style_1_1, style_2_1, style_3_1 = F.avg_pool2d(self.conv1(style_0_1), kernel_size=8), F.avg_pool2d(self.conv2(style_1_1),kernel_size=8), F.avg_pool2d(self.conv3(style_2_1), kernel_size=4), F.avg_pool2d(self.conv4(style_3_1), kernel_size=2)
        style_0_2, style_1_2, style_2_2, style_3_2 = F.avg_pool2d(self.conv1(style_0_2), kernel_size=8), F.avg_pool2d(self.conv2(style_1_2),kernel_size=8), F.avg_pool2d(self.conv3(style_2_2), kernel_size=4), F.avg_pool2d(self.conv4(style_3_2), kernel_size=2)
        style1 = torch.cat((style_0_1, style_1_1, style_2_1, style_3_1), dim=1)
        style2 = torch.cat((style_0_2, style_1_2, style_2_2, style_3_2), dim=1)
        
        h, w = img1.shape[2:]
        
        img_mod_1 = self.conv(style1)
        img_mod_1 = img_mod_1.flatten(2).permute(2,0,1)         #49*batch_size*300
        img_mod_2 = self.conv(style2)
        img_mod_2 = img_mod_2.flatten(2).permute(2,0,1)         #49*batch_size*300
        for dec in self.zsl_prt_dec:
            concept_res_1 = dec(concept_res_1, img_mod_1)     #83*batch_size*300
            concept_res_2 = dec(concept_res_2, img_mod_2)     #83*batch_size*300
        concept_1 = concept + concept_res_1
        concept_1 = concept_1.permute(1,0,2)
        concept_2 = concept + concept_res_2
        concept_2 = concept_2.permute(1,0,2)
        
        img1 = self.dy_conv(img1, style1)
        img1 = self.relu(self.batchnorm(img1))
        img1 = self.img_avg_pool(img1.view(bs, -1, h, w)).squeeze()
        img2 = self.dy_conv(img2, style2)
        img2 = self.relu(self.batchnorm(img2))
        img2 = self.img_avg_pool(img2.view(bs, -1, h, w)).squeeze()
        
        img1 = self.img_final(img1)     #batch_size*300
        img2 = self.img_final(img2)
        
        
        pred1 = self.classifier(img1, concept_1)      #batch_size*1398
        pred2 = self.classifier(img2, concept_2)
        
        pair_loss_1 = F.cross_entropy(pred1, pair_labels)       #batch_size
        pair_loss_2 = F.cross_entropy(pred2, pair_labels)
        loss = pair_loss_1 + pair_loss_2
        #print('Training correct: %d Loss %.2f' %(correct_sum1, loss))
        out = {
            'loss_total': loss
        }
        return out

    def val_forward(self, batch):
        img1, img2, img3 = batch['img1'], batch['img2'], batch['img3']
        bs = img1.shape[0]
        attr_labels = batch['attr']
        obj_labels = batch['obj']
        pair_labels = batch['pair']
        if self.cfg.wordembs == 'prompt':
            concept = self.all_text_feats
            concept = concept.half().to(device)
        else:
            concept = self.compose_word_embeddings(mode='val') # [n_pairs, emb_dim].
        
        concept = concept.unsqueeze(1).repeat(1, bs, 1).cuda()      #83*batch_size*300
        concept_res_1, concept_res_2, concept_res_3 = concept.clone().detach(), concept.clone().detach(), concept.clone().detach()
        
        with autocast():
            img1, style_stat_1, style_0_1, style_1_1, style_2_1, style_3_1  = self.feat_extractor(img1)
            img2, style_stat_2, style_0_2, style_1_2, style_2_2, style_3_2  = self.feat_extractor(img2)
            img3, style_stat_3, style_0_3, style_1_3, style_2_3, style_3_3  = self.feat_extractor(img3)
            #img1, img2, img3 = self.feat_extractor(img1), self.feat_extractor(img2), self.feat_extractor(img3)
        if 'vit' in self.cfg.feat_extractor:
            img1 = img1[:,0,:].squeeze(1)
            img2 = img2[:,0,:].squeeze(1)
            img3 = img3[:,0,:].squeeze(1)
        
        if self.cfg.use_fp16 == False:
            img1, style_stat_1, style_0_1, style_1_1, style_2_1, style_3_1 = img1.type(torch.float32).cuda(), style_stat_1.type(torch.float32).cuda(), style_0_1.type(torch.float32).cuda(), style_1_1.type(torch.float32).cuda(), style_2_1.type(torch.float32).cuda(), style_3_1.type(torch.float32).cuda()
            img2, style_stat_2, style_0_2, style_1_2, style_2_2, style_3_2 = img2.type(torch.float32).cuda(), style_stat_2.type(torch.float32).cuda(), style_0_2.type(torch.float32).cuda(), style_1_2.type(torch.float32).cuda(), style_2_2.type(torch.float32).cuda(), style_3_2.type(torch.float32).cuda()
            img3, style_stat_3, style_0_3, style_1_3, style_2_3, style_3_3 = img3.type(torch.float32).cuda(), style_stat_3.type(torch.float32).cuda(), style_0_3.type(torch.float32).cuda(), style_1_3.type(torch.float32).cuda(), style_2_3.type(torch.float32).cuda(), style_3_3.type(torch.float32).cuda()
        
        h, w = img1.shape[2:]
        style_0_1, style_1_1, style_2_1, style_3_1 = F.avg_pool2d(self.conv1(style_0_1), kernel_size=8), F.avg_pool2d(self.conv2(style_1_1),kernel_size=8), F.avg_pool2d(self.conv3(style_2_1), kernel_size=4), F.avg_pool2d(self.conv4(style_3_1), kernel_size=2)
        style_0_2, style_1_2, style_2_2, style_3_2 = F.avg_pool2d(self.conv1(style_0_2), kernel_size=8), F.avg_pool2d(self.conv2(style_1_2),kernel_size=8), F.avg_pool2d(self.conv3(style_2_2), kernel_size=4), F.avg_pool2d(self.conv4(style_3_2), kernel_size=2)
        style_0_3, style_1_3, style_2_3, style_3_3 = F.avg_pool2d(self.conv1(style_0_3), kernel_size=8), F.avg_pool2d(self.conv2(style_1_3),kernel_size=8), F.avg_pool2d(self.conv3(style_2_3), kernel_size=4), F.avg_pool2d(self.conv4(style_3_3), kernel_size=2)
        
        style1 = torch.cat((style_0_1, style_1_1, style_2_1, style_3_1), dim=1)
        style2 = torch.cat((style_0_2, style_1_2, style_2_2, style_3_2), dim=1)
        style3 = torch.cat((style_0_3, style_1_3, style_2_3, style_3_3), dim=1)
        
        img_mod_1 = self.conv(style1)
        img_mod_2 = self.conv(style2)
        img_mod_3 = self.conv(style3)
        img_mod_1 = img_mod_1.flatten(2).permute(2,0,1)         #49*batch_size*300
        img_mod_2 = img_mod_2.flatten(2).permute(2,0,1)         #49*batch_size*300
        img_mod_3 = img_mod_3.flatten(2).permute(2,0,1)         #49*batch_size*300
        
        for dec in self.zsl_prt_dec:
            concept_res_1 = dec(concept_res_1, img_mod_1)     #83*batch_size*300
            concept_res_2 = dec(concept_res_2, img_mod_2)     #83*batch_size*300
            concept_res_3 = dec(concept_res_3, img_mod_3)     #83*batch_size*300
        concept_1 = concept + concept_res_1
        concept_2 = concept + concept_res_2
        concept_3 = concept + concept_res_3
        concept_1 = concept_1.permute(1,0,2)
        concept_2 = concept_2.permute(1,0,2)
        concept_3 = concept_3.permute(1,0,2)
        
        h, w = img1.shape[2:]
        img1 = self.dy_conv(img1, style1)
        img1 = self.relu(self.batchnorm(img1))
        img1 = self.img_avg_pool(img1.view(bs, -1, h, w)).squeeze()
        img2 = self.dy_conv(img2, style2)
        img2 = self.relu(self.batchnorm(img2))
        img2 = self.img_avg_pool(img2.view(bs, -1, h, w)).squeeze()
        img3 = self.dy_conv(img3, style3)
        img3 = self.relu(self.batchnorm(img3))
        img3 = self.img_avg_pool(img3.view(bs, -1, h, w)).squeeze()
        
        img1 = self.img_final(img1)
        pred1 = self.classifier(img1, concept_1, scale=False)
        
        img2 = self.img_final(img2)
        pred2 = self.classifier(img2, concept_2, scale=False)
        
        img3 = self.img_final(img3)
        pred3 = self.classifier(img3, concept_3, scale=False)
        
        #label1 = torch.argmax(pred1, dim=1)     #batch_size
        #correct1 = torch.eq(label1, pair_labels)
        #correct_sum2 = torch.sum(correct1)
        #print(pred1)
        #print('Testing correct: %d' %correct_sum2)
        
        out = {}
        out['pred_1'] = pred1
        out['pred_2'] = pred2
        out['pred_3'] = pred3

        out['scores_1'], out['scores_2'], out['scores_3'] = {}, {}, {}
        for _, pair in enumerate(self.val_pairs):
            out['scores_1'][pair] = pred1[:,self.pair2idx[pair]]
            out['scores_2'][pair] = pred2[:,self.pair2idx[pair]]
            out['scores_3'][pair] = pred3[:,self.pair2idx[pair]]

        return out

    def forward(self, x):
        if self.training:
            out = self.train_forward(x)
        else:
            with torch.no_grad():
                out = self.val_forward(x)
        return out

class PrtAttLayer(nn.Module):
    def __init__(self, dim, nhead, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = nn.ReLU(inplace=True)

    def prt_assign(self, vis_prt, vis_query):
        vis_prt = self.multihead_attn(query=vis_prt,
                                   key=vis_query,
                                   value=vis_query)[0]
        return vis_prt
        
    def prt_refine(self, vis_prt):
        new_vis_prt = self.linear2(self.activation(self.linear1(vis_prt)))
        return new_vis_prt + vis_prt

    def forward(self, vis_prt, vis_query):
        # sem_prt: 196*bs*c
        # vis_query: wh*bs*c
        vis_prt = self.prt_assign(vis_prt,vis_query)
        vis_prt = self.prt_refine(vis_prt)
        return vis_prt

class CosineClassifier(nn.Module):
    def __init__(self, dataset_name, temp=0.05):
        super(CosineClassifier, self).__init__()
        self.temp = temp
        self.dataset_name = dataset_name

    def forward(self, img, concept, scale=True):
        """
        img: (bs, emb_dim)
        concept: (bs, n_class, emb_dim)
        """
        bs = img.shape[0]
        if self.dataset_name != 'zappos':
            img = F.normalize(img, dim=-1)
            concept = F.normalize(concept, dim=-1)
        
        pred = torch.zeros(bs, concept.shape[1]).cuda()
        for i in range(bs):
            img_i = img[i,:].squeeze()  #300
            concept_i = concept[i,:].squeeze()      #n_pairs*300
            pred[i] = torch.matmul(img_i, concept_i.transpose(0, 1))
        
        if scale:
            pred = pred / self.temp
        return pred

class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)

class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x, att):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(att)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output