import argparse

device = 'cluster'
dataset = 'mitstates'    #mitstates|cgqa
test_domain = 'cartoon'    #photo |art |cartoon
feat_extractor = 'resnet18'     #resnet18 | resnet50 | resnet101 |  vit-b_16 | vit-b_8 | vit-s_16 |  vit-s_8
split_root = "/data3/home_huxiaoming/dataset/" if device == '1080Ti' else "/ghome/huxm/rzsl/data/"
DATA_FOLDER = "/data3/home_huxiaoming/dataset/" if device == '1080Ti' else "/gdata/huxm/rzsl_data/"
model_root = "/data3/home_huxiaoming/dataset/" if device == '1080Ti' else "/ghome/huxm/rzsl/models/"
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--device', default=device)
parser.add_argument('--img_split', default=split_root+dataset+'_img_split/')
parser.add_argument('--pair_split', default=split_root+dataset+'_pair_split/')
parser.add_argument('--word_vector', default=split_root+dataset+'_vector/')
parser.add_argument('--dataset_name', default=dataset)
parser.add_argument('--checkpoint_dir', default=DATA_FOLDER)
parser.add_argument('--root_dir', default=DATA_FOLDER+dataset)
parser.add_argument('--model_root', default=model_root)
parser.add_argument('--feat_extractor', default=feat_extractor)
parser.add_argument('--test_domain', default=test_domain)
parser.add_argument('--subset', action='store_true', default=False, help='test on a 1000 image subset (debug purpose)')
parser.add_argument('--use_fp16', default=False)
parser.add_argument('--wordembs', default='glove')      #glove|fasttext|word2vec|prompt
parser.add_argument('--emb_dim', default=300)
parser.add_argument('--load_checkpoint', default=False)
parser.add_argument('--weights', default='')
parser.add_argument('--optim_weights', default='')
parser.add_argument('--finetune_backbone', default=True)
parser.add_argument('--use_precomputed_features', default=False)
parser.add_argument('--comb_features', default=False)
parser.add_argument('--seed', default=0)
parser.add_argument('--num_workers', default=1)
parser.add_argument('--batch_size', default=64)
parser.add_argument('--test_batch_size', default=256)
parser.add_argument('--save_every_epoch', default=1)
parser.add_argument('--eval_every_epoch', default=1)
parser.add_argument('--topk', default=3)
parser.add_argument('--use_cosine', default=True)
parser.add_argument('--wd', default=0.00005)
parser.add_argument('--lr', default=0.001)
parser.add_argument('--lr_word_embedding', default=0)
parser.add_argument('--lr_encoder', default=0.0005)
parser.add_argument('--img_emb_drop', default=0.3)
parser.add_argument('--wordemb_compose', default='obj-conditioned')
parser.add_argument('--wordemb_compose_dropout', default=0.05)
parser.add_argument('--max_epoch', default=40)
parser.add_argument('--start_epoch_validate', default=1)
parser.add_argument('--decay_strategy', default='milestone')
parser.add_argument('--decay_factor', default=0.1)
parser.add_argument('--lr_decay_milestones', default=[15, 30])
parser.add_argument('--cosine_cls_temp', default=0.05)
