import os
import sys
import argparse
import torch
import random
import numpy as np
import time
import logging
from argparse import Namespace
from utils.data_utills import load_feat,load_data
dataType = ['EntityTriples', 'train', 'test', 'val']

def get_hits():
    # hits=[1, 3, 10, 20, 50 , 100]
    hits=[1, 3, 10]
    return hits

def get_args():
    parser = argparse.ArgumentParser(description='BiMLI')
    # 数据集  FB15K-237
    parser.add_argument('--dataset', type=str, default='DB15K',choices=['DB15K','FB15K','FB15K-237','YAGO15K','MedMKG','MKG-Y','MKG-W'] )
    parser.add_argument('--datapath', type=str, default='/data/zh/code/dataset/') 

    parser.add_argument('--image', type=int, default=1, help='parameter for the image_features (if needed)')
    parser.add_argument('--text', type=int, default=1, help='parameter for the text_features (if needed)')
    parser.add_argument('--pre_trained', type=int, default=1, help='parameter for the gat_features (if needed 1 other 0 [if 0 Random initialization])')
    parser.add_argument('--reverse', default=True, action='store_true', help='whether to add reverse') 

    parser.add_argument('--miss_text', type=float, default = 0) #5e-6
    parser.add_argument('--miss_image', type=float, default = 0) #5e-6
    
    parser.add_argument("--batch_size", type=int, default= 16384) #16384 8192 4096 2048 1024 512 与RGCNN相关　3000可以  40000不行  1024*64超出范围
    # 数据处理
    parser.add_argument('--log_dir', type=str, default='./logs/BiMLI/')
    parser.add_argument('--model_select', type=str, default='BiMLI', choices=['BiMLI','M'], help='choice of model')
    # model_ml_embd 和 model_lm_embd  model_mlm_agg都需要比较好
    parser.add_argument('--model_ml_embd', type=int, default=1, choices=[0,1], help='choice of gat model ml get every modal')
    parser.add_argument('--model_lm_embd', type=int, default=1, choices=[0,1], help='choice of gat model lm get entity embeding of link')
    parser.add_argument('--model_mlm_agg', type=int, default=1, choices=[0,1], help='choice of gat model mlm get entity embedin')
    
    parser.add_argument('--every_loss_verbose', type=int, default=2, help='0 no verbose；1 logger;2  wandb;3:wandb_sweep whether to print per-epoch logs wandb')
    
    '''以上都固定 除batch_size'''
    parser.add_argument('--optim', type=str, default='RMSprop', choices=['SGD', 'AdaGrad', 'RMSprop', 'Adam','Adadelta'], help='optim select')
    parser.add_argument('--momentum', type=float, default=0.001) #0.1没效果  1e-2 SGD和RMSprop有用
    parser.add_argument('--lr', type=float, default=0.000001387) #0.1没效果  8e-4
    parser.add_argument('--lr_scheduler', type=str, default='cos', choices=['sgdr', 'cos', 'zigzag', 'none'], help='lr scheduler')
    parser.add_argument('--l2reg', type=float, default = 0.000005438) #5e-6

    parser.add_argument("--n_epochs", type=int, default=1000) # 2000 手动设置
    # 验证
    parser.add_argument("--eval_freq", type=int, default=20) # 手动设置
    parser.add_argument('--patience', type=int, default=20, help='number of patience steps for early stopping') # 手动设置
    parser.add_argument("--e_dim", type=int, default=256) #初始化实体 # 手动设置
    parser.add_argument("--r_dim", type=int, default=256) #初始化关系 # 手动设置
    parser.add_argument("--h_dim", type=int, default=256) # 手动设置
    parser.add_argument('--dropout_text_att', type=float, default=0.05426)  #0.41
    parser.add_argument('--dropout_img_att', type=float, default=0.26734) 
    parser.add_argument('--dropout_gat_att', type=float, default=0.1984) 
    parser.add_argument('--dropout_multi_modal_att', type=float, default=0.53759) 
    parser.add_argument('--dropout_att', type=float, default=0.07424) 

    parser.add_argument("--nheads_att", type=int, default=2) #初始化关系 手动设置
    parser.add_argument("--nheads_att_sub", type=int, default=2) #初始化关系 手动设置

    parser.add_argument('--calcul_method', type=str, default='optimizes', choices=['batch', 'optimizes']) # ml计算多模态表示时，是否需要优化
    parser.add_argument('--attention_type', type=str, default='mha', choices=['mha', 'cross']) #attention_type: 多模态计算实体表示注意力类型 ('mha' 或 'cross')
    parser.add_argument('--decode_type', type=str, default='RotatE', choices=['RotatE', 'TuckER','ConvE','TransE']) # Mutan #attention_type: 多模态计算实体表示注意力类型 ('mha' 或 'cross')
    # ConvE k_w * k_h = h_dim
    parser.add_argument('--k_w', type=int, default=8) # 手动设置
    parser.add_argument('--k_h', type=int, default=32) # 手动设置
    parser.add_argument('--out_channels', type=int, default=32) # 手动设置
    parser.add_argument('--kernel_size', type=int, default=3) # 手动设置

    
    parser.add_argument('--cein_drop', type=float, default=0.2) # ConvE
    parser.add_argument('--cehid_drop', type=float, default=0.2) # ConvE
    parser.add_argument('--ceout_drop', type=float, default=0.2) # ConvE
    parser.add_argument('--te_input_drop', type=float, default=0.5) #TuckER
    parser.add_argument('--te_hidden_drop', type=float, default=0.3) #TuckER
    parser.add_argument('--te_out_drop', type=float, default=0.5) # TuckER
    parser.add_argument('--roat_dropout', type=float, default=0.184) #RotatE

    parser.add_argument('--activation_fun_att', type=str, default='ELU',
                        choices=['sigmoid','ReLU','softmax','softmin',
                                 'ELU','tanh','leaky_relu','rrelu','softsign']) # 
    parser.add_argument('--activation_fun_multi_vec', type=str, default='sigmoid',
                        choices=['sigmoid','ReLU','softmax','softmin','ELU',
                                 'tanh','leaky_relu','rrelu','softsign','none']) # 
    # activation_fun_dec
    parser.add_argument('--loss_type', type=str, default='cross',
                        choices=['bec','becWith','binary_cross','cross'])
    
    # contrastive_loss
    parser.add_argument('--contrastive_select', type=int, default=111111) #放弃，未实现，和双向学习相关，目前是模态作为q，链路作为V得到的模态进行学习，还有的是学到最终的融合模态后对比学习
                                                                    # 选择那种对比学习  0 无对比学习; 1 选择多模态对比学习；2 选择实体对比学习   3 两种对比学习都选择 后续再说；
    parser.add_argument('--contrastive_type', type=str, default='CLIP',choices=['IMF','TSAM','CLIP']) # 多模态对比学习类型，IMF，TSAM，CLIP
    parser.add_argument('--mm_select', type=int, default=1,choices=[0, 1, 2, 3]) #0:不要  1:Original：初始模态 ，2: att_ 注意力后模态 3:两种都要
    parser.add_argument('--neg_num', type=float, default=0.5127)
    parser.add_argument('--temp', type=float, default=0.3)
    # alpha 
    parser.add_argument('--alpha', type=float, default=0.83288,help='loss_contrastive')
    parser.add_argument('--beta', type=float, default=0.7481,help='loss_')
    
    """以下都手动设置"""
    # cpu性能
    parser.add_argument('--n_workers', type=int, default=35, help='number of CPU processes for finding counterfactual links in the first run')
    # 是否打印相关参数
    parser.add_argument('--verbose', type=int, default=1, help='whether to print per-epoch logs')
    # 是否存储模型
    parser.add_argument('--save', type=int, default=0, help='whether to save model')

    # 多gpu
    # 是否启用SyncBatchNorm
    # parser.add_argument('--syncBN', type=bool, default=True)
    # parser.add_argument('--freeze_layers', type=bool, default=False)
    # gpu选择
    parser.add_argument('--seed', type=int, default=1000, help='fix random seed if needed')
    parser.add_argument('--device', default='2', help='mutil_gpu:cuda，Otherwise:-1,0,1,mean: cuda device id 0 or 1 or cpu(-1)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    
    # parser.add_argument('--world_size', default=2, type=int,
    #                     help='number of distributed processes')
    # parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    # set_model_params(parser)
    args = parser.parse_args()
    args.argv = sys.argv
    # 分布式：cuda(分布式)；other : cpu or cuda_single
    print(type(args.device))

    
    if args.device == 'cuda':
        args.device = torch.device(args.device)
    else:
        if int(args.device) >= 0:
            torch.cuda.set_device(int(args.device))
        args.device = torch.device(f'cuda:{args.device}' if int(args.device) >= 0 else 'cpu')
    print(f'args.devicelll:{args.device}')
   
    
    args.seed = 5000
    args.model_lm_embd = 0
    args.model_ml_embd = 1
    args.model_mlm_agg = 0
    args.mm_select = 0

    test = f'experiment_seed'
    if args.every_loss_verbose == 2:
        

        args.model_select_wandb = f'{args.model_select}-{test}'
    
    return args

def set_seed(seed):
    if seed > 0:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

def get_data(args):
    data_ = load_data(args.datapath, args.dataset,dataType)
    gat_emb,img_emb,text_emb = load_feat(args.datapath, args.dataset,args.pre_trained,args.image,args.text,args.miss_text,args.miss_image)
    
    data_all = data_[dataType[0]]  # EntityTriples
    data_train = data_[dataType[1]]  # train
    data_test = data_[dataType[2]]  # test
    data_val = data_[dataType[3]]  # val

    all_triplets = np.array(data_all[0])
    train_triplets = np.array(data_train[0])
    test_triplets = np.array(data_test[0])
    val_triplets = np.array(data_val[0])
    
    entity2id = data_all[3][0]
    id2entity = data_all[3][1]
    relation2id = data_all[3][2]
    id2relation = data_all[3][3]
    args.rel_num = len(relation2id)
    args.ent_num = len(entity2id)

    # args.gat_emb = gat_emb
    # args.img_emb = img_emb
    # args.text_emb = text_emb
    tri_datas = (all_triplets,train_triplets,test_triplets,val_triplets)
    emb_data = (gat_emb,img_emb,text_emb)
    return tri_datas,emb_data

def get_all_logger(args):
    log_path = f'{args.log_dir}{args.dataset}/'
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    log_name = f'{log_path}{time.strftime("%d_%m_%Y")}_{time.strftime("%H:%M:%S")}_log'
    log_loss_name = f'{log_path}{time.strftime("%d_%m_%Y")}_{time.strftime("%H:%M:%S")}_loss'
    log_every_loss_name = f'{log_path}{time.strftime("%d_%m_%Y")}_{time.strftime("%H:%M:%S")}_every_loss'
    log_metric_name = f'{log_path}{time.strftime("%d_%m_%Y")}_{time.strftime("%H:%M:%S")}__metric'
    
    logger = get_logger(log_name)
    # logger_loss = get_logger(log_loss_name)
    logger_every_loss = get_logger(log_every_loss_name)
    logger_metric = get_logger(log_metric_name)
    loggerss = (logger,logger_every_loss,logger_metric)
    # loggerss = (logger,logger_loss,logger_every_loss,logger_metric)
    
    logger.info(f'Input argument vector: {args.argv[1:]}')
    logger.info(f'args: {args}')
    logger_every_loss.info(f'args: {args}')
    logger_metric.info(f'args: {args}')
    return loggerss

def get_logger(name):
    """ create a nice logger """
    logger = logging.getLogger(name)
    # clear handlers if they were created in other runs
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # create console handler add add to logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # create file handler add add to logger when name is not None
    if name is not None:
        fh = logging.FileHandler(f'{name}.log')
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    return logger

# gpu中数据处理方法 测试集、验证集，不进行batch_size处理
def get_gpu_dataloader(train_triplets,test_triplets,val_triplets,batch_size,rank):
    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_triplets)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_triplets)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_triplets)
    
    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True) #drop_last True 向下取整，False 向上取整
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    
    train_loader = torch.utils.data.DataLoader(train_triplets,
                                            batch_sampler=train_batch_sampler,
                                            pin_memory=True,
                                            num_workers=nw)
    test_len = test_triplets.shape[0]
    val_len = val_triplets.shape[0]
    test_loader = test_triplets
    val_loader = val_triplets
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
        print('val Using {} dataloader workers every process'.format(val_len))
        print('test Using {} dataloader workers every process'.format(test_len))
        print(f'len(train_loader):{len(train_loader)},len(val_loader):{val_len},len(test_loader):{test_len}')
    
    return train_sampler,train_loader,test_loader,val_loader