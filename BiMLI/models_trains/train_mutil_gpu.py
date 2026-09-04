import os
import torch
import time
from utils.distributed_utils import init_distributed_mode, dist, cleanup
from torch.multiprocessing import Process
# from ogb.linkproppred import Evaluator
import tempfile
import wandb
from args import get_hits,get_data,get_gpu_dataloader
from utils.util import MultipleOptimizer,get_optims
from models_trains.train_mutil_gpu_epoch import train_one_epoch_gpu
from models_trains.train_model import vaild_test_mrr_eval_metrics, get_best_metrics
from utils.data_loader import generate_samples,get_test_val_samples_batch_value
def train_mutil_gpu(opt,model_name,loggerss):
    # params,all_triplets,data_train,data_test,data_valid,hits,best_val_metrics,best_test_metrics
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    gpu_run_type = 1 #两种不同的GPU使用方式
    if gpu_run_type:
        init_distributed_mode(args=opt)
        multi_gpu_train_fun(opt,model_name,loggerss)
    else:
        world_size = opt.world_size
        processes = []
        for rank in range(world_size):
            p = Process(target=multi_gpu_train_spawn, args=(rank, world_size, opt,model_name,loggerss,config))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


def multi_gpu_train_spawn(rank,world_size,args,model_name,loggerss,config):
    # 初始化各进程环境 start
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    args.rank = rank
    args.world_size = world_size
    args.gpu = rank
    args.distributed = True
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
    args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()
    # 初始化各进程环境 end
    multi_gpu_train_fun(args,config,model_name,loggerss)

def multi_gpu_train_fun(args,model_name,loggerss):
    logger,logger_every_loss,logger_metric = loggerss
    # 初始化各进程环境
    rank = args.rank
    batch_size = args.batch_size
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    device = args.device
    checkpoint_path = ""
    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        logger.info('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        # tb_writer = SummaryWriter()
        if os.path.exists("./weights") is False:
            os.makedirs("./weights")
    hits = get_hits()
    all_triplets,train_triplets,test_triplets,val_triplets = get_data(args)
    train_sampler,train_loader,test_loader,val_loader = get_gpu_dataloader(train_triplets,test_triplets,val_triplets,batch_size,rank)
    model = model_name[args.model_select](args)
    model = model.to(device)
    cnt_wait = 0
    best_val_metrics = model.init_metric_dict(hits)  # 训练过程中，出现较好的结果后，从新计算val级结果
    best_test_metrics = model.init_metric_dict(hits) # 训练过程中，出现较好的结果后，暂时存储 val_metrics
    best_metrics = model.init_metric_dict(hits) #  训练整个过程中结果最好的
    checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
    # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
    if rank == 0 and args.save:
        torch.save(model.state_dict(), checkpoint_path)
    dist.barrier()
    # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)
    else:
        # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
        if args.syncBN:
            # 使用SyncBatchNorm后训练会更耗时
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    
    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optim = get_optims(args,pg)
    optims = MultipleOptimizer(args.lr_scheduler, optim)
    t_tl = time.time()

    if args.every_loss_verbose == 2:
        wandb.watch(model,log="all")
    if args.every_loss_verbose >1 :
        model.run_id = wandb.run.id
        model_path = f'./checkpoint/20251125-BiMLI_LLL/wandb/{args.dataset}/{args.model_select}.pth'
        logger.info(f'model.run_id:{model.run_id};project_name:{args.model_select};model_path:{model_path}')
    logger.info(str(model))

    val_samples_real,val_relabeled_edges,val_no_rev_relabeled_edges,val_uniq_entity = generate_samples(val_triplets,args.rel_num,args.reverse)
    val_samples_real = val_samples_real.to(args.device)

    test_samples_real,test_relabeled_edges,test_no_rev_relabeled_edges,test_uniq_entity = generate_samples(test_triplets,args.rel_num,args.reverse)
    test_samples_real = test_samples_real.to(args.device)

    # all_sr2o = get_sr20_dict(all_triplets,args.rel_num,args.reverse)
    val_head_rel_indices,val_y = get_test_val_samples_batch_value(val_triplets,args.rel_num,args.ent_num,args.reverse)
    test_head_rel_indices,test_y = get_test_val_samples_batch_value(test_triplets,args.rel_num,args.ent_num,args.reverse)
    
    for epoch in (range(args.n_epochs)):
        t = time.time()
        train_sampler.set_epoch(epoch)
        mean_loss,lr = train_one_epoch_gpu(args,model,optims,train_loader,epoch,
                                           logger_every_loss)
        if rank == 0:
            logger.info(' '.join(['Epoch: {:04d}, Loss: {:.4f}, lr: {:.4f}, epoch_time {:.4f}, total_time {:.4f} h'
                        .format(epoch + 1,mean_loss, lr,time.time() -t,(time.time() -t_tl)/3600)]))
            if args.every_loss_verbose > 1:
                wandb.log({'Epoch':epoch + 1,'total Loss':mean_loss,'lr':lr,'epoch_time':time.time() -t})
    
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            with torch.no_grad():
                # 看看行不行，不行用batch来计算
                val_metrics = vaild_test_mrr_eval_metrics(args,model,val_samples_real,val_head_rel_indices,val_y)        
            if rank == 0 :
                if args.save:
                    torch.save(model.module.state_dict(), "./weights/model-{}.pth".format(epoch))
                logger_metric.info(' '.join(['Epoch: {:04d}'.format(epoch + 1)]))
                logger_metric.info(model.get_dict_format_metrics(val_metrics, 'val'))
            best_metrics = get_best_metrics(val_metrics,best_metrics)
            if args.verbose and rank == 0:
                if args.every_loss_verbose > 1:
                    wandb.log({'Epoch':epoch + 1})
                    wandb.log(model.get_dict_format_metrics(val_metrics, 'val'))
                    wandb.log(model.get_dict_format_metrics(best_metrics, 'best_metrics'))
                logger_metric.info(model.get_dict_format_metrics(val_metrics, 'val'))
                logger_metric.info(model.get_dict_format_metrics(best_metrics, 'best_metrics'))
            if model.has_improved(best_test_metrics, val_metrics):
                best_test_metrics = val_metrics
                with torch.no_grad():
                    best_val_metrics = vaild_test_mrr_eval_metrics(args,model,val_samples_real,val_head_rel_indices,val_y)
                counter = 0
                if args.verbose and rank == 0:
                    if args.every_loss_verbose > 1:
                        wandb.log({'Epoch':epoch + 1})
                        wandb.log(model.get_dict_format_metrics(val_metrics, 'val'))
                        wandb.log(model.get_dict_format_metrics(best_val_metrics, 'best_val_metrics'))
                        
                    logger_metric.info(' '.join(['Epoch: {:04d}\n'.format(epoch + 1)]))
                    logger_metric.info(model.get_dict_format_metrics(val_metrics, 'val'))
                    logger_metric.info(model.get_dict_format_metrics(best_val_metrics, 'best_val_metrics'))
            else:
                counter += 1
                if counter >= args.patience:
                    print("Early stopping")
                    break
    print('Optimization Finished!')
    # print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))
    if not best_test_metrics:
        model.eval()
        with torch.no_grad():  
            best_test_metrics = vaild_test_mrr_eval_metrics(args,model,test_samples_real,test_head_rel_indices,test_y)
    if args.verbose and rank == 0:
        if args.every_loss_verbose > 1:
            wandb.log({'Last Result':'metrics'})
            wandb.log(model.get_dict_format_metrics(val_metrics, 'val'))
            wandb.log(model.get_dict_format_metrics(best_val_metrics, 'best_val_metrics'))
            wandb.log(model.get_dict_format_metrics(best_test_metrics, 'best_test_metrics'))
        logger_metric.info('Last Result:')
        logger_metric.info(model.get_dict_format_metrics(val_metrics, 'val'))
        logger_metric.info(model.get_dict_format_metrics(best_val_metrics, 'best_val_metrics')) #验证集中结果最好的
        logger_metric.info(model.get_dict_format_metrics(best_test_metrics, 'best_test_metrics'))
    if args.save:
        torch.save(model.state_dict(), f'./checkpoint/{args.dataset}/BiMLI_{time.strftime("%d_%m_%Y")}.pth')
        print('Saved model!')
    if args.every_loss_verbose > 1:
        wandb.finish()
    # 删除临时缓存文件
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    cleanup()
    return