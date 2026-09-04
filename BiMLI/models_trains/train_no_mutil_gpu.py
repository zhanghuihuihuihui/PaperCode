import torch
import time
import numpy as np
import wandb

from models_trains.train_no_mutil_gpu_epoch import train_no_mutil_gpu_epoch
from models_trains.train_model import vaild_test_mrr_eval_metrics, get_best_metrics
from utils.util import MultipleOptimizer,get_optims
from utils.data_loader import generate_samples,get_test_val_samples_batch_value
# args_config,model_name,loggerss,hits,tri_datas,emb_data
def train_no_mutil_gpu(args,model_name,loggerss,hits,tri_datas,emb_data):
    logger,logger_every_loss,logger_metric = loggerss
    device = args.device
    all_triplets,train_triplets,test_triplets,val_triplets = tri_datas
    # corpus = BiMLICorpus(args,train_triplets,test_triplets,val_triplets)
    model = model_name[args.model_select](args,emb_data)
    model = model.to(device)
    cnt_wait = 0
    best_val_metrics = model.init_metric_dict(hits)  # 训练过程中，出现较好的结果后，从新计算val级结果
    best_test_metrics = model.init_metric_dict(hits) # 训练过程中，出现较好的结果后，暂时存储 val_metrics
    best_metrics = model.init_metric_dict(hits) #  训练整个过程中结果最好的

    # 手动修改

# params: 要优化的参数（如模型的 parameters()）。
# lr: 学习率（默认值为 0.001）。
# betas: 一阶和二阶动量的衰减系数（默认值为 (0.9, 0.999)）。
# eps: 防止除零的微小值（默认值为 1e-8）。
# weight_decay: 权重衰减系数，用于 L2 正则化。
# ————————————————
# 版权声明：本文为CSDN博主「阿正的梦工坊」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/shizheng_Li/article/details/144447269
    
    optim = get_optims(args,model.parameters())
    optims = MultipleOptimizer(args.lr_scheduler, optim)

    if args.every_loss_verbose == 2:
        wandb.watch(model,log="all")
    if args.every_loss_verbose >1 :
        model.run_id = wandb.run.id
        model_path = f'./checkpoint/20251125-BiMLI_LLL/wandb/{args.dataset}/{args.model_select_wandb}.pth'
        logger.info(f'model.run_id:{model.run_id};project_name:{args.model_select_wandb};model_path:{model_path}')
    logger.info(str(model))
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    logger.info(f'Total number of parameters: {tot_params}')
    logger.info(f'all_triplets.shape:{all_triplets.shape},train_triplets.shape:{train_triplets.shape},test_triplets.shape:{test_triplets.shape},val_triplets.shape:{val_triplets.shape},')
    
    # train_samples_real,train_relabeled_edges,train_uniq_entity = generate_samples(train_triplets,args.rel_num,args.reverse)
    # train_samples_real = train_samples_real.to(args.device)
    # samples_real,relabeled_edges,no_rev_relabeled_edges,uniq_entity
    val_samples_real,val_relabeled_edges,val_no_rev_relabeled_edges,val_uniq_entity = generate_samples(val_triplets,args.rel_num,args.reverse)
    val_samples_real = val_samples_real.to(args.device)

    test_samples_real,test_relabeled_edges,test_no_rev_relabeled_edges,test_uniq_entity = generate_samples(test_triplets,args.rel_num,args.reverse)
    test_samples_real = test_samples_real.to(args.device)

    # all_sr2o = get_sr20_dict(all_triplets,args.rel_num,args.reverse)
    val_head_rel_indices,val_y = get_test_val_samples_batch_value(val_triplets,args.rel_num,args.ent_num,args.reverse)
    test_head_rel_indices,test_y = get_test_val_samples_batch_value(test_triplets,args.rel_num,args.ent_num,args.reverse)
    
    t_tl = time.time()
    mincount = 0
    mincontinuecount = 0
    nev = (args.n_epochs/args.eval_freq) * 0.1
    # minbreak = nev if nev >=5 else 5
    minbreak = 5
    epoch_all_train_time = 0
    epoch_all_test_time = 0
    all_train_number = 0
    all_test_number = 0
    for epoch in (range(args.n_epochs)):
        t = time.time()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        loss,lr,mlm_entity,rel_emd = train_no_mutil_gpu_epoch(args,model,train_triplets,optims,logger_every_loss)
        torch.cuda.synchronize()

        epoch_train_peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        epoch_train_peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)

        print(f"Peak allocated GPU memory: {epoch_train_peak_allocated:.3f} GiB")
        print(f"Peak reserved GPU memory:  {epoch_train_peak_reserved:.3f} GiB")
        epoch_train_time = time.time() -t
        epoch_all_train_time = epoch_all_train_time + epoch_train_time
        if args.every_loss_verbose > 1:
            wandb.log({'Epoch':epoch + 1,'total Loss':loss,'lr':lr,
                       'epoch_time':epoch_train_time,'epoch_all_train_time':epoch_all_train_time,
                       'epoch_all_train_time_mean':epoch_all_train_time/(epoch+1),
                       'epoch_train_peak_allocated':epoch_train_peak_allocated,
                       'epoch_train_peak_reserved':epoch_train_peak_reserved,})

        logger.info(' '.join(['Epoch: {:04d}, Loss: {:.4f}, lr: {:.4f}, epoch_time {:.4f}, total_time {:.4f}'
                        .format(epoch + 1,loss, lr,time.time() -t,(time.time() -t_tl)/3600)]))
        # '''
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            t1 = time.time()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            with torch.no_grad():
                # 看看行不行，不行用batch来计算
                val_metrics = vaild_test_mrr_eval_metrics(args,model,mlm_entity,rel_emd,val_samples_real,val_head_rel_indices,val_y)
            epoch_test_time = time.time() -t1
            epoch_all_test_time = epoch_all_test_time + epoch_test_time
            all_test_number += 1
            torch.cuda.synchronize()

            epoch_test_peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            epoch_test_peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)

            print(f"Peak allocated GPU memory: {epoch_test_peak_allocated:.3f} GiB")
            print(f"Peak reserved GPU memory:  {epoch_test_peak_reserved:.3f} GiB")
            logger.info(' '.join(['Epoch_test: {:04d}, epoch_test_time {:.4f}'
                        .format(epoch + 1,epoch_test_time)]))
            logger_metric.info(model.get_dict_format_metrics(val_metrics, 'val'))
            best_metrics = get_best_metrics(val_metrics,best_metrics)
            if best_metrics["Hits@1"] >  val_metrics["Hits@1"]:
                mincontinuecount += 1
            else:
                mincontinuecount -= 1
            if val_metrics["Hits@1"] < 0.1:
                mincount += 1
                if val_metrics["Hits@1"] < 0.01:
                    mincount += 1
            
            if mincount >= minbreak or mincontinuecount >= minbreak:
                return
            if args.verbose:
                if args.every_loss_verbose > 1:
                    wandb.log({'Epoch_test':epoch + 1,'epoch_test_time':epoch_test_time,
                               'epoch_all_test_number':all_test_number,'epoch_all_test_time':epoch_all_test_time,
                               'epoch_all_test_time_mean':epoch_all_test_time/all_test_number,
                               'epoch_test_peak_allocated':epoch_test_peak_allocated,'epoch_test_peak_reserved':epoch_test_peak_reserved})
                    wandb.log(model.get_dict_format_metrics(val_metrics, 'val'))
                    wandb.log(model.get_dict_format_metrics(best_metrics, 'best'))
                logger_metric.info(model.get_dict_format_metrics(val_metrics, 'val'))
                logger_metric.info(model.get_dict_format_metrics(best_metrics, 'best'))
            if model.has_improved(best_test_metrics, val_metrics):
                with torch.no_grad():
                    best_test_metrics = vaild_test_mrr_eval_metrics(args,model,mlm_entity,rel_emd,test_samples_real,test_head_rel_indices,test_y)
                    # best_val_metrics = vaild_test_mrr_eval_metrics(args,model,mlm_entity,rel_emd,val_samples_real,val_head_rel_indices,val_y)
                counter = 0
                if args.verbose:
                    if args.every_loss_verbose > 1:
                        wandb.log({'Epoch':epoch + 1})
                        wandb.log(model.get_dict_format_metrics(val_metrics, 'val_improved'))
                        wandb.log(model.get_dict_format_metrics(best_test_metrics, 'best_test'))
                        
                    logger_metric.info(' '.join(['Epoch: {:04d}\n'.format(epoch + 1)]))
                    logger_metric.info(model.get_dict_format_metrics(val_metrics, 'val_improved'))
                    logger_metric.info(model.get_dict_format_metrics(best_test_metrics, 'best_test'))
            else:
                counter += 1
                if counter >= args.patience:
                    print("Early stopping")
                    break
    print('Optimization Finished!')
    # print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))
    # if not best_test_metrics:
    model.eval()
    with torch.no_grad():  
        best_test_metrics = vaild_test_mrr_eval_metrics(args,model,mlm_entity,rel_emd,test_samples_real,test_head_rel_indices,test_y)
    if args.verbose:
        if args.every_loss_verbose > 1:
            wandb.log({'Last Result':'metrics'})
            wandb.log(model.get_dict_format_metrics(val_metrics, 'val'))
            # wandb.log(model.get_dict_format_metrics(best_val_metrics, 'best_val'))
            wandb.log(model.get_dict_format_metrics(best_test_metrics, 'final_test'))
        logger_metric.info('Last Result:')
        logger_metric.info(model.get_dict_format_metrics(val_metrics, 'val'))
        # logger_metric.info(model.get_dict_format_metrics(best_val_metrics, 'best_val')) #验证集中结果最好的
        logger_metric.info(model.get_dict_format_metrics(best_test_metrics, 'final_test'))
    if args.save:
        # torch.save(model.state_dict(), f'./checkpoint/{args.dataset}/BiMLI_{time.strftime("%d_%m_%Y")}.pth')
        print('Saved model!')
    if args.every_loss_verbose > 1:
        wandb.finish()
    logger.info(f'project_name:{args.model_select};model.run_id:{model.run_id};dataset:{args.dataset};')
    logger.info(f'device:{args.device}')
    # '''
    return