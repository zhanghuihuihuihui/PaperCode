
from utils.data_loader import generate_samples,get_samples_batch_value
from utils.util import get_model_obj
import numpy as np
import torch
import time
# encoder 目前是全局的  train_samples_real 训练集中所有的三元组，带逆关系
# triplets 原始训练集中，每个batch，不带逆关系
def train_model(args,model,triplets):
   
    samples_real,relabeled_edges,no_rev_relabeled_edges,uniq_entity = generate_samples(triplets,args.rel_num,args.reverse)
    
    # samples_real 真实三元组，有逆关系；relabeled_edges 实体重新排序三元组，有逆关系  no_rev_relabeled_edges 实体重新排序三元组，无逆关系
    # uniq_entity 真实三元组实体，顺序由小到大，其序号为重新排序三元组实体
    head_rel_indices,batch_values = get_samples_batch_value(no_rev_relabeled_edges,args.rel_num,len(uniq_entity),args.reverse)
    batch_real_indices,batch_real_values = get_samples_batch_value(triplets,args.rel_num,args.ent_num,args.reverse)

    # batch_cindices 合并了所有头实体、关系相同的三元组  (h,r,-1) 三元组中顺序与relabeled_edges不一致，但 len 一致
#     samples_real = samples_real.to(args.device)
    relabeled_edges = relabeled_edges.to(args.device)  # batch * 3
    uniq_entity = uniq_entity.to(args.device)
    head_rel_indices = head_rel_indices.to(args.device) # batch * 3  (h,r,-1)
#     batch_values = batch_values.to(args.device) # batch * len(uniq_entity)
#     batch_real_indices = batch_real_indices.to(args.device) # batch * 3  (h,r,-1)
    batch_real_values = batch_real_values.to(args.device) # n * ent_num
    if args.model_select == 'BiMLI':
       score,mlm_entity_emd,rel_emd,ent_mutil_ml_emd,ent_mm_emd = model(relabeled_edges,uniq_entity,head_rel_indices)
      
       # 计算损失 、对比损失
       loss,loss_contrastive,bceloss = get_model_obj(model).loss_func(score,batch_real_values,ent_mutil_ml_emd,ent_mm_emd)
      
    return loss,loss_contrastive,bceloss,mlm_entity_emd,uniq_entity,rel_emd

def vaild_test_mrr_eval_metrics(args,model,mlm_entity,rel_emd,val_samples_real,val_head_rel_indices,val_y):
       # 目前没有批处理
       # all_sr2o是否需要全部，待定，目前是验证集的
       # val_triplets 原始三元组
       # val_samples_real ：带有逆关系的三元组
    ranks = []
    reciprocal_ranks = []
    val_samples_real = val_samples_real.to(args.device)
    val_head_rel_indices = val_head_rel_indices.to(args.device)
    val_y = val_y.to(args.device)
    entity = (torch.arange(args.ent_num)).to(args.device)
    # 预测
    pred = model.predict(mlm_entity,rel_emd,val_head_rel_indices)
    # pred,mlm_entity_emd,rel_emd,ent_mutil_ml_emd,ent_mm_emd = model(val_samples_real,entity,val_head_rel_indices)
    b_range = torch.arange(pred.shape[0], device=args.device)
    
    target = val_head_rel_indices[:, 2]
    target_pred = pred[b_range, target]

    pred = torch.where(val_y.byte(), torch.zeros_like(pred), pred)
    pred[b_range, target] = target_pred
    pred = pred.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    t1 = time.time()
    for i in range(pred.shape[0]):
       scores = pred[i]
       tar = target[i]
       tar_scr = scores[tar]
       scores = np.delete(scores, tar)
       rand = np.random.randint(scores.shape[0])
       scores = np.insert(scores, rand, tar_scr)
       sorted_indices = np.argsort(-scores, kind='stable')
       ranks.append(np.where(sorted_indices == rand)[0][0]+1)
       reciprocal_ranks.append(1.0 / ranks[-1])
    t2 = time.time()
    print(f'vaild_test_mrr_eval_metrics ranks :{t2-t1}')
    hits_at_100,hits_at_50,hits_at_20,hits_at_10,hits_at_3,hits_at_1 = 0,0,0,0,0,0
    # 1, 3, 10, 20, 50 , 100
    for i in range(len(ranks)):
        # if ranks[i] <= 100:
        #     hits_at_100 += 1
        # if ranks[i] <= 50:
        #     hits_at_50 += 1
        # if ranks[i] <= 20:
        #     hits_at_20 += 1
        if ranks[i] <= 10:
            hits_at_10 += 1
        if ranks[i] <= 3:
            hits_at_3 += 1
        if ranks[i] == 1:
            hits_at_1 += 1
    assert len(ranks) == len(reciprocal_ranks)
    # hits_100 = hits_at_100 / len(ranks)
    # hits_50 = hits_at_50 / len(ranks)
    # hits_20 = hits_at_20 / len(ranks)
    hits_10 = hits_at_10 / len(ranks)
    hits_3 = hits_at_3 / len(ranks)
    hits_1 = hits_at_1 / len(ranks)
    mean_rank = sum(ranks) / len(ranks)
    mean_reciprocal_rank = sum(reciprocal_ranks) / len(reciprocal_ranks)
    # metrics = {"Hits@100": hits_100, "Hits@50": hits_50, "Hits@20": hits_20,
    #            "Hits@10": hits_10, "Hits@3": hits_3,"Hits@1": hits_1,
    #             "Mean Rank": mean_rank, "Mean Reciprocal Rank": mean_reciprocal_rank}
    metrics = {"Hits@10": hits_10, "Hits@3": hits_3,"Hits@1": hits_1,
                "Mean Rank": mean_rank, "Mean Reciprocal Rank": mean_reciprocal_rank}

    return metrics

def get_best_metrics(val_metrics,best_test_metrics):
    if val_metrics['Mean Reciprocal Rank'] > best_test_metrics['Mean Reciprocal Rank']:
       best_test_metrics['Mean Reciprocal Rank'] = val_metrics['Mean Reciprocal Rank']
    if val_metrics['Mean Rank'] < best_test_metrics['Mean Rank']:
       best_test_metrics['Mean Rank'] = val_metrics['Mean Rank']
    if val_metrics['Hits@1'] > best_test_metrics['Hits@1']:
          best_test_metrics['Hits@1'] = val_metrics['Hits@1']
    if val_metrics['Hits@3'] > best_test_metrics['Hits@3']:
          best_test_metrics['Hits@3'] = val_metrics['Hits@3']
    if val_metrics['Hits@10'] > best_test_metrics['Hits@10']:
          best_test_metrics['Hits@10'] = val_metrics['Hits@10']
    # if val_metrics['Hits@20'] > best_test_metrics['Hits@20']:
    #       best_test_metrics['Hits@20'] = val_metrics['Hits@20']
    # if val_metrics['Hits@50'] > best_test_metrics['Hits@50']:
    #       best_test_metrics['Hits@50'] = val_metrics['Hits@50']
    # if val_metrics['Hits@100'] > best_test_metrics['Hits@100']:
    #      best_test_metrics['Hits@100'] = val_metrics['Hits@100']
    return best_test_metrics