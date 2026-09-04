import torch
from torch.utils.data import DataLoader
from models_trains.train_model import train_model
import time                                                    
def train_no_mutil_gpu_epoch(args,model,train_triplets,optims,logger_every_loss):
    epoch_loss = []
    total_examples = 0
    total_loss = 0
    model.train()
    mlm_entity_emds = torch.zeros(size=(args.ent_num , args.h_dim)).to(args.device)
    mlm_entity_emds_num = torch.zeros(size=(args.ent_num , 1)).to(args.device)
    for perm in DataLoader(range(len(train_triplets)), args.batch_size, shuffle=True):
        triplets = train_triplets[perm]
        
        lr = optims.update_lr(args.lr)
        optims.zero_grad()
        t = time.time()
        loss,loss_contrastive,bceloss,mlm_entity_emd,uniq_entity,rel_emd = train_model(args,model,triplets)
        mlm_entity_emds[uniq_entity] = mlm_entity_emds[uniq_entity] + mlm_entity_emd
        mlm_entity_emds_num[uniq_entity] = mlm_entity_emds_num[uniq_entity] + 1
        logger_every_loss.info(' '.join(['loss:{:.8f},loss_contrastive: {:.8f} ,bceloss: {:.8f} ,time: {:.4f} \n'.format(loss,loss_contrastive,bceloss,time.time() - t)]))
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optims.step()
        total_loss += loss.item()
        epoch_loss.append(loss.data.item())
    loss = sum(epoch_loss) / len(epoch_loss)
    # 1. 将维度 (ent_num, 1) 压缩为 (ent_num, )，方便后续生成一维掩码
    counts = mlm_entity_emds_num.squeeze()

    # 2. 生成布尔掩码 (Boolean Mask) 来替代直接寻找索引
    # non_zero_mask 中值为 True 的位置代表 counts 不为 0
    non_zero_mask = counts > 0
    zero_mask = counts == 0

    # counts = counts.to(args.device)
    # 3. 初始化最终的实体特征张量
    mlm_entity = torch.zeros_like(mlm_entity_emds)

    # 4. 处理不为 0 的部分：直接相除求平均
    # 注意：使用 counts[non_zero_mask].unsqueeze(1) 恢复形状为 (N, 1) 以便进行广播除法
    mlm_entity[non_zero_mask] = mlm_entity_emds[non_zero_mask] / counts[non_zero_mask].unsqueeze(1)

    # 5. 计算已更新实体特征的全局平均值 (按特征维度即 dim=0 求均值)
    # valid_mean 形状为 (h_dim, )
    valid_mean = mlm_entity[non_zero_mask].mean(dim=0)

    # 6. 处理为 0 的部分：使用刚刚计算的全局平均值进行填充
    mlm_entity[zero_mask] = valid_mean
    return loss,lr,mlm_entity,rel_emd