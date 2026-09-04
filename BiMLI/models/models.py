# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from models.models_layer import *
from models.contrastive_loss import *
from models.decoder_mlm import ConvE, TuckER,TransE,RotatE

class BiMLI(BaseModel):
    def __init__(self, args,emb_data):
        super(BiMLI, self).__init__(args,emb_data)
        self.encoder = Encoder_LMI(args,emb_data)
        # 'ConvE', 'TuckER','TransE','RotatE' 
        self.rotatE = RotatE(self.h_dim,self.roat_dropout)
        self.criterion = nn.CrossEntropyLoss()
        self.contrastive_loss = Contrastive_Loss(self.contrastive_select,
                                                 self.pre_trained,self.text,self.image,
                                                 self.contrastive_type,self.mm_select,
                                                 self.neg_num,self.temp)


            
    def forward(self,relabeled_edges,uniq_entity,ent_rel_indices):
        '''
        forward 的 Docstring
        :param self: 说明
        :param relabeled_edges: 新编码后，三元组 
        :param uniq_entity: 原始实体由小到大排序，构成新三元组
        :param ent_rel_indices: 说明  新编码后三元组，相同关系、实体聚合，用于解码  三元组中顺序与relabeled_edges不一致，但 len 一致
        '''
        mlm_entity_emd,rel_emd,ent_mutil_ml_emd,ent_mm_emd = self.encoder(relabeled_edges,uniq_entity)  # 以batch三元组计算
        
        head_index = ent_rel_indices[:,0] # 以batch三元组中，头实体、关系为组，预测其尾实体
        real_index = ent_rel_indices[:,1]
        h_e_emd = mlm_entity_emd[head_index]
        r_emd = rel_emd[real_index]
        ent_emd_weight = self.entity_embeddings.weight
        scores = self.rotatE(h_e_emd,r_emd,ent_emd_weight)
        return scores,mlm_entity_emd,rel_emd,ent_mutil_ml_emd,ent_mm_emd
    
    def predict(self,mlm_entity_emd,rel_emd,ent_rel_indices):
        head_index = ent_rel_indices[:,0] # 以batch三元组中，头实体、关系为组，预测其尾实体
        real_index = ent_rel_indices[:,1]
        h_e_emd = mlm_entity_emd[head_index]
        r_emd = rel_emd[real_index]
        ent_emd_weight = self.entity_embeddings.weight
        scores = self.rotatE(h_e_emd,r_emd,ent_emd_weight)
        return scores
    def loss_func(self,output, target,ent_mutil_ml_emd,ent_mm_emd):
        loss_contrastive = self.contrastive_loss(ent_mutil_ml_emd,ent_mm_emd)
        if self.loss_type == 'bec':
            output = F.sigmoid(output)
        bceloss = self.criterion(output,target)
            
        loss = self.beta * bceloss + self.alpha * loss_contrastive
        return loss,loss_contrastive,bceloss
    