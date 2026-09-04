import torch
import torch.nn as nn
import torch.nn.functional as F
class Contrastive_Loss(nn.Module):
    def __init__(self,contrastive_select,pre_trained,text,image,contrastive_type,mm_select,neg_num,temp):
        super(Contrastive_Loss, self).__init__()
        self.pre_trained = pre_trained
        self.text = text
        self.image = image
        self.mm_select = mm_select  #0:不要  1:Original：初始模态 ，2: att_ 注意力后模态 3:两种都要  
        self.contrastive_select = contrastive_select #选择那种对比学习  0 无对比学习; 1 选择多模态对比学习；2 选择实体对比学习  后续再说； 3 两种对比学习都选择
        self.contrastive_type = contrastive_type # 多模态对比学习类型，IMF，TSAM，CLIP

        self.neg_num = neg_num
        
        self.temp=temp
    
    def contrastive_loss_CLIP(self,s_embed, v_embed, t_embed):
        # 计算相似度矩阵
        sim_matrix_SV = torch.matmul(s_embed, v_embed.T) / self.temp
        sim_matrix_ST = torch.matmul(s_embed, t_embed.T) / self.temp
        sim_matrix_VT = torch.matmul(v_embed, t_embed.T) / self.temp
        # 正样本对：对角线元素
        labels = torch.arange(s_embed.size(0)).to(s_embed.device)
        loss_sv = F.cross_entropy(sim_matrix_SV, labels)
        loss_vs = F.cross_entropy(sim_matrix_SV.T, labels)

        loss_st = F.cross_entropy(sim_matrix_ST, labels)
        loss_ts = F.cross_entropy(sim_matrix_ST.T, labels)

        loss_vt = F.cross_entropy(sim_matrix_VT, labels)
        loss_tv = F.cross_entropy(sim_matrix_VT.T, labels)
        return (loss_sv + loss_vs +loss_st + loss_ts + loss_vt + loss_tv)/6
    
   
    def forward(self, ent_mutil_ml_emd,ent_mm_emd):
        ms = self.mm_select
        if not self.pre_trained or not self.text or not self.image or not ms:
            return 0
        ent_gat_emb,ent_img_emb,ent_text_emb = ent_mm_emd
        e_gat_emd_att,e_img_emd_att,e_text_emd_att = ent_mutil_ml_emd
        ent_gat_emb = F.normalize(ent_gat_emb, p=2, dim=-1, eps=1e-5)
        e_gat_emd_att = F.normalize(e_gat_emd_att, p=2, dim=-1, eps=1e-5)
        ent_text_emb = F.normalize(ent_text_emb, p=2, dim=-1, eps=1e-5)
        e_text_emd_att = F.normalize(e_text_emd_att, p=2, dim=-1, eps=1e-5)
        ent_img_emb = F.normalize(ent_img_emb, p=2, dim=-1, eps=1e-5)
        e_img_emd_att = F.normalize(e_img_emd_att, p=2, dim=-1, eps=1e-5)
        loss = self.contrastive_loss_CLIP(ent_gat_emb, ent_img_emb, ent_text_emb) if ms == 1 else 0
            
        return loss