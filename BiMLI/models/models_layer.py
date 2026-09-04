import torch
import torch.nn as nn
from models.layer import *
from models.encoder_lm_large_mha import *
from models.encoder_ml_large_mha import *
from models.encoder_mlm_agg_large_mha import  *
from collections import defaultdict
from models.model_util import activation_function_
class Encoder_LMI(BaseModel):
     def __init__(self,args,emb_data):
        super(Encoder_LMI, self).__init__(args,emb_data)
        if self.pre_trained:
            self.W_gat = nn.Parameter(torch.zeros(size=(self.gat_emb_dim, self.h_dim)))
            nn.init.xavier_normal_(self.W_gat.data, gain=1.414)
        if self.image:
            self.W_img = nn.Parameter(torch.zeros(size=(self.img_emb_dim, self.h_dim)))
            nn.init.xavier_normal_(self.W_img.data, gain=1.414)
        if self.text:
            self.W_text = nn.Parameter(torch.zeros(size=(self.text_emb_dim, self.h_dim)))
            nn.init.xavier_normal_(self.W_text.data, gain=1.414)
        l_dim = 2 * self.e_dim + self.r_dim
        self.W_l = nn.Parameter(torch.zeros(size=(l_dim, self.h_dim)))
        nn.init.xavier_normal_(self.W_l.data, gain=1.414)

        self.W_r = nn.Parameter(torch.zeros(size=(self.r_dim, self.h_dim)))
        nn.init.xavier_normal_(self.W_r.data, gain=1.414)
        ent_num = self.ent_num
        num_rel = self.num_rel
        h_dim = self.h_dim
        nheads_att = self.nheads_att
        nheads_att_sub = self.nheads_att_sub
        dropout_gat_att = self.dropout_gat_att
        dropout_img_att = self.dropout_img_att
        dropout_text_att = self.dropout_text_att
        dropout_multi_modal_att = self.dropout_multi_modal_att
        dropout_att = self.dropout_att
        num_modalities = self.image + self.text + self.pre_trained
        e_dim = self.e_dim
        r_dim = self.r_dim

        activation_fun_att = self.activation_fun_att

        self.W_E_Tri = nn.Parameter(torch.zeros(size=(e_dim , self.h_dim)))
        nn.init.xavier_normal_(self.W_E_Tri.data, gain=1.414)

        self.W_M_Tri = nn.Parameter(torch.zeros(size=(2 * self.h_dim , self.h_dim)))
        nn.init.xavier_normal_(self.W_M_Tri.data, gain=1.414)

        self.gat_attention = Encoder_ML_Large_MHA(h_dim,
                              nheads_att, nheads_att_sub,
                              dropout_gat_att,activation_fun_att)
        self.img_attention = Encoder_ML_Large_MHA(h_dim,
                              nheads_att, nheads_att_sub,
                              dropout_img_att,activation_fun_att)
        self.text_attention = Encoder_ML_Large_MHA(h_dim,
                              nheads_att, nheads_att_sub,
                              dropout_text_att,activation_fun_att)
        self.multimodall_attention = MultimodalKnowledgeGraph(h_dim,
                                    nheads_att, nheads_att_sub,
                                    num_modalities,
                                    dropout_multi_modal_att,activation_fun_att)
        self.multimodall_agg_attention = Encoder_MLM_AGA_Large_MHA(
                                    e_dim, num_rel , h_dim,
                                    nheads_att, nheads_att_sub,
                                    dropout_att,activation_fun_att)
     def get_mutil_embd(self,uniq_entity): 
         ent_gat_emb,ent_img_emb,ent_text_emb = [],[],[]
         if self.pre_trained:
            ent_gat_emb = self.gat_emb(uniq_entity)
            ent_gat_emb = torch.mm(ent_gat_emb,self.W_gat)
         if self.image:
            ent_img_emb = self.img_emb(uniq_entity)
            ent_img_emb = torch.mm(ent_img_emb,self.W_img)
         if self.text:
            ent_text_emb = self.text_emb(uniq_entity)
            ent_text_emb = torch.mm(ent_text_emb,self.W_text)
         return ent_gat_emb,ent_img_emb,ent_text_emb
     
     def compute_link_vectors(self, relabeled_edges,entity_emd):
        
        # 分解三元组
        head_indices = relabeled_edges[:, 0]
      #   print(head_indices)
        relation_indices = relabeled_edges[:, 1]
      #   print(relation_indices)
        tail_indices = relabeled_edges[:, 2]
      #   print(tail_indices)
        # 批量获取嵌入向量
        head_vecs = entity_emd[head_indices]  # [num_triples, embedding_dim]
        relation_vecs = self.relation_embeddings(relation_indices)  # [num_triples, embedding_dim]
        tail_vecs = entity_emd[tail_indices]  # [num_triples, embedding_dim]
        

        # 链接并投影
        concatenated = torch.cat([head_vecs, relation_vecs, tail_vecs], dim=1)  # [num_triples, 3*embedding_dim]
        # link_vectors = self.link_projection(concatenated)  # [num_triples, projected_dim]
        link_vectors = torch.mm(concatenated,self.W_l)

        # 构建实体到链接的映射
        entity_links = defaultdict(list)
        for i, h_idx in enumerate(head_indices.tolist()):
            entity_links[h_idx].append(i)

        return link_vectors,entity_links
     
     def compute_multi_modal_entity_vectors(self,ent_emd,indices,trip_ent_att_emd,e_mm_emd_att,mm_need):
        '''
        ent_emd：所有实体初始化表示
        indices: 头实体或尾实体 索引
        trip_ent_att_emd：利用多模态信息获取的关于每个链接的头或尾实体实体表示
        e_mm_emd_att：利用链接获取的每个实体的多模态信息表示
        mm_need：使用的多模态类型
        '''
        trip_ent_emd = ent_emd[indices]
        trip_ent_emd = torch.mm(trip_ent_emd,self.W_E_Tri)
        if self.model_lm_embd:
            trip_ent_emd = trip_ent_emd + trip_ent_att_emd
        e_gat_emd_att,e_img_emd_att,e_text_emd_att = e_mm_emd_att
        pre_trained,image,text = mm_need
        if pre_trained:
           e_gat_emd_att = e_gat_emd_att[indices]
           trip_ent_emd = trip_ent_emd + e_gat_emd_att
        if image:
           e_img_emd_att = e_img_emd_att[indices]
           trip_ent_emd = trip_ent_emd + e_img_emd_att
        if text:
           e_text_emd_att = e_text_emd_att[indices]
           trip_ent_emd = trip_ent_emd + e_text_emd_att 
        trip_ent_emd = activation_function_(self.activation_fun_multi_vec,trip_ent_emd)
        return trip_ent_emd
     
     def compute_multi_modal_link_vectors(self, ent_emd,triples,trip_h_ent_emd,
                                         trip_t_ent_emd,e_mm_emd_att,mm_need):
        # 分解三元组
        head_indices = triples[:, 0]
        tail_indices = triples[:, 2]
        h_emdd = self.compute_multi_modal_entity_vectors(ent_emd,head_indices,
                                                         trip_h_ent_emd,e_mm_emd_att,
                                                         mm_need)
        
        t_emdd = self.compute_multi_modal_entity_vectors(ent_emd,tail_indices,
                                                         trip_t_ent_emd,e_mm_emd_att,
                                                         mm_need)

        # 链接并投影
        concatenated = torch.cat([h_emdd, t_emdd], dim=1)  # [num_triples, 3*embedding_dim]
        # link_vectors = self.link_projection(concatenated)  # [num_triples, projected_dim]
        link_Ent_vectors = torch.mm(concatenated,self.W_M_Tri)
        return link_Ent_vectors
     def compute_multi_modal_h_t_vectors(self, ent_emd,triples,trip_h_ent_emd,
                                         trip_t_ent_emd,e_mm_emd_att,mm_need):
        # 分解三元组
        head_indices = triples[:, 0]
        tail_indices = triples[:, 2]
        h_emdd = self.compute_multi_modal_entity_vectors(ent_emd,head_indices,
                                                         trip_h_ent_emd,e_mm_emd_att,
                                                         mm_need)
        
        t_emdd = self.compute_multi_modal_entity_vectors(ent_emd,tail_indices,
                                                         trip_t_ent_emd,e_mm_emd_att,
                                                         mm_need)

       
        return h_emdd,t_emdd
     
     def forward(self,relabeled_edges,uniq_entity):
        head_indices = relabeled_edges[:, 0]
      #   rel_indices = relabeled_edges[:, 1]
      #   tial_indices = triples[:, 2]
        # 批量计算所有链路向量
        entity_emd = self.entity_embeddings(uniq_entity)
        ent_num = len(uniq_entity)
        rel_emd = self.relation_embeddings(self.relation)
      #   print(relabeled_edges)
        link_vectors,entity_links = self.compute_link_vectors(relabeled_edges,entity_emd)  # [num_triples, projected_dim]
        
        # 批量计算头实体每个模态投影
        ent_gat_emb,ent_img_emb,ent_text_emb = self.get_mutil_embd(uniq_entity)
        mm_need = (self.pre_trained,self.image,self.text)
        ent_mm_emd = (ent_gat_emb,ent_img_emb,ent_text_emb)
        e_gat_emd_att,e_img_emd_att,e_text_emd_att = [],[],[]
   
        e_gat_emd_att = self.gat_attention(ent_num,head_indices,link_vectors,
                                       ent_gat_emb,self.calcul_method,'gat')

        e_img_emd_att = self.img_attention(ent_num,head_indices,link_vectors,
                                       ent_img_emb,self.calcul_method,'image')

        e_text_emd_att = self.text_attention(ent_num,head_indices,link_vectors,
                                          ent_text_emb,self.calcul_method,'text')

        ent_mutil_ml_emd = (e_gat_emd_att,e_img_emd_att,e_text_emd_att)
        trip_h_ent_emd,trip_t_ent_emd = [],[]
        
        trip_h_ent_emd,att_weights = self.multimodall_attention(head_indices ,link_vectors,
                                                               ent_mm_emd,mm_need,self.attention_type)
        trip_len = int(relabeled_edges.shape[0])
        mid_trip_len = int(trip_len/2)
        trip_t_ent_emd = torch.cat([trip_h_ent_emd[mid_trip_len:trip_len,:],trip_h_ent_emd[0:mid_trip_len,:]],0)
        
        rel_emd = torch.mm(rel_emd,self.W_r)
        
            # 链接并投影
        h_embd,t_embd = self.compute_multi_modal_h_t_vectors(entity_emd,relabeled_edges,
                                                               trip_h_ent_emd,trip_t_ent_emd,
                                                               ent_mutil_ml_emd,mm_need)
        concatenated = torch.cat([h_embd, t_embd], dim=1)  # [num_triples, 3*embedding_dim]
        link_Ent_vectors = torch.mm(concatenated,self.W_M_Tri)
         
        mlm_entity_emd = self.multimodall_agg_attention(ent_num,entity_emd,relabeled_edges,
                                                         link_Ent_vectors,self.calcul_method)

       
        return mlm_entity_emd,rel_emd,ent_mutil_ml_emd,ent_mm_emd

        
        

class Decoder_LMI(BaseModel):
     def __init__(self,args):
        super(Decoder_LMI, self).__init__(args)
        #   decode_type,ent_num, dim,k_w,k_h,out_channels,kernel_size
        #   'ConvE', 'TuckER','Mutan'
        
         
     def forward(self,batch_indices,ent_emd,rel_emd):
         
         return
     
