import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_scatter import scatter_mean, scatter_softmax, scatter_add
from models.model_util import activation_function_
import pickle
class Encoder_ML_Large_MHA(nn.Module):
    """
    大规模知识图谱高效多头注意力表示学习
    专为大规模数据优化，完全向量化实现
    # M 多模态 L 链接 M作为K，L为Q，V计算每种模态表示
    """
    def __init__(self,h_dim,nheads_att,nheads_att_sub,dropout,activation_fun_att):
        super(Encoder_ML_Large_MHA, self).__init__()
        
        # self.num_entities = ent_num
        # self.num_relations = num_rel
        self.num_nheads = nheads_att
        self.num_heads = nheads_att_sub
        self.embedding_dim = h_dim
        self.activation_fun_att = activation_fun_att
        
        # 验证维度可分割性
        assert h_dim % self.num_heads == 0, "projected_dim必须能被num_heads整除"
        self.head_dim = h_dim // self.num_heads
        
        # 多头注意力参数 (Q, K, V投影)
        # self.w_q = nn.Linear(h_dim, h_dim, bias=False)
        # self.w_k = nn.Linear(h_dim, h_dim, bias=False)
        # self.w_v = nn.Linear(h_dim, h_dim, bias=False)
        
        # 输出投影
        self.fc_out = nn.Linear(h_dim, h_dim)
        
        # LayerNorm 和 Dropout
        self.layer_norm1 = nn.LayerNorm(h_dim)
        self.layer_norm2 = nn.LayerNorm(h_dim)
        self.dropout = nn.Dropout(dropout)
        self.W = nn.Parameter(torch.zeros(size=(2 * self.embedding_dim , self.embedding_dim)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        
    
    
    def compute_entity_representations_optimized(self,ent_num, head_indices,link_vectors,m_embd,model):
        """
        高度优化的实体表示计算
        使用矩阵运算和分组操作，完全避免循环
        """
        head_m_embd = m_embd[head_indices]
        # 重塑为多头格式
        link_vectors_multi = link_vectors.view(-1, self.num_heads, self.head_dim)  # [num_triples, num_heads, head_dim]
        head_emd_multi = head_m_embd.view(-1, self.num_heads, self.head_dim)  # [num_triples, num_heads, head_dim]
        
        # 计算每个头的注意力分数
        attention_scores = torch.sum(head_emd_multi * link_vectors_multi, dim=2)  # [num_triples, num_heads]
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # 对每个头应用分组softmax
        attention_weights = torch.zeros_like(attention_scores)
        for head in range(self.num_heads):
            attention_weights[:, head] = scatter_softmax(
                attention_scores[:, head], head_indices, dim=0
            )
        # path = f'/mnt/hui/code/MyPaperCode/20251212-BiMLI/20251125-BiMLI_Local_ok-v2-ablation_copy/attention_weight/{model}.pkl'
        # pickle.dump(attention_weights,open(path,'wb'))
        
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        weighted_links_multi = link_vectors_multi * attention_weights.unsqueeze(2)  # [num_triples, num_heads, head_dim]
        
        # 合并多头
        weighted_links = weighted_links_multi.view(-1, self.embedding_dim)  # [num_triples, projected_dim]
        
        # 分组求和
        entity_repr = scatter_add(weighted_links, head_indices, dim=0, dim_size=ent_num)
        entity_repr = self.fc_out(entity_repr)
        # 处理没有三元组的实体
        head_counts = torch.bincount(head_indices, minlength=ent_num)
        empty_heads = head_counts == 0
        if torch.any(empty_heads):
            empty_head_proj = m_embd[empty_heads]
            entity_repr[empty_heads] = empty_head_proj
        
        return entity_repr, {
            'attention_weights': attention_weights,
            'head_counts': head_counts
        }
    def forward(self,ent_num, head_indices,link_vectors,m_embd,calcul_method,model):
        x ,info = self.compute_entity_representations_optimized(ent_num,head_indices,link_vectors, m_embd,model)
        x = activation_function_(self.activation_fun_att,x)
        # 后续看需不需要加入线性变化和drop等操作
        return x