import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_scatter import scatter_mean, scatter_softmax, scatter_add
from models.model_util import activation_function_
class Encoder_MLM_AGA_Large_MHA(nn.Module):
    """
    大规模知识图谱高效多头注意力表示学习
    专为大规模数据优化，完全向量化实现
    # M 多模态 L 链接 M作为K，L为Q，V计算每种模态表示
    """
    def __init__(self,e_dim,num_rel,h_dim,nheads_att,nheads_att_sub,dropout,activation_fun_att):
        super(Encoder_MLM_AGA_Large_MHA, self).__init__()
        self.num_relations = num_rel
        self.num_nheads = nheads_att
        self.num_heads = nheads_att_sub
        self.embedding_dim = h_dim
        self.activation_fun_att = activation_fun_att
        # 验证维度可分割性
        assert h_dim % self.num_heads == 0, "projected_dim必须能被num_heads整除"
        self.head_dim = h_dim // self.num_heads
        
        # 多头注意力参数 (Q, K, V投影)
        self.w_q = nn.Linear(e_dim, h_dim, bias=False)
        self.w_k = nn.Linear(e_dim, h_dim, bias=False)
        self.w_v = nn.Linear(h_dim, h_dim, bias=False)

        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        

        self.W_M_Tri = nn.Parameter(torch.zeros(size=(2 * self.embedding_dim , self.embedding_dim)))
        nn.init.xavier_normal_(self.W_M_Tri.data, gain=1.414)
        # 输出投影
        self.fc_out = nn.Linear(h_dim, h_dim)
        
        # LayerNorm 和 Dropout
        self.layer_norm1 = nn.LayerNorm(h_dim)
        self.layer_norm2 = nn.LayerNorm(h_dim)
        self.dropout = nn.Dropout(dropout)
        self.W = nn.Parameter(torch.zeros(size=( 2 * self.embedding_dim , self.embedding_dim)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
    
    def compute_entity_representations_batch(self,ent_num, triples,link_Ent_vectors,ent_embd):
        """
        批量计算实体表示 (内存高效)
        
        Args:
            triples_tensor: 三元组张量 [num_triples, 3]
            batch_size: 批次大小
            
        Returns:
            entity_repr: 实体表示 [num_entities, projected_dim]
            attention_info: 注意力相关信息
        """
        head_indices = triples[:, 0]
        tail_indices = triples[:, 2]
        head_embd = ent_embd[head_indices]
        tail_embd = ent_embd[tail_indices]
        # 使用图散射操作进行高效分组注意力计算
        # print("使用图散射操作进行分组注意力计算...")
        Q = self.w_q(head_embd) 
        K = self.w_k(tail_embd)
        # 计算每个头实体对应的三元组数量
        head_counts = torch.bincount(head_indices, minlength=ent_num)
        
        # 使用散射操作计算分组注意力
        # 1. 计算注意力分数 (头实体K与链路Q的点积)
        attention_scores = torch.sum(Q * K, dim=1) / math.sqrt(self.head_dim)  # [num_triples]
        
        # 2. 使用scatter_softmax计算分组softmax
        attention_weights = scatter_softmax(attention_scores, head_indices, dim=0)  # [num_triples]
        
        attention_weights = self.dropout(attention_weights)
        # 3. 应用注意力权重到链路向量(V)
        weighted_links = link_Ent_vectors * attention_weights.unsqueeze(1)  # [num_triples, projected_dim]
        
        # 4. 使用scatter_add进行分组求和
        entity_repr = scatter_add(weighted_links, head_indices, dim=0, dim_size=ent_num)  # [num_entities, projected_dim]
        entity_repr = self.fc_out(entity_repr)
        # 5. 对于没有三元组的实体，使用其投影后的向量
        empty_heads = head_counts == 0
        if torch.any(empty_heads):
            # 获取没有三元组的实体的投影向量
            empty_head_proj = ent_embd[empty_heads]
            entity_repr[empty_heads] = empty_head_proj
        
        return entity_repr, {
            'attention_weights': attention_weights,
            'head_counts': head_counts
        }
    
    def compute_entity_representations_optimized(self,ent_num, triples,link_Ent_vectors,ent_embd):
        """
        高度优化的实体表示计算
        使用矩阵运算和分组操作，完全避免循环
        """
        head_indices = triples[:, 0]
        tail_indices = triples[:, 2]
        head_embd = ent_embd[head_indices]
        tail_embd = ent_embd[tail_indices]
        Q = self.w_q(head_embd)  # [batch_size, seq_len, projected_dim]
        K = self.w_k(tail_embd)
        
        # 重塑为多头格式
        link_vectors_multi = link_Ent_vectors.view(-1, self.num_heads, self.head_dim)  # [num_triples, num_heads, head_dim]
        Q = Q.view(-1, self.num_heads, self.head_dim)  # [num_triples, num_heads, head_dim]
        K = K.view(-1, self.num_heads, self.head_dim)
        # 计算每个头的注意力分数
        attention_scores = torch.sum(Q * K, dim=2)  # [num_triples, num_heads]
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # 对每个头应用分组softmax
        attention_weights = torch.zeros_like(attention_scores)
        for head in range(self.num_heads):
            attention_weights[:, head] = scatter_softmax(
                attention_scores[:, head], head_indices, dim=0
            )
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
            empty_head_proj = ent_embd[empty_heads]
            entity_repr[empty_heads] = empty_head_proj
        
        return entity_repr, {
            'attention_weights': attention_weights,
            'head_counts': head_counts
        }
   
   
    def forward(self,ent_num,ent_emd ,triples,link_Ent_vectors,calcul_method):
        
        # triples,link_Ent_vectors,ent_embd
        x,info = self.compute_entity_representations_optimized(ent_num,triples,
                                                                                 link_Ent_vectors,ent_emd)      
        
        x = activation_function_(self.activation_fun_att,x)
        # 后续看需不需要加入线性变化和drop等操作
        return x