import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Optional
from models.model_util import activation_function_
class MultiHeadAttention(nn.Module):
    """
    自定义多头注意力机制
    实现公式: Attention(Q, K, V) = softmax(QK^T/√d_k)V
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 线性变换层: 将输入投影到Q, K, V
        # self.q_proj = nn.Linear(embed_dim, embed_dim)
        # self.k_proj = nn.Linear(embed_dim, embed_dim)
        # self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        
        # 初始化权重
        self._reset_parameters()
    
    def _reset_parameters(self):
        """初始化权重参数"""
        # nn.init.xavier_uniform_(self.q_proj.weight)
        # nn.init.xavier_uniform_(self.k_proj.weight)
        # nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        # if self.q_proj.bias is not None:
        #     nn.init.zeros_(self.q_proj.bias)
        # if self.k_proj.bias is not None:
        #     nn.init.zeros_(self.k_proj.bias)
        # if self.v_proj.bias is not None:
        #     nn.init.zeros_(self.v_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        参数:
            query: 查询张量 [batch_size, seq_len_q, embed_dim]
            key: 键张量 [batch_size, seq_len_k, embed_dim]
            value: 值张量 [batch_size, seq_len_v, embed_dim]
            key_padding_mask: 键填充掩码 [batch_size, seq_len_k]
            need_weights: 是否返回注意力权重
            
        返回:
            output: 注意力输出 [batch_size, seq_len_q, embed_dim]
            attn_weights: 注意力权重 [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # 线性投影得到Q, K, V
        # q = self.q_proj(query)  # [batch_size, seq_len_q, embed_dim]
        # k = self.k_proj(key)    # [batch_size, seq_len_k, embed_dim]
        # v = self.v_proj(value)  # [batch_size, seq_len_v, embed_dim]
        q = query  # [batch_size, seq_len_q, embed_dim]
        k = key    # [batch_size, seq_len_k, embed_dim]
        v = value  # [batch_size, seq_len_v, embed_dim]
        
        # 重塑为多头形式: [batch_size, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        
        # 转置以便进行批量矩阵乘法: [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len_q, head_dim]
        k = k.transpose(1, 2)  # [batch_size, num_heads, seq_len_k, head_dim]
        v = v.transpose(1, 2)  # [batch_size, num_heads, seq_len_k, head_dim]
        
        # 计算注意力分数: QK^T / √d_k
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        # [batch_size, num_heads, seq_len_q, seq_len_k]
        
        # 应用键填充掩码
        if key_padding_mask is not None:
            # 扩展掩码维度以匹配注意力分数
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len_k]
            attn_scores = attn_scores.masked_fill(key_padding_mask, float('-inf'))
        
        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=2)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重到V
        output = attn_weights * v
        # 转置回原始形状: [batch_size, seq_len_q, num_heads, head_dim]
        output = output.transpose(1, 2).contiguous()
        
        # 合并多头: [batch_size, seq_len_q, embed_dim]
        output = output.view(batch_size, seq_len_q, self.embed_dim)
        output = torch.sum(output,1)
        # 输出投影
        output = self.out_proj(output)
        
        if need_weights:
            return output, attn_weights
        else:
            return output, None


class ModalityAttention(nn.Module):
    """
    模态注意力层
    计算多模态信息对三元组的贡献权重
    """
    def __init__(self, embed_dim: int, num_modalities: int):
        super(ModalityAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        
        # 模态注意力权重计算
        self.modal_weight = nn.Parameter(torch.randn(num_modalities))
        
        # 注意力计算层
        self.attention_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )
        
        # 初始化
        nn.init.uniform_(self.modal_weight, -0.1, 0.1)
    
    def forward(self, 
                modalities: torch.Tensor, 
                link_representation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数:
            modalities: 多模态表示 [batch_size, num_modalities, embed_dim]
            link_representation: 链路表示 [batch_size, embed_dim]
            
        返回:
            weighted_output: 加权输出 [batch_size, embed_dim]
            attention_weights: 注意力权重 [batch_size, num_modalities]
        """
        batch_size, num_modalities, embed_dim = modalities.shape
        
        # 扩展链路表示以匹配模态维度
        link_expanded = link_representation.unsqueeze(1).expand(-1, num_modalities, -1)
        # [batch_size, num_modalities, embed_dim]
        
        # 计算注意力分数
        combined = torch.cat([modalities, link_expanded], dim=-1)
        # [batch_size, num_modalities, embed_dim * 2]
        
        # 通过注意力层计算原始分数
        attention_scores = self.attention_layer(combined).squeeze(-1)
        # [batch_size, num_modalities]
        
        # 加上模态特定权重
        modal_bias = self.modal_weight.unsqueeze(0).expand(batch_size, -1)
        attention_scores = attention_scores + modal_bias
        
        # 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        # [batch_size, num_modalities]
        
        # 计算加权和
        attention_weights_expanded = attention_weights.unsqueeze(-1)
        # [batch_size, num_modalities, 1]
        
        weighted_output = torch.sum(attention_weights_expanded * modalities, dim=1)
        # [batch_size, embed_dim]
        
        return weighted_output, attention_weights


class MultimodalKnowledgeGraph(nn.Module):
    """
    多模态知识图谱模型
    使用多头注意力机制计算模态贡献
    """
    def __init__(self, 
                 h_dim: int,
                 nheads_att: int,
                 nheads_att_sub: int,
                 num_modalities:int,
                 dropout: float,
                 activation_fun_att:str):
        super(MultimodalKnowledgeGraph, self).__init__()
        self.num_nheads = nheads_att
        self.num_heads = nheads_att_sub
        self.embed_dim = h_dim
        self.activation_fun_att = activation_fun_att
        # 多头注意力机制
        self.multihead_attn = MultiHeadAttention(self.embed_dim, self.num_heads, dropout)
        
        # 模态注意力
        self.modality_attention = ModalityAttention(self.embed_dim, num_modalities)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.W = nn.Parameter(torch.zeros(size=(2* self.embed_dim , self.embed_dim)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

    def get_entity_multimodal(self,h_index,mm_emd,mm_need):
        """
        获取实体的多模态表示
        
        参数:
            entity_idx: 实体索引 [batch_size]
            
        返回:
            multimodal: 多模态表示 [batch_size, 3, embed_dim]
            avg_rep: 平均表示 [batch_size, embed_dim]
        """
        struct_emb,image_emb,text_emb = mm_emd
        pre_trained,image,text = mm_need
        # 获取三个模态的嵌入
        if text:
            h_text_emb = text_emb[h_index]  # [batch_size, embed_dim]
            multimodal = h_text_emb
        if image:
            h_image_emb = image_emb[h_index]  # [batch_size, embed_dim]
            multimodal = image_emb
        if pre_trained:
            h_struct_emb = struct_emb[h_index]  # [batch_size, embed_dim]
            multimodal = struct_emb
        
        if text and image and pre_trained:
            # 堆叠为多模态表示
            multimodal = torch.stack([h_text_emb, h_image_emb, h_struct_emb], dim=1)
            # [batch_size, 3, embed_dim]
        else:
            if text and image:
                # 堆叠为多模态表示
                multimodal = torch.stack([h_text_emb, h_image_emb], dim=1)
                # [batch_size, 2, embed_dim]
            if  image and pre_trained:   
                multimodal = torch.stack([h_image_emb, h_struct_emb], dim=1)
            if text and pre_trained:
                multimodal = torch.stack([h_text_emb, h_struct_emb], dim=1)
             
        # 计算平均表示
        avg_rep = multimodal.mean(dim=1)  # [batch_size, embed_dim]
        
        return multimodal, avg_rep
    
    
    def compute_modal_contributions_mha(self,head_modalities,link_vectors) :
        """
        使用多头注意力计算模态贡献
        
        参数:
            head_modalities: 头实体多模态表示 [batch_size, 3, embed_dim]
            link_rep: 链路表示 [batch_size, embed_dim]
            
        返回:
            head_rep: 加权头实体表示 [batch_size, embed_dim]
            attention_weights: 注意力权重 [batch_size, num_heads, 3, 1]
        """
        
        
        # 扩展链路表示作为键(K)
        link_expanded = link_vectors.unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # 使用多头注意力: Q=head_modalities, K=link_expanded, V=head_modalities
        attn_output, attn_weights = self.multihead_attn(
            query=head_modalities,  # [batch_size, 3, embed_dim]
            key=link_expanded,      # [batch_size, 1, embed_dim]
            value=head_modalities,  # [batch_size, 3, embed_dim]
            need_weights=True
        )
        
        # 注意力权重形状: [batch_size, num_heads, 3, 1]
        
        # 计算模态贡献权重(平均所有头)
        # modal_weights = attn_weights.mean(dim=1)  # [batch_size, 3, 1]
        # modal_weights = modal_weights.squeeze(-1)  # [batch_size, 3]
        
        # # 对权重进行softmax归一化
        # modal_weights = F.softmax(modal_weights, dim=-1)  # [batch_size, 3]
        
        # # 计算加权头实体表示
        # modal_weights_expanded = modal_weights.unsqueeze(-1)  # [batch_size, 3, 1]
        # head_rep = torch.sum(modal_weights_expanded * head_modalities, dim=1)  # [batch_size, embed_dim]
        
        return attn_output, attn_weights
    
    def compute_modal_contributions_cross_attention(self,head_modalities,link_rep):
        """
        使用交叉注意力计算模态贡献
        
        参数:
            head_modalities: 头实体多模态表示 [batch_size, 3, embed_dim]
            link_rep: 链路表示 [batch_size, embed_dim]
            
        返回:
            head_rep: 加权头实体表示 [batch_size, embed_dim]
            modal_weights: 模态权重 [batch_size, 3]
        """
        # 使用模态注意力层
        head_rep, modal_weights = self.modality_attention(head_modalities, link_rep)
        
        return head_rep, modal_weights
    
    def forward(self,head_indices ,link_vectors,mm_emd,mm_need, attention_type: str = 'mha'):
        """
        前向传播
        
        参数:
            head_idx: 头实体索引 [batch_size]
            relation_idx: 关系索引 [batch_size]
            tail_idx: 尾实体索引 [batch_size]
            attention_type: 注意力类型 ('mha' 或 'cross')
            
        返回:
            包含各种表示的字典
        """
        # 获取头实体和尾实体的多模态表示
        # text_emb,image_emb,struct_emb  mm_emd,mm_need
        head_modalities, head_avg = self.get_entity_multimodal(head_indices,mm_emd,mm_need)
        x , attention_weights = self.compute_modal_contributions_mha(head_modalities,link_vectors)
                    # 使用交叉注意力
       
        x = activation_function_(self.activation_fun_att,x)
        # 添加残差连接和层归一化
        x = self.norm1(head_avg + self.dropout(x))
        return x,attention_weights


