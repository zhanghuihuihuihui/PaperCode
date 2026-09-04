import torch
import torch.nn as nn
import torch.nn.functional as F
CUDA = torch.cuda.is_available()
# 定义基础评价指标等相关信息
class BaseModel(nn.Module):
    def __init__(self, args,emb_data):
        super(BaseModel, self).__init__()
        self.device = args.device
        self.pre_trained = args.pre_trained
        self.image = args.image
        self.text = args.text
        self.model_ml_embd = args.model_ml_embd
        self.model_lm_embd = args.model_lm_embd
        self.model_mlm_agg = args.model_mlm_agg
        self.reverse = args.reverse
        self.ent_num = args.ent_num
        self.rel_num = args.rel_num
        self.num_rel = args.rel_num
        if args.reverse:
            self.num_rel = 2 * args.rel_num
        self.entity = (torch.arange(self.ent_num)).to(args.device)
        self.relation = (torch.arange(self.num_rel)).to(args.device)

        self.image = args.image
        self.text = args.text
        self.pre_trained = args.pre_trained

        self.entity_embeddings = nn.Embedding(
            self.ent_num, args.e_dim, padding_idx=None
        )
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(
            self.num_rel, args.r_dim, padding_idx=None
        )
        nn.init.xavier_normal_(self.relation_embeddings.weight)

        self.e_dim = args.e_dim
        self.r_dim = args.r_dim

        gat_emb,img_emb,text_emb = emb_data
        if self.pre_trained:
            self.gat_emb_dim = gat_emb.shape[-1]
            self.gat_emb = nn.Embedding.from_pretrained(gat_emb, freeze=False).to(self.device)
        if self.image:
            self.img_emb_dim = img_emb.shape[-1]
            self.img_emb = nn.Embedding.from_pretrained(img_emb, freeze=False).to(self.device) 
        if self.text:
            self.text_emb_dim = text_emb.shape[-1]
            self.text_emb = nn.Embedding.from_pretrained(text_emb, freeze=False).to(self.device) 
        self.h_dim = args.h_dim

        self.dropout_text_att = args.dropout_text_att
        self.dropout_img_att = args.dropout_img_att
        self.dropout_gat_att = args.dropout_gat_att
        self.dropout_multi_modal_att = args.dropout_multi_modal_att
        self.dropout_att = args.dropout_att

        self.nheads_att = args.nheads_att
        self.nheads_att_sub = args.nheads_att_sub
        self.calcul_method = args.calcul_method
        self.attention_type = args.attention_type

        self.decode_type = args.decode_type

        self.k_w = args.k_w
        self.k_h = args.k_h
        self.out_channels = args.out_channels
        self.kernel_size = args.kernel_size
        
        self.cein_drop = args.cein_drop # 0.2
        self.cehid_drop = args.cehid_drop # 0.2
        self.ceout_drop = args.ceout_drop # 0.2

        # input_drop,hidden_drop,out_drop
        self.te_input_drop = args.te_input_drop # 0.3
        self.te_hidden_drop = args.te_hidden_drop # 0.4
        self.te_out_drop = args.te_out_drop # 0.5

        self.roat_dropout = args.roat_dropout # 0.3

        self.alpha = args.alpha
        self.beta = args.beta
        # self. = args.

        self.contrastive_select = args.contrastive_select
        self.contrastive_type = args.contrastive_type
        self.mm_select = args.mm_select
        self.neg_num = args.neg_num
        self.temp = args.temp

        self.activation_fun_att = args.activation_fun_att
        self.activation_fun_multi_vec = args.activation_fun_multi_vec
        
        self.loss_type = args.loss_type
        
    
    @staticmethod
    def init_metric_dict(hits):
        metrics = {}
        for hit in hits:
            metrics[f'Hits@{hit}'] = -1
        metrics['Mean Reciprocal Rank'] = -1
        metrics['Mean Rank'] = 10000
        return metrics
    @staticmethod
    def has_improved(m1, m2):
        return (m1["Mean Rank"] > m2["Mean Rank"]) or (
            m1["Mean Reciprocal Rank"] < m2["Mean Reciprocal Rank"]
        )
    @staticmethod
    def get_dict_format_metrics(metrics, split):
        res = {}
        for metric_name, metric_val in metrics.items():
            key = split+'_'+metric_name
            res[key] = metric_val
        return res
    

class ConvELayer(nn.Module):
    def __init__(self, dim, out_channels, kernel_size, k_h, k_w,cein_drop,cehid_drop,ceout_drop):
        super(ConvELayer, self).__init__()

        self.input_drop = nn.Dropout(cein_drop)
        self.conv_drop = nn.Dropout2d(cehid_drop)
        self.hidden_drop = nn.Dropout(ceout_drop)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm1d(dim)

        self.conv = torch.nn.Conv2d(1, out_channels=out_channels, kernel_size=(kernel_size, kernel_size),
                                    stride=1, padding=0, bias=True)
        assert k_h * k_w == dim
        flat_sz_h = int(2*k_w) - kernel_size + 1
        flat_sz_w = k_h - kernel_size + 1
        self.flat_sz = flat_sz_h * flat_sz_w * out_channels
        self.fc = nn.Linear(self.flat_sz, dim, bias=True)

    def forward(self, conv_input):
        # 输入dropout和BatchNorm
        x = self.bn0(conv_input)
        x = self.input_drop(x)
        # 卷积层
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv_drop(x)
         # 展平
        x = x.view(-1, self.flat_sz)  # x = x.view(self.flat_sz,-1)
        # 全连接层
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x

