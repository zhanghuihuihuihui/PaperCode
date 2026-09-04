import pickle
import torch.nn.functional as F
import torch
def load_data(datapath, dataset,dataType):
    path = datapath + dataset + '/'
    data = {}
    for split in dataType:
        triples, adj , unique_entities, entrel= get_adj(path,split)
        data[split] = (triples, adj , unique_entities, entrel)
    return data

def get_adj(path,split):
    entity2id,id2entity = read_entity_from_id(path)  #返回字典：key：实体（<http://dbpedia.org/resource/Kyrgyzstan>）;value:序号（6）
    relation2id,id2relation = read_relation_from_id(path) #同上
    triples = []
    rows, cols, data = [], [], []
    unique_entities = set()
    with open(path+split+'.txt', 'r') as f:
        for line in f:
            instance = line.strip().split(' ')  #strip()去除首位空格
            e1, r, e2 = instance[0], instance[1], instance[2]
            unique_entities.add(e1)  #返回set集合，实体集合，实体唯一
            unique_entities.add(e2)
            triples.append((entity2id[e1], relation2id[r], entity2id[e2])) # 存储为实体、关系的编码序号（0,1,2....）
            rows.append(entity2id[e2]) #尾实体序号（0、1、2、3...）
            cols.append(entity2id[e1]) #头实体序号
            data.append(relation2id[r]) #关系实体序号

    # id2relation.update({relation2id[rel] : rel for rel in relation2id})
    # 返回 数组，每个元素为元组（元组形式唯三元组-序号-）；三元组，三元组为尾头关系-序号-的三个数组；实体集合
    return triples, (rows, cols, data), unique_entities, (entity2id,id2entity,relation2id,id2relation)       

def read_entity_from_id(path):
    entity2id = {}
    id2entity = {}
    with open(path + 'entity2id.txt', 'r') as f:
        for line in f:
            instance = line.strip().split()
            entity2id[instance[0]] = int(instance[1])
            id2entity[int(instance[1])] = instance[0]
    #返回字典：key：实体（<http://dbpedia.org/resource/Kyrgyzstan>）;value:序号（6）
    return entity2id,id2entity

def read_relation_from_id(path):
    relation2id = {}
    id2relation = {}
    with open(path + 'relation2id.txt', 'r') as f:
        for line in f:
            instance = line.strip().split()
            relation2id[instance[0]] = int(instance[1])
            id2relation[int(instance[1])] = instance[0]

    return relation2id,id2relation
def load_feat(datapath, dataset,pre_trained,image,text,miss_text,miss_image):
    path = datapath + dataset + '/'
    # rel_feat = pickle.load(open(path+'gat_relation_vec.pkl', 'rb'))
    img_emb = []
    text_emb = []
    gat_emb = []
    if image == 1:
        img_feat = pickle.load(open(path+'img_features.pkl', 'rb'))
        
        print(type(img_feat))
        if miss_image > 0:
            img_feat = torch.Tensor(img_feat)
            img_feat = get_miss_feat(img_feat,miss_image)
        img_emb = F.normalize(torch.Tensor(img_feat), p=2, dim=1)
    if text == 1:
        text_feat = pickle.load(open(path+'text_features.pkl', 'rb'))
        print(type(text_feat))
        if miss_text > 0:
            text_feat = torch.Tensor(text_feat)
            text_feat = get_miss_feat(text_feat,miss_text)
        text_emb = F.normalize(torch.Tensor(text_feat), p=2, dim=1)
    if pre_trained:
        gat_feat = pickle.load(open(path+'gat_entity_vec.pkl', 'rb'))
        gat_emb = F.normalize(torch.Tensor(gat_feat), p=2, dim=1)
    return gat_emb,img_emb,text_emb

def get_miss_feat(feat,miss):
    n,m = feat.shape
    replace_count = int(n * miss)
    indices = torch.randperm(n)

    replace_idx = indices[:replace_count]
    keep_idx = indices[replace_count:]
    mean_feat = feat[keep_idx].mean(dim=0)

    feat[replace_idx] = mean_feat
    return feat