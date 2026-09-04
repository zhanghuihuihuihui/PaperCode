import torch
import numpy as np
def generate_samples(triples,rel_num,reverse):
    if isinstance(triples,torch.Tensor):
        triples = triples.numpy()
    src, rel, dst = triples.transpose()
    # 此处uniq_entity为真实实体编码，且由小到大排序，edges为三元组头实体尾实体，重排序后编码【即uniq_entity的索引】
    uniq_entity, edges = np.unique((src, dst), return_inverse=True)  
    src, dst = np.reshape(edges, (2, -1))
    no_rev_relabeled_edges = np.stack((src, rel, dst)).transpose()
    if reverse:
        # Create bi-directional graph
        src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
        rel = np.concatenate((rel, rel + rel_num))
    relabeled_edges = np.stack((src, rel, dst)).transpose()
    src_real = uniq_entity[src] # 真实编码
    dst_real = uniq_entity[dst]
    samples_real = np.stack((src_real, rel, dst_real)).transpose()
    samples_real = torch.from_numpy(samples_real) #原图谱中编码
    relabeled_edges = torch.from_numpy(relabeled_edges) #新编码
    uniq_entity = torch.from_numpy(uniq_entity) #原图谱中编码，由小到大排序后，用于新编码
    no_rev_relabeled_edges = torch.from_numpy(no_rev_relabeled_edges)
    return samples_real,relabeled_edges,no_rev_relabeled_edges,uniq_entity

def get_samples_batch_value(triples,rel_num,ent_num,reverse):
    if isinstance(triples,torch.Tensor):
        triples = triples.numpy()
    sr2o = get_sr20_dict(triples,rel_num,reverse)
    tri_indices = [{'triple': (head, relation, -1), 'label': list(sr2o[(head, relation)])}
                    for (head, relation), tail in sr2o.items()]
    batch_indices = torch.LongTensor([indice['triple'] for indice in tri_indices])
    label = [np.int32(indice['label']) for indice in tri_indices]
    y = np.zeros((len(tri_indices), ent_num), dtype=np.float32)

    for idx in range(len(label)):
        for l in label[idx]:
            y[idx][l] = 1.0
    y = 0.9 * y + (1.0 / ent_num)
    batch_values = torch.FloatTensor(y)
    return batch_indices,batch_values
# 时间慢的话，就使用batch 看情况 
def get_test_val_samples_batch_value(val_triplets,rel_num,ent_num,reverse):
    if isinstance(val_triplets,torch.Tensor):
        val_triplets = val_triplets.numpy()
    vail_sr2o = get_sr20_dict(val_triplets,rel_num,reverse)
    h_t_indices = [{'triple': (head, relation, tail), 'label': list(vail_sr2o[(head, relation)])}
                                 for (head, relation, tail) in val_triplets]
    head_batch_indices = torch.LongTensor([indice['triple'] for indice in h_t_indices])
    label = [np.int32(indice['label']) for indice in h_t_indices]
    y = np.zeros((len(h_t_indices), ent_num), dtype=np.float32)
    for idx in range(len(label)):
        for l in label[idx]:
            y[idx][l] = 1.0
    y = torch.FloatTensor(y)
    return head_batch_indices,y

def get_sr20_dict(triples,rel_num,reverse):
    sr2o = {}
    for (head, relation, tail) in triples:
        if (head, relation) not in sr2o.keys():
                sr2o[(head, relation)] = set()
        sr2o[(head, relation)].add(tail)
        if reverse:
            if (tail, relation+rel_num) not in sr2o.keys():
                sr2o[(tail, relation+rel_num)] = set()
            sr2o[(tail, relation+rel_num)].add(head)
    return sr2o


