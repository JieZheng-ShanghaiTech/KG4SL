import numpy as np
import pandas as pd
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import scipy.sparse as sp

# Config
relation2id_path = '../data/relation2id.csv'
kg_path =  '../data/kg_triplet.csv'
human_sl_path = '../data/sl_data'
kg_save = '../data/kg2id.txt'
sl_save = '../data/sl2id.txt'
entity_save = '../data/entity2id.txt'
 
# -----------------------------------------------------Begin-------------------------------------------
relation2id = pd.read_csv(relation2id_path)

human_SL = pd.read_csv(human_sl_path, sep=' ')
human_SL = human_SL[['gene_a.identifier','gene_b.identifier']]

print('Read kg:')
kg = pd.read_csv(kg_path, sep=',')

# kg: relation name -> id
for i in range(len(relation2id)):
    a = relation2id['type'][i]
    b = relation2id['id'][i]
    kg.loc[kg['type(r)']==a,'type(r)'] = b

# delect relation 1,14,24 (SL, nonSL, SR)
kg_delete = kg[(kg['type(r)'] != 1)&(kg['type(r)'] != 14)&(kg['type(r)'] != 24)]

# change column order
order = ['ID(a)', 'type(r)', 'ID(b)']
kg_delete = kg_delete[order]
kg_delete = kg_delete.reset_index(drop=True)

# reindex for relations (new_indices: 0~23)
relation_old = list(set(kg_delete['type(r)']))
relation_map = {}
for i in range(len(relation_old)):
    relation_map[relation_old[i]] = i
relation_new = []
for i in kg_delete['type(r)']:
    relation_new.append(relation_map[i])
kg_delete['type(r)'] = relation_new

print('The first 10 rows of kg: ')
print(kg_delete[:10])
#--------------------------------delect the same head-tail/tail-head in kg and Human_SL pair (Optional）----------------------
index_list = []
for index, row in human_SL.iterrows():
    gene_a = row['gene_a.identifier']
    gene_b = row['gene_b.identifier']
    index_list.append(kg_delete[(kg_delete['ID(a)'] == gene_a)&(kg_delete['ID(b)'] == gene_b)].index.tolist())
    index_list.append(kg_delete[(kg_delete['ID(b)'] == gene_a)&(kg_delete['ID(a)'] == gene_b)].index.tolist())

list_same = []
for a in index_list:
    for b in a:
        list_same.append(b)

# print('The index of triples in kg for same head-tail/tail-head with Human_SL: ',list_same)
print('The length of triples in kg for same head-tail/tail-head with Human_SL:', len(list_same))

kg_delete = kg_delete.drop(kg_delete.index[list_same])  # Drop for kg
kg_delete = kg_delete.reset_index(drop=True)
#---------------------------------------------Delect genes that in Huaman_SL but not in kg------------------------------

print('Read Human_SL matrix:')
print(human_SL[:10])
set_gene_a = set(human_SL['gene_a.identifier'])
set_gene_b = set(human_SL['gene_b.identifier'])
set_sl = set_gene_a | set_gene_b
set_IDa = set(kg_delete['ID(a)'])
set_IDb = set(kg_delete['ID(b)'])
set_kg = set_IDa | set_IDb

list_different = list(set_sl - set_kg)  
print('The human_sl pairs:', len(human_SL))
print('The human_sl genes:', len(set_sl))
# print('The number of genes in SL pairs not in kg:', len(list_different))
row_delete = []  
for index, row in human_SL.iterrows():
    if(row['gene_a.identifier'] in list_different or row['gene_b.identifier'] in list_different):
        row_delete.append(index)
# print(row_delete)
# print('The number of rows in Human_SL to delete (not in the kg): ',len(row_delete))

human_SL = human_SL.drop(human_SL.index[row_delete])
print('Human_SL length after deleting rows not in kg:', len(human_SL))

#---------------------------------------------Human_SL reindex---------------------------------------
set_gene_a = set(human_SL['gene_a.identifier'])
set_gene_b = set(human_SL['gene_b.identifier'])
list_all = list(set_gene_a | set_gene_b)

# print("The number of genes for reindex: " ,len(list_all))

# reindex
entity_key = {}  
for i in range(len(list_all)):
    origin = list_all[i]
    entity_key[origin] = i

# reindex
for key in entity_key:
    human_SL.loc[human_SL['gene_a.identifier']==key,'gene_a.reindex'] = entity_key[key]
    human_SL.loc[human_SL['gene_b.identifier']==key,'gene_b.reindex'] = entity_key[key]

# print('human_sl after single reindex:')
# print(human_SL[:10])

human_SL_reindex = human_SL[['gene_a.reindex','gene_b.reindex']]
human_SL_reindex = human_SL_reindex.astype(int)

#-------------------------------------------------------Create Negative Samples----------------------------------------------
human_SL_retain = human_SL_reindex.reset_index(drop=True)  

# Create negative samples
inter_pairs = []
for index, row in human_SL_retain.iterrows():
    name1 = row['gene_a.reindex']
    name2 = row['gene_b.reindex']
    inter_pairs.append((name1, name2))

inter_pairs = np.array(inter_pairs, dtype=np.int32)

u = torch.tensor(inter_pairs[:,0])
v = torch.tensor(inter_pairs[:,1])
g = dgl.graph((u, v))
# g = dgl.to_bidirected(g)
# print(g)

len1 = np.max(v.numpy())
len2 = np.max(u.numpy())
len_ = max(len1,len2) + 1  # add 0

# Find all negative edges and sample
print('Begin to sample negative SL pairs...')

adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())),shape = (len_,len_))
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  
adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
neg_u, neg_v = np.where(adj_neg != 0)
neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
print('neg_eids lens: ', len(neg_eids))

positive_ids = [1] * len(human_SL_retain)
human_SL_retain['type'] = positive_ids

neg_idx = 0
for i in range(len(positive_ids), len(positive_ids) + len(positive_ids)):
    human_SL_retain.loc[i] = [neg_u[neg_eids[neg_idx]], neg_v[neg_eids[neg_idx]], 0]
    neg_idx = neg_idx + 1

print('neg_idx; ',neg_idx)

print("The len of human_sl positive + negative pairs: ", len(human_SL_retain))

human_SL = human_SL_retain

entity_key_reverse = {value:key for key,value in entity_key.items()}   

# reindex
for key in entity_key_reverse:
    human_SL.loc[human_SL['gene_a.reindex']==key,'gene_a.identifier'] = entity_key_reverse[key]
    human_SL.loc[human_SL['gene_b.reindex']==key,'gene_b.identifier'] = entity_key_reverse[key]

human_SL = human_SL.astype(int)
human_SL = human_SL[['gene_a.identifier','gene_b.identifier','type']]

# print('Human_sl after add nagative samples (origin identifiers):')
# print(human_SL[:10])

#-----------------------------------------------reindex for all entities in Human_SL and kg-----------------------------------------------
set_IDa = set(kg_delete['ID(a)'])
set_IDb = set(kg_delete['ID(b)'])
set_gene_a = set(human_SL['gene_a.identifier'])
set_gene_b = set(human_SL['gene_b.identifier'])
list_all = list(set_IDa | set_IDb | set_gene_a | set_gene_b)


print('The number of entities in kg and human_sl: ',len(list_all))

# reindex
entity_key = {}   
for i in range(len(list_all)):
    origin = list_all[i]
    entity_key[origin] = i

# reindex
for key in entity_key:
    kg_delete.loc[kg_delete['ID(a)']==key,'ID(a)'] = entity_key[key]
    kg_delete.loc[kg_delete['ID(b)']==key,'ID(b)'] = entity_key[key]
    human_SL.loc[human_SL['gene_a.identifier']==key,'gene_a.identifier'] = entity_key[key]
    human_SL.loc[human_SL['gene_b.identifier']==key,'gene_b.identifier'] = entity_key[key]

# print('Final ke_delete:')
print('The first 10 rows of kg after preprocessing:')
print(kg_delete[:10])
# print('The length of final ke_delete:', len(kg_delete))
print('The length of final kg:', len(kg_delete))
# print('Final human_sl:')
print('The first 10 rows of human_sl matrix after preprocessing:')
print(human_SL[:10])

# entity2id
entity2id = {
    "a":entity_key.keys(),
    "b":entity_key.values(),
}
entity2id = pd.DataFrame(entity2id)

# Save all files
kg_delete.to_csv(kg_save,index=False,header=None,sep='\t')
human_SL.to_csv(sl_save,index=False,header=None,sep='\t')
entity2id.to_csv(entity_save,index=False,sep='\t')
