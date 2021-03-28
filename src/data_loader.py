import numpy as np
import os
import pandas as pd

def load_data(args):
    n_nodea, n_nodeb = load_sl2id(args)
    n_entity, n_relation, adj_entity, adj_relation = load_kg2id(args)


    print('data loaded.')
    print('n_nodea:',n_nodea)
    print('n_nodeb:', n_nodeb)
    print('n_entity:', n_entity)

    return n_nodea, n_nodeb, n_entity, n_relation, adj_entity, adj_relation


def load_sl2id(args):
    print('reading sl2id file ...')

    # reading sl2id file
    sl2id_file = '../data/sl2id'
    sl2id_np = np.loadtxt(sl2id_file + '.txt', dtype=np.int64)
    np.save(sl2id_file + '.npy', sl2id_np)

    # if os.path.exists(sl2id_file + '.npy'):
    #     sl2id_np = np.load(sl2id_file + '.npy')
    # else:
    #     sl2id_np = np.loadtxt(sl2id_file + '.txt', dtype=np.int64)
    #     np.save(sl2id_file + '.npy', sl2id_np)


    # if train on v1 test on v2
    if (args.trainv1_testv2):
        version2_notin_version1_file = '../data/version2_notin_version1'
        version2_notin_version1_np = np.loadtxt(version2_notin_version1_file + '.txt', dtype=np.int64)
        np.save(version2_notin_version1_file + '.npy', version2_notin_version1_np)

        # if os.path.exists(version2_notin_version1_file + '.npy'):
        #     version2_notin_version1_np = np.load(version2_notin_version1_file + '.npy')
        # else:
        #     version2_notin_version1_np = np.loadtxt(version2_notin_version1_file + '.txt', dtype=np.int64)
        #     np.save(version2_notin_version1_file + '.npy', version2_notin_version1_np)

        sl2id_np = np.concatenate((sl2id_np,version2_notin_version1_np),axis=0)

    n_nodea = len(set(sl2id_np[:, 0]))
    n_nodeb = len(set(sl2id_np[:, 1]))

    return n_nodea, n_nodeb

def load_kg2id(args):
    print('reading kg2id file ...')

    # reading kg2id file
    kg2id_file = '../data/kg2id'
    kg2id_np = np.loadtxt(kg2id_file + '.txt', dtype=np.int64)
    np.save(kg2id_file + '.npy', kg2id_np)

    # if os.path.exists(kg2id_file + '.npy'):
    #     kg2id_np = np.load(kg2id_file + '.npy')
    # else:
    #     kg2id_np = np.loadtxt(kg2id_file + '.txt', dtype=np.int64)
    #     np.save(kg2id_file + '.npy', kg2id_np)

    n_entity = len(set(kg2id_np[:, 0]) | set(kg2id_np[:, 2]))

    n_relation = len(set(kg2id_np[:, 1]))

    kg2dict = construct_kg2dict(kg2id_np)
    adj_entity, adj_relation = construct_adj(args, kg2dict, n_entity)

    return n_entity, n_relation, adj_entity, adj_relation


def construct_kg2dict(kg2id_np):
    print('constructing knowledge graph dict ...')
    kg2dict = dict()
    for triple in kg2id_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]

        if head not in kg2dict:
            kg2dict[head] = []
        kg2dict[head].append((tail, relation))

        # treat the KG as an undirected graph
        if tail not in kg2dict:
            kg2dict[tail] = []
        kg2dict[tail].append((head, relation))

        # # treat the KG as an undirected graph except relation #3 (GrG), 9(DuG),12(AuG),17(DdG),18(CuG),19(CdG),23(AdG) 并统计数量
        # if (relation == 4 | relation == 7):
        #     if tail not in kg2dict:
        #         kg2dict[tail] = []
        #     kg2dict[tail].append((head, relation))
        # # else:
        # #     if tail not in kg2dict:
        # #         kg2dict[tail] = []
        # #     kg2dict[tail].append((head, relation))


    return kg2dict

def construct_adj(args, kg2dict, n_entity):
    print('constructing adjacency matrix including entity and relation ...')
    # each line of adj_entity stores the sampled neighbor entities for a given entity
    # each line of adj_relation stores the corresponding sampled neighbor relations

    isolated_point = []
    adj_entity = np.zeros([n_entity, args.neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([n_entity, args.neighbor_sample_size], dtype=np.int64)
    for entity,entity_name in enumerate(kg2dict.keys()):
        if (entity in kg2dict.keys()):
            neighbors = kg2dict[entity]
        else:
            neighbors = [(entity,24)]
            isolated_point.append(entity)

        n_neighbors = len(neighbors)
        if n_neighbors >= args.neighbor_sample_size:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=True)
        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

    (pd.DataFrame(isolated_point)).to_csv('../results/isolated_point.csv')

    return adj_entity, adj_relation
