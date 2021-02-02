for i in range(rating_np.shape[0]):
item_index_old = array[1]
if item_index_old not in item_index_old2new:  # the item is not in the final item set
    continue
item_index = item_index_old2new[item_index_old]  # 找到rating里的item id对应的新的index赋值给item_index

adj_entity = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
adj_relation = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
for entity, entity_name in enumerate(kg.keys()):
    neighbors = kg[entity]
    n_neighbors = len(neighbors)
    if n_neighbors >= args.neighbor_sample_size:
        sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=False)
    else:
        sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=True)
    adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
    adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])


    kg = dict()
    for triple in kg_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
    return kg