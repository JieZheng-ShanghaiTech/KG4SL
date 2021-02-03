# KG4SL
## Files in the folder: 
    > data
        1. input datasets for prepocessing.py
            kg_triplet: head, relation, tail
            relation2id: relation_name, relation_id
            sl_data: gene_a, gene_b
        2. input datasets for main.py (The datasets below are generated through preprocessing.py.)
            sl2id: gene_a_id, gene_b_id, 0/1
            kg2id: head_id, relation_id, tail_id
            entity2id: origin_id, new_id
    > results
        eval_data_final_1_X: dataset used for validation
        test_data_final_1: dataset used for test
        train_data_final_1_X: dataset used for train
        loss_curve_final_1_X: save the values of losses and three metrics within the increase of epochs
        training_curve_final_1_X: training curve curve (It is recommended that you turn off the early stop mechanism while getting the training curve.)
        
        note: 
        The first number in the file naming process represents the process of retrieving test data, which is partitioned only once in this article. 
        The second number that appears represents the process of dividing train data and validation data, which is repeated five times in this article. 
        The third number that appears indicates that in n_epoch, the optimal result appears for the X time.
    > src
        implementations of KG4SL
    
## Running the code:
    cd src
    python preprocessing.py
    python main.py
    
## Requirements:
    python36
    tensorflow-gpu 1.15.0
    torch 1.1.0
    dgl 0.5.2
