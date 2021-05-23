# KG4SL
Synthetic lethality (SL) is a promising gold mine for the discovery of anti-cancer drug targets.
KG4SL is a novel graph neural network (GNN)-based model, by incorporating knowledgegraph message-passing into SL prediction. The knowledge graph was constructed using 11 kinds of entities including genes,compounds, diseases, biological processes, and 24 kinds of relationships that could be pertinent to SL. The integration of knowledge graph can help harness the independence issue and circumvent manual feature engineering by conducting message-passing on the knowledge graph.
## Dataset collection
    The data used to train and test the KG4SL is downloaded from a comprehensive database of synthetic lethal gene pairs named SyLethDB (http://synlethdb.sist.shanghaitech.edu.cn/v2/#/). 
    Its latest version includes a set of 36,402 human SL pairs, as well as a knowledge graph (KG) with 11 kinds of entities and 24 kinds of relationships. The details of the data refer to the paper 'KG4SL: Knowledge Graph Neural Network for Synthetic Lethality Prediction in Human Cancers'. 
    Here we listed the information of the SL pairs and knowledge graph.
   ![image](https://github.com/JieZheng-ShanghaiTech/KG4SL/blob/main/table1.png)
   ![image](https://github.com/JieZheng-ShanghaiTech/KG4SL/blob/main/table2.png)
   
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
        3. input dataset for train.py
            dbid2name: _id, name

    > results
        eval_data_final_1_X: dataset used for validation
        test_data_final_1: dataset used for test
        test_data_mapping_final_1: dataset uesd for test with gene names
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
    pandas 1.1.5
    scikit-learn 0.24.0
    matplotlib 3.3.3
    
    note: You can install all the packages through the command 'pip install -r pip_install.txt'.
 
 ## Supplementary:
    We've uploaded some additional experiments in supplementray file.
    
