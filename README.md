# KG4SL
Synthetic lethality (SL) is a promising gold mine for the discovery of anti-cancer drug targets.
KG4SL is a novel graph neural network (GNN)-based model, by incorporating knowledgegraph message-passing into SL prediction. The knowledge graph was constructed using 11 kinds of entities including genes,compounds, diseases, biological processes, and 24 kinds of relationships that could be pertinent to SL. The integration of knowledge graph can help harness the independence issue and circumvent manual feature engineering by conducting message-passing on the knowledge graph.

## Dataset collection
The data used to train and test the KG4SL is downloaded from a comprehensive database of synthetic lethal gene pairs named SynLethDB (http://synlethdb.sist.shanghaitech.edu.cn/v2/#/). Its latest version includes a set of 36,402 human SL pairs, as well as a knowledge graph (KG) with 11 kinds of entities and 24 kinds of relationships. And the knowledge graph named SynLethKG that passes message into SL prediction is constructed based on `SynLethDB` and `Hetionet`. The details of the data refer to the paper 'KG4SL: Knowledge Graph Neural Network for Synthetic Lethality Prediction in Human Cancers'. Here we listed the information of the SL pairs and knowledge graph.
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
 ## Acknowledgments:
 The code is inspired by [KGNN-LS](https://github.com/hwwang55/KGNN-LS).
 
 >[Knowledge-aware Graph Neural Networks with Label Smoothness Regularization for Recommender Systems](https://dl.acm.org/doi/abs/10.1145/3292500.3330836)  
 Wang, Hongwei, et al. "Knowledge-aware graph neural networks with label smoothness regularization for recommender systems." Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining. 2019.
 
 The SynLethKG is provided by [SynLethDB 1.0](http://synlethdb.sist.shanghaitech.edu.cn/), [SynLethDB 2.0](http://synlethdb.sist.shanghaitech.edu.cn/v2/#/) and [Hetionet](https://github.com/hetio/hetionet).
 
 >[SynLethDB: synthetic lethality database toward discovery of selective and sensitive anticancer drug targets](https://academic.oup.com/nar/article/44/D1/D1011/2502617?login=true)  
Guo, Jing, Hui Liu, and Jie Zheng. "SynLethDB: synthetic lethality database toward discovery of selective and sensitive anticancer drug targets." Nucleic acids research 44.D1 (2016): D1011-D1017.

>[Systematic integration of biomedical knowledge prioritizes drugs for repurposing](https://elifesciences.org/articles/26726)  
Himmelstein, Daniel Scott, et al. "Systematic integration of biomedical knowledge prioritizes drugs for repurposing." Elife 6 (2017): e26726.
 
## References:
```
@article{wang2021kg4sl,
  title={KG4SL: knowledge graph neural network for synthetic lethality prediction in human cancers},
  author={Wang, Shike and Xu, Fan and Li, Yunyang and Wang, Jie and Zhang, Ke and Liu, Yong and Wu, Min and Zheng, Jie},
  journal={Bioinformatics},
  volume={37},
  number={Supplement\_1},
  pages={i418--i425},
  year={2021},
  publisher={Oxford University Press}
}
```
