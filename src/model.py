import tensorflow as tf
from aggregators import SumAggregator
from sklearn.metrics import f1_score, roc_auc_score,precision_recall_curve
import sklearn.metrics as m
import pandas as pd
import copy
import numpy as np

"""
class KG4SL is a modification of http://arxiv.org/abs/1905.04413.
"""
class KG4SL(object):
    def __init__(self, args, n_entity, n_relation, adj_entity, adj_relation):
        self._parse_args(args, adj_entity, adj_relation)
        self._build_inputs()
        self._build_model(n_entity, n_relation)
        self._build_train()

    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def _parse_args(self, args, adj_entity, adj_relation):
        self.adj_entity = adj_entity # [entity_num, neighbor_sample_size]
        self.adj_relation = adj_relation
        self.n_hop = args.n_hop
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr

    def _build_inputs(self):
        self.nodea_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='nodea_indices')
        self.nodeb_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='nodeb_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

    def _build_model(self, n_entity, n_relation):
        self.entity_emb_matrix = tf.get_variable(shape=[n_entity, self.dim], initializer=KG4SL.get_initializer(), name='entity_emb_matrix')
        self.relation_emb_matrix = tf.get_variable(shape=[n_relation, self.dim], initializer=KG4SL.get_initializer(), name='relation_emb_matrix')

        # [batch_size, dim]
        nodea_embeddings_initial = tf.nn.embedding_lookup(self.entity_emb_matrix, self.nodea_indices)
        nodeb_embeddings_initial = tf.nn.embedding_lookup(self.entity_emb_matrix, self.nodeb_indices)

        nodea_entities, nodea_relations = self.get_neighbors(self.nodea_indices)
        nodeb_entities, nodeb_relations = self.get_neighbors(self.nodeb_indices)

        # [batch_size, dim]
        self.nodea_embeddings, self.nodea_aggregators = self.aggregate(nodea_entities, nodea_relations, nodeb_embeddings_initial)
        self.nodeb_embeddings, self.nodeb_aggregators = self.aggregate(nodeb_entities, nodeb_relations, nodea_embeddings_initial)

        # [batch_size]
        self.scores = tf.reduce_sum(self.nodea_embeddings * self.nodeb_embeddings, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def get_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_hop):
            neighbor_entities = tf.reshape(tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1])
            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    # feature propagation
    def aggregate(self, entities, relations, embeddings_agg):
        aggregators = []  # store all aggregators
        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]
        embeddings_aggregator = embeddings_agg

        for i in range(self.n_hop):
            if i == self.n_hop - 1:
                aggregator = SumAggregator(self.batch_size, self.dim, act=tf.nn.tanh)
            else:
                aggregator = SumAggregator(self.batch_size, self.dim)
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.n_hop - i):
                shape = [self.batch_size, -1, self.n_neighbor, self.dim]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                                    nodea_embeddings=embeddings_aggregator,
                                    masks=None)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])


        return res, aggregators

    # loss
    def _build_train(self):
        # base loss
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        # L2 loss
        self.l2_loss = tf.nn.l2_loss(
            self.entity_emb_matrix) + tf.nn.l2_loss(self.relation_emb_matrix)

        for aggregator in self.nodeb_aggregators:
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)

        for aggregator in self.nodea_aggregators:
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)

        self.loss = self.base_loss + self.l2_weight * self.l2_loss

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        nodea_emb, nodeb_emb = sess.run([self.nodea_embeddings, self.nodeb_embeddings], feed_dict)
        scores_output = copy.deepcopy(scores)

        auc = roc_auc_score(y_true=labels, y_score=scores)
        p, r, t = precision_recall_curve(y_true=labels, probas_pred=scores)
        aupr = m.auc(r, p)

        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0

        scores_binary_output = scores

        f1 = f1_score(y_true=labels, y_pred=scores)
        return nodea_emb, nodeb_emb, scores_output, scores_binary_output, auc, f1, aupr

    def get_scores(self, sess, feed_dict):
        return sess.run([self.nodeb_indices, self.scores_normalized], feed_dict)

