import tensorflow as tf
from abc import abstractmethod

LAYER_IDS = {}

"""
class Aggregator and SumAggregator refer to http://arxiv.org/abs/1905.04413 and https://dl.acm.org/doi/10.1145/3308558.3313417.
"""
def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Aggregator(object):
    def __init__(self, batch_size, dim, dropout, act, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations, nodea_embeddings, masks):
        outputs = self._call(self_vectors, neighbor_vectors, neighbor_relations, nodea_embeddings, masks)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, nodea_embeddings, masks):
        # dimension:
        # self_vectors: [batch_size, -1, dim] ([batch_size, -1] for LabelAggregator)
        # neighbor_vectors: [batch_size, -1, n_neighbor, dim] ([batch_size, -1, n_neighbor] for LabelAggregator)
        # neighbor_relations: [batch_size, -1, n_neighbor, dim]
        # nodea_embeddings: [batch_size, dim]
        # masks (only for LabelAggregator): [batch_size, -1]
        pass

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, nodea_embeddings):
        avg = False
        if not avg:
            # [batch_size, 1, 1, dim]
            nodea_embeddings = tf.reshape(nodea_embeddings, [self.batch_size, 1, 1, self.dim])

            # [batch_size, -1, n_neighbor]
            nodea_relation_scores = tf.reduce_mean(nodea_embeddings * neighbor_relations, axis=-1)
            nodea_relation_scores_normalized = tf.nn.softmax(nodea_relation_scores, dim=-1)

            # [batch_size, -1, n_neighbor, 1]
            nodea_relation_scores_normalized = tf.expand_dims(nodea_relation_scores_normalized, axis=-1)

            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(nodea_relation_scores_normalized * neighbor_vectors, axis=2)
        else:
            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)

        return neighbors_aggregated


class SumAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(SumAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, nodea_embeddings, masks):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, nodea_embeddings)

        # [-1, dim]
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)
