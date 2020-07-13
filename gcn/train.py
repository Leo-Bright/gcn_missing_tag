from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP

# Set random seed
seed = 123
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'sanfrancisco', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
# flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

# count = 0
# for item in y_test.tolist():
#     if item == 1:
#         count += 1
# print(count)

# Some preprocessing
features = preprocess_features(features)  # calculate D^-1 * A ,as tupled(coords, values, shape)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]  # (D + I)^-1/2 * (A + I) * (D + I)^-1/2
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.compat.v1.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.compat.v1.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.compat.v1.placeholder(tf.int32),
    'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.compat.v1.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.compat.v1.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs, model.placeholders, model.activations], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_f1=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")


def save_emb_to_file(emb_vector, idx2node_dict_pkl_path, emb_idx_pkl_path, emb_file_path):

    with open(idx2node_dict_pkl_path, "rb") as f:
        idx2node = pkl.load(f)

    with open(emb_idx_pkl_path, "rb") as f:
        emb_idx = pkl.load(f)

    assert len(emb_idx) == len(emb_vector)

    with open(emb_file_path, "w") as f:
        for i in range(len(emb_idx)):
            node_id = idx2node[emb_idx[i]]
            emb = emb_vector[i]
            emb_str = map(str, emb)
            f.write(node_id + ' ' + ' '.join(emb_str) + '\n')


# save gcn embeddings to file
out_activations = outs[5]
last_layer_activation = out_activations[2]
gcn_emb_file_path = 'sanfrancisco/embeddings/sf_gcn_raw_feature_none_16d_target_is_crossing.embedding'
gcn_emb_idx_pkl_path = 'sanfrancisco/embeddings/sf_gcn_raw_feature_none_16d_target_is_crossing.embedding.idx.pkl'
idx_node_dict_pkl_path = 'sanfrancisco/sf_idx_node_dict.pkl'
save_emb_to_file(last_layer_activation, idx_node_dict_pkl_path, gcn_emb_idx_pkl_path, gcn_emb_file_path)
print("Embeddings Saved to " + gcn_emb_file_path + ' !')


# Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "f1_score=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))