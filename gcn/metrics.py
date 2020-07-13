import tensorflow as tf


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    mask = tf.cast(mask, dtype=tf.int64)
    sample_size = tf.count_nonzero(mask)

    preds_negative = tf.argmax(preds, 1)
    preds_negative *= mask
    tp_and_fp = sample_size - tf.count_nonzero(preds_negative)
    tn_and_fn = sample_size - tp_and_fp

    labels_negative = tf.argmax(labels, 1)
    labels_negative *= mask
    tn = tf.count_nonzero(preds_negative * labels_negative)

    preds_positive = tf.argmin(preds, 1)
    preds_negative *= mask
    labels_positive = tf.argmin(labels, 1)
    labels_positive *= mask
    tp = tf.count_nonzero(preds_positive * labels_positive)

    precision = tp / tp_and_fp
    recall = tp / (tp + tn_and_fn - tn)

    f1_score = 2 * precision * recall / (precision + recall)

    return f1_score

    # correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    # accuracy_all = tf.cast(correct_prediction, tf.float32)
    # mask = tf.cast(mask, dtype=tf.float32)
    # mask /= tf.reduce_mean(mask)
    # accuracy_all *= mask
    # return tf.reduce_mean(accuracy_all)
