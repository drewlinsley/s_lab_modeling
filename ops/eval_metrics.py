"""Metrics for evaluation of TF models."""
import tensorflow as tf


def metric_interpreter(metric, pred, labels):
    """Interpret string as a metric."""
    if metric == 'accuracy':
        return class_accuracy(
            pred=pred,
            labels=labels)
    elif metric == 'mean':
        return mean_score(
            pred=pred,
            labels=labels)
    elif metric == 'l2':
        return l2_score(
            pred=pred,
            labels=labels)
    elif metric == 'pearson':
        return tf.reduce_mean(
            pearson_score(
                pred=pred,
                labels=labels))
    else:
        raise RuntimeError('Cannot understand the dataset metric.')


def mean_score(pred, labels, force_dtype=tf.float32):
    """Accuracy with integer type."""
    if force_dtype:
        if pred.dtype != force_dtype:
            pred = tf.cast(pred, force_dtype)
        if labels.dtype != force_dtype:
            labels = tf.cast(labels, force_dtype)
    return tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32))


def class_accuracy(pred, labels):
    """Accuracy of 1/n*sum(pred_i == label_i)."""
    return tf.reduce_mean(
        tf.to_float(tf.equal(tf.argmax(pred, 1), tf.cast(
            labels, dtype=tf.int64))))


def l2_score(pred, labels):
    """L2 distance."""
    return tf.nn.l2_loss(pred - labels)


def tf_confusion_matrix(pred, labels):
    """Wrapper for calculating confusion matrix."""
    return tf.contrib.metrics.confusion_matrix(pred, labels)


def pearson_score(pred, labels, eps_1=1e-4, eps_2=1e-12):
    """Pearson loss function."""
    x_shape = [int(x) for x in pred.get_shape()]
    y_shape = [int(x) for x in labels.get_shape()]
    if len(x_shape) > 2:
        # Reshape tensors
        x1_flat = tf.contrib.layers.flatten(pred)
    else:
        # Squeeze off singletons to make x1/x2 consistent
        x1_flat = tf.squeeze(pred)
    if len(y_shape) > 2:
        x2_flat = tf.contrib.layers.flatten(labels)
    else:
        x2_flat = tf.squeeze(labels)
    x1_mean = tf.reduce_mean(x1_flat, keep_dims=True, axis=[-1]) + eps_1
    x2_mean = tf.reduce_mean(x2_flat, keep_dims=True, axis=[-1]) + eps_1

    x1_flat_normed = x1_flat - x1_mean
    x2_flat_normed = x2_flat - x2_mean

    count = int(x2_flat.get_shape()[-1])
    cov = tf.div(
        tf.reduce_sum(
            tf.multiply(
                x1_flat_normed, x2_flat_normed),
            -1),
        count)
    x1_std = tf.sqrt(
        tf.div(
            tf.reduce_sum(
                tf.square(x1_flat - x1_mean),
                -1),
            count))
    x2_std = tf.sqrt(
        tf.div(
            tf.reduce_sum(
                tf.square(x2_flat - x2_mean),
                -1),
            count))
    return cov / (tf.multiply(x1_std, x2_std) + eps_2)
