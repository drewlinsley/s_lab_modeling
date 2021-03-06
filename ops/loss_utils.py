"""Functions for losses and metrics."""
import numpy as np
import tensorflow as tf


def add_wd(weights, wd_dict):
    """Derive weight decay regularizations."""
    losses = []
    for k, v in wd_dict.iteritems():
        if k in weights.keys():
            wd_type = v['type']
            wd_strength = v['strength']
            print 'Adding %s weight decay, lambda = %s, for: %s' % (
                wd_type,
                wd_strength,
                k)
            if wd_type == 'l2':
                losses += [tf.nn.l2_loss(weights[k]) * wd_strength]
            elif wd_type == 'laplace_l2':
                losses += [tf.nn.l2_loss(laplacian(weights[k])) * wd_strength]
            elif wd_type == 'laplace_l1':
                losses += [l1_loss(laplacian(weights[k])) * wd_strength]
            elif wd_type == 'l1':
                losses += [l1_loss(weights[k]) * wd_strength]
            else:
                raise NotImplementedError(k)
    return tf.add_n(losses)


def l1_loss(x):
    """L1 loss: sum(abs(x))."""
    return tf.reduce_sum(tf.abs(x))


def laplacian(weights, ltype=4):
    """Convolution with the Laplacian of weights."""
    if ltype == 4:
        l = np.asarray(
            [
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ]
        )
    elif ltype == 8:
        l = np.asarray(
            [
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]
            ]
        )
    else:
        raise NotImplementedError(ltype)
    kernel = l[:, :, None, None]
    return tf.nn.conv2d(weights, kernel, [1, 1, 1, 1], padding='SAME')


def optimizer_interpreter(
        loss,
        lr,
        optimizer):
    """Router for loss functions."""
    if optimizer == 'adam':
        return tf.train.AdamOptimizer(lr).minimize(loss)
    elif optimizer == 'sgd':
        return tf.train.GradientDescentOptimizer(lr).minimize(loss)
    elif optimizer == 'momentum':
        return momentum(loss=loss, lr=lr)
    elif optimizer == 'rmsprop':
        return tf.train.RMSPropOptimizer(lr).minimize(loss)
    else:
        raise RuntimeError('Cannot understand your loss function.')


def loss_interpreter(
        logits,
        labels,
        loss_type,
        weights=None,
        max_spikes=None,
        dataset_module=None):
    """Router for loss functions."""
    if loss_type is None:
        loss_type = dataset_module.default_loss_function
    if loss_type == 'cce':
        "Return average CCE loss per channel."""
        slabels = tf.split(labels, int(labels.get_shape()[-1]), axis=1)
        import ipdb;ipdb.set_trace()
        slabels = tf.split(logits, int(logits.get_shape()[-1]), axis=2)
        return cce(
            logits=logits,
            labels=labels,
            weights=weights)
    elif loss_type == 'l2':
        return l2(
            logits=logits,
            labels=labels)
    elif loss_type == 'l1':
        return l1(
            logits=logits,
            labels=labels)
    elif loss_type == 'log_loss':
        return log_loss(
            logits=logits,
            labels=labels)
    elif loss_type == 'huber':
        return huber(
            logits=logits,
            labels=labels,
            weights=weights)
    elif loss_type == 'sigmoid':
        return sigmoid_ce(
            logits=logits,
            labels=labels,
            weights=weights)
    elif loss_type == 'pearson':
        return pearson_loss(
            logits=logits,
            labels=labels)
    else:
        raise RuntimeError('Cannot understand your loss function.')


def cce(logits, labels, weights=None):
    """Sparse categorical cross entropy with weights."""
    if weights is not None:
        weights = tf.get_variable(
            name='weights', initializer=weights)[None, :]
        weights_per_label = tf.matmul(
            tf.one_hot(labels, 2), tf.transpose(tf.cast(weights, tf.float32)))
        return tf.reduce_mean(
            tf.multiply(
                weights_per_label,
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels))), tf.nn.softmax(logits)
    else:
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels)), tf.nn.softmax(logits)


def momentum(loss, lr, momentum=0.9):
    """Wrapper for SGD with momentum."""
    tf.train.MomentumOptimizer(lr, momentum=momentum).minimize(loss)


def l2(logits, labels):
    """Wrapper for l2 loss."""
    l2_loss = tf.nn.l2_loss(
        logits - labels)
    return l2_loss, l2_loss


def l1(logits, labels):
    """Wrapper for l2 loss."""
    l1_loss = tf.reduce_sum(
        tf.abs(logits - labels))
    return l1_loss, l1_loss


def huber(logits, labels, weights):
    """Wrapper for huber loss."""
    logits = tf.squeeze(logits)
    labels = tf.squeeze(labels)
    if weights is None:
        weights = 1.
    return tf.losses.huber_loss(
        predictions=logits,
        labels=labels,
        weights=weights), tf.nn.l2_loss(logits - labels)


def log_loss(logits, labels):
    """Wrapper for log loss."""
    logits = tf.squeeze(logits)
    labels = tf.squeeze(labels)
    ll = tf.losses.log_loss(
        predictions=logits,
        labels=labels)
    return ll, ll


def pearson_loss(logits, labels):
    """Pearson dissimilarity loss."""
    rhos = 1 - pearson_score(x1=logits, x2=labels)
    mean_rhos = tf.reduce_mean(rhos)
    return mean_rhos, rhos


def sigmoid_ce(logits, labels, weights, force_dtype=tf.float32):
    """Wrapper for sigmoid cross entropy loss."""
    if force_dtype:
        if logits.dtype != force_dtype:
            logits = tf.cast(logits, force_dtype)
        if labels.dtype != force_dtype:
            labels = tf.cast(labels, force_dtype)
    if weights is None:
        weights = 1.
    sig_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels,
            logits=logits) * weights)
    return sig_loss, sig_loss


def pearson_score(x1, x2, shape=None):
    """Pearson correlation."""
    x1_flat = tf.contrib.layers.flatten(x1)
    x2_flat = tf.contrib.layers.flatten(x2)

    x1_mean = tf.reduce_mean(x1_flat, keep_dims=True, axis=[-1]) + 1e-8
    x2_mean = tf.reduce_mean(x2_flat, keep_dims=True, axis=[-1]) + 1e-8

    x1_flat_normed = x1_flat - x1_mean
    x2_flat_normed = x2_flat - x2_mean

    count = int(x2_flat.get_shape()[-1])
    cov = tf.div(
        tf.reduce_sum(
            tf.multiply(x1_flat_normed, x2_flat_normed), -1), count)
    x1_std = tf.sqrt(
        tf.div(
            tf.reduce_sum(tf.square(x1_flat - x1_mean), -1), count))
    x2_std = tf.sqrt(
        tf.div(
            tf.reduce_sum(tf.square(x2_flat - x2_mean), -1), count))

    corr = cov / (tf.multiply(x1_std, x2_std) + 1e-4)
    return corr

