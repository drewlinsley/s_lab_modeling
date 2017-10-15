"""Training routine for TF."""
import os
import time
import cv2
import numpy as np
from datetime import datetime


def pad_im(img, crop_size):
    "Pad image if necessary."""
    x, y = img.shape[:2]
    cx, cy = crop_size
    startx = x - cx
    starty = y - cy
    padx = np.min([startx, 0])
    pady = np.min([starty, 0])
    if padx < 0 or pady < 0:
        padx = np.abs(padx)
        pady = np.abs(pady)
        t = padx // 2
        b = padx - t
        l = pady // 2
        r = pady - l
        img = cv2.copyMakeBorder(
            img,
            top=t,
            bottom=b,
            left=l,
            right=r,
            borderType=cv2.BORDER_CONSTANT,
            value=0.)
    return img


def crop_center(img, crop_size):
    """Center crop images."""
    h, w = img.shape[:2]
    ch, cw = crop_size
    starth = h // 2 - (ch // 2)
    startw = w // 2 - (cw // 2)
    return img[starth:starth + ch, startw:startw + cw, :]


def renormalize(img, max_value, min_value):
    """Normalize images to [0, 1]."""
    return (img - min_value) / (max_value - min_value)


def preprocess_images(ims, target_size):
    """Center crop images."""
    out = []
    for im in ims:
        im = pad_im(im, target_size)
        out += [crop_center(im, target_size)]
    return np.asarray(out)


def check_early_stop(
        perf_history,
        minimum_length=20,
        short_history=3,
        long_history=5,
        fail_function=np.less_equal):
    """
    Determine whether to stop early.

    We determine this point by comparing the performance from
    the last three epochs to the performance five epochs before those.
    Training runs for at least 20 epochs, and is terminated if all three
    of the last epochs show decreased performance or if
    800 epochs are reached.

    """
    if len(perf_history) < minimum_length:
        early_stop = False
    else:
        short_perf = perf_history[-short_history:]
        long_perf = perf_history[-long_history + short_history:short_history]
        short_check = fail_function(np.mean(long_perf), short_perf)
        if all(short_check):  # If we should stop
            early_stop = True
        else:
            early_stop = False

    return early_stop


def cv_interpret(X, y, cv):
    """Prepare CV indices."""
    cv_type = cv.keys()[0]
    cv_val = cv.values()[0]
    if cv_type == 'k_fold':
        obs = len(y)
        rep_val = obs / cv_val
        assert rep_val == np.round(rep_val),\
            'Choose a k-fold that is evenly divisible with %s.' % obs
        cv_seed = np.eye(cv_val)
        folds = [
            np.expand_dims(x, -1).repeat(rep_val, axis=-1)
            for x in cv_seed]
    elif cv_type == 'hold_out':
        obs = len(y)
        train_len = np.round(obs * cv_val).astype(int)
        val_len = obs - train_len
        folds = [np.concatenate(
            (np.zeros((train_len)),
            np.ones((val_len))), axis=0)]
    else:
        raise NotImplementedError

    # Prepare CV data
    out_folds = []
    for fold in folds:
        fold = fold.squeeze()
        it_val = {}
        val_X = X[fold == 1]
        val_y = y[fold == 1]
        it_val['X'] = val_X
        it_val['y'] = val_y
        it_train = {}
        train_X = X[fold == 0]
        train_y = y[fold == 0]
        it_train['X'] = train_X
        it_train['y'] = train_y
        out_folds += [{
            'val': it_val,
            'train': it_train
        }]
    return out_folds


def training_loop(
        config,
        neural_data,
        images,
        target_size,
        sess,
        train_vars,
        val_vars,
        summary_op,
        summary_writer,
        checkpoint_dir,
        summary_dir,
        saver):
    """Run a training loop."""
    step, time_elapsed = 0, 0
    train_losses, train_accs, timesteps = {}, {}, {}
    val_losses, val_accs = {}, {}
    start_time = time.time()

    # Prepare crossval
    cv_folds = cv_interpret(
        X=images,
        y=neural_data,
        cv=config.cv)
    num_folds = len(cv_folds)
    train_cv_out, val_cv_out, weights = [], [], []
    for idx, fold in enumerate(cv_folds):
        train_data = fold['train']
        val_data = fold['val']
        train_num_steps = len(train_data['y'])
        val_num_steps = len(val_data['y'])
        fold_train_losses, fold_train_scores, fold_val_losses, fold_val_scores = {}, {}, {}, {}
        for epoch in range(config.num_epochs):
            print 'Starting epoch %s/%s of fold %s/%s' % (
                epoch,
                config.num_epochs,
                idx + 1,
                num_folds)
            image_count = 0
            fold_train_losses[epoch] = []
            fold_train_scores[epoch] = []
            fold_val_losses[epoch] = []
            fold_val_scores[epoch] = []
            # Training loop
            while image_count < train_num_steps:
                it_inds = np.random.permutation(train_num_steps)[:config.batch_size]
                it_images = preprocess_images(train_data['X'][it_inds], target_size)
                it_neural_data = train_data['y'][it_inds]
                feed_dict = {
                    train_vars['images']: it_images,
                    train_vars['neural_data']: it_neural_data     
                }
                train_data = sess.run(
                    train_vars.values(),
                    feed_dict=feed_dict)
                train_dict = {k: v for k, v in zip(train_vars.keys(), train_data)}
                import ipdb; ipdb.set_trace()
                assert not np.any(np.isnan(train_dict['loss'])), 'NaN in loss.'
                image_count += config.batch_size
                fold_train_losses[epoch] += [train_dict['loss']]
                fold_train_scores[epoch] += [train_dict['score']]
                duration = time.time() - start_time
                status_string = ('%s, step: %s, training loss: %s, ',
                    'training score: %s, log dir: %s, '
                    '(%.1f examples/sec, %.3f sec/batch)' % (
                        datetime.now(),
                        image_count // config.batch_size,
                        float(train_dict['loss']),
                        float(train_dict['score']),
                        summary_dir,
                        config.batch_size / duration,
                        float(duration)
                    ))
                print status_string
            # Run validation after every epoch
            image_count = 0
            while image_count < val_num_steps:
                it_inds = np.random.permutation(val_num_steps)[:config.batch_size]
                it_images = preprocess_images(val_data['X'][it_inds], target_size)
                it_neural_data = val_data['y'][it_inds]
                feed_dict = {
                    train_vars['images']: it_images,
                    train_vars['neural_data']: it_neural_data
                }
                val_data = sess.run(
                    val_vars.values(),
                    feed_dict=feed_dict)
                val_dict = {k: v for k, v in zip(val_vars.keys(), val_data)}
                image_count += config.batch_size
                fold_val_losses[epoch] += [val_dict['loss']]
                fold_val_scores[epoch] += [val_dict['score']]
                status_string = ('VALIDATION: %s, step: %s, val loss: %s, ',
                    'val score: %s, ckpt dir: %s, '
                    '(%.1f examples/sec, %.3f sec/batch)' % (
                        datetime.now(),
                        image_count // config.batch_size,
                        float(val_dict['loss']),
                        float(val_dict['score']),
                        ckpt_dir,
                        config.batch_size / duration,
                        float(duration)
                    ))
                print status_string
        train_cv_out += [
            {
                'losses': fold_train_losses,
                'scores': fold_train_scores
            }]
        val_cv_out += [
            {
                'losses': fold_val_losses,
                'scores': fold_val_scores
            }]
        if config.cm_wd_types is not None:
            weights += [{k: train_dict[k] for k in cm_wd_types.keys()}]
    return train_cv_out, val_cv_out, weights
