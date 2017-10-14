import os
import re
import sys
import tensorflow as tf
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
from ops.data_loader import inputs
from ops.tf_fun import make_dir, training_loop
from ops.loss_utils import softmax_loss, finetune_learning, wd_loss
from ops.metrics import class_accuracy
from config import clickMeConfig
from models import baseline_vgg16 as vgg16
from cluster_db import db


def batchnorm(layer):
    m, v = tf.nn.moments(layer, [0])
    return tf.nn.batch_normalization(layer, m, v, None, None, 1e-3)


def min_max(t):
    mint = tf.reduce_min(t, keep_dims=True)
    maxt = tf.reduce_max(t, keep_dims=True)
    return (t - mint) / (maxt - mint)


# Train or finetune a vgg16 while cuing to clickme
# Train or finetune a vgg16 while cuing to clickme
def train_vgg16(
        train_dir=None,
        validation_dir=None,
        hp_optim=False):
    config = clickMeConfig()
    if hp_optim:
        hp_params, hp_combo_id = db.get_parameters()
        if hp_combo_id is None:
            print 'Exiting.'
            sys.exit(1)
        for k, v in hp_params.iteritems():
            setattr(config, k, v)

    if train_dir is None:  # Use globals
        train_data = os.path.join(
            config.tf_record_base, config.tf_train_name)
        train_meta_name = os.path.join(
            config.tf_record_base,
            re.split('.tfrecords', config.tf_train_name)[0] + '_meta.npz')
        train_meta = np.load(train_meta_name)
    print 'Using train tfrecords: %s | %s image/heatmap combos' % (
            [train_data], len(train_meta['labels']))

    if validation_dir is None:  # Use globals
        validation_data = os.path.join(
            config.tf_record_base, config.tf_val_name)
        val_meta_name = os.path.join(
            config.tf_record_base,
            re.split('.tfrecords', config.tf_val_name)[0] + '_meta.npz')
        val_meta = np.load(val_meta_name)
        print 'Using validation tfrecords: %s | %s images' % (
            validation_data, len(val_meta['labels']))
    elif validation_dir is False:
        print 'Not using validation data.'
    else:
        validation_data = os.path.join(
            validation_dir, config.tf_val_name)
        val_meta_name = os.path.join(
            validation_dir,
            re.split('.tfrecords', config.tf_val_name)[0] + '_meta.npz')
        val_meta = np.load(val_meta_name)

    # Make output directories if they do not exist
    dt_stamp = 'baseline_' +\
        str(config.new_lr)[2:] + '_' + str(
            len(train_meta['labels'])) + '_' + re.split(
            '\.', str(datetime.now()))[0].\
        replace(' ', '_').replace(':', '_').replace('-', '_')
    config.train_checkpoint = os.path.join(
        config.train_checkpoint, dt_stamp)  # timestamp this run
    out_dir = os.path.join(config.results, dt_stamp)
    dir_list = [
        config.train_checkpoint, config.train_summaries,
        config.results, out_dir]
    [make_dir(d) for d in dir_list]

    print '-'*60
    print('Training model:' + dt_stamp)
    print '-'*60

    # Prepare data on CPU
    with tf.device('/cpu:0'):
        train_images, train_labels, train_heatmaps = inputs(
            train_data, config.train_batch, config.image_size,
            config.model_image_size[:2],
            train=config.data_augmentations,
            num_epochs=config.epochs,
            return_heatmaps=True)
        val_images, val_labels = inputs(
            validation_data, config.validation_batch, config.image_size,
            config.model_image_size[:2],
            num_epochs=None,
            return_heatmaps=False)

    # Prepare model on GPU
    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn') as scope:
            vgg = vgg16.Vgg16(
                vgg16_npy_path=config.vgg16_weight_path,
                fine_tune_layers=config.initialize_layers)
            train_mode = tf.get_variable(name='training', initializer=True)
            vgg.build(
                train_images,
                output_shape=config.output_shape,
                train_mode=train_mode,
                batchnorm=config.batchnorm_layers)

            # Prepare the loss function
            loss = softmax_loss(vgg.fc8, train_labels)

            # Add weight decay of fc6/7/8
            if config.wd_penalty is not None:
                loss = wd_loss(
                    loss=loss,
                    trainables=tf.trainable_variables(),
                    config=config)

            # Finetune the learning rates
            train_op = finetune_learning(
                loss,
                trainables=tf.trainable_variables(),
                config=config
                )

            train_accuracy = class_accuracy(
                vgg.prob, train_labels)  # training accuracy

            # Add summaries for debugging
            tf.summary.image('train images', train_images)
            tf.summary.image('validation images', val_images)
            tf.summary.scalar("loss", loss)
            tf.summary.scalar("training accuracy", train_accuracy)

            # Setup validation op
            if validation_data is not False:
                scope.reuse_variables()

                # Validation graph is the same as training except no batchnorm
                val_vgg = vgg16.Vgg16(
                    vgg16_npy_path=config.vgg16_weight_path,
                    fine_tune_layers=config.fine_tune_layers)
                val_vgg.build(val_images, output_shape=config.output_shape)

                # Calculate validation accuracy
                val_accuracy = class_accuracy(val_vgg.prob, val_labels)
                tf.summary.scalar("validation accuracy", val_accuracy)

    # Set up summaries and saver
    saver = tf.train.Saver(
        tf.global_variables(), max_to_keep=config.keep_checkpoints)
    summary_op = tf.summary.merge_all()

    # Initialize the graph
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # Need to initialize both of these if supplying num_epochs to inputs
    sess.run(tf.group(tf.global_variables_initializer(),
             tf.local_variables_initializer()))
    summary_dir = os.path.join(
        config.train_summaries, dt_stamp)
    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

    # Set up exemplar threading
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Start training loop
    np.save(os.path.join(out_dir, 'training_config_file'), config)
    training_loop(
        config,
        coord,
        sess,
        train_op,
        summary_op,
        summary_writer,
        loss,
        saver,
        threads,
        out_dir,
        summary_dir,
        validation_data,
        val_accuracy,
        train_accuracy,
        hp_optim=hp_optim)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--train_dir", type=str, dest="train_dir",
        default=None, help="Directory of training data tfrecords bin file.")
    parser.add_argument(
        "--validation_dir", type=str, dest="validation_dir",
        default=None, help="Directory of validation data tfrecords bin file.")
    parser.add_argument(
        "--hp_optim", dest="hp_optim",
        action='store_true', help="Turn this into a hp optimization worker.")
    args = parser.parse_args()
    train_vgg16(**vars(args))
