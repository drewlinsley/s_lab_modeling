"""Extract VGG features and fit to neural data."""
import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from config import Config
from ops import loss_utils, eval_metrics, rf_sizes, training
from layers import normalizations, ff
from models import baseline_vgg16 as vgg16
from utils import py_utils
from argparse import ArgumentParser


def extract_vgg_features(
        cm_type='contextual_vector',
        layer_name='pool3',
        output_type='sparse_pool',
        project_name=None,
        model_type='vgg16',
        timesteps=5,
        dtype=tf.float32):
    """Main extraction and training script."""
    assert project_name is not None, 'Need a project name.'

    # 1. Get file paths and load config
    config = Config()
    project_path = config.projects[project_name]

    # 2. Assert the model is there and load neural data.
    print 'Loading preprocessed data...'
    data = np.load(
        os.path.join(
            project_path,
            '%s.npz' % project_name))
    neural_data = data['data_matrix']
    # TODO: across_session_data_matrix is subtracted version
    images = data['all_images'].astype(np.float32)
    # TODO: create AUX dict with each channel's X/Y
    output_aux = None
    rfs = rf_sizes.get_eRFs(model_type)[layer_name]

    # 3. Create a output directory if necessary and save a timestamped numpy.
    dt_stamp = '%s' % str(datetime.now())[0].replace(
        ' ', '_').replace(
        ':', '_').replace(
        '-', '_')
    out_dir = os.path.join(
        config.results,
        dt_stamp)
    checkpoint_dir = os.path.join(
        out_dir,
        'checkpoints')
    out_file = os.path.join(
        out_dir,
        dt_stamp,
        'data')
    dirs = [
        config.results,
        config.summaries,
        out_dir
    ]
    [py_utils.make_dir(x) for x in dirs]
    print '-' * 60
    print('Training model:' + out_file)
    print '-' * 60

    # 4. Prepare data on CPU
    neural_shape = list(neural_data.shape)
    num_neurons = neural_shape[-1]
    with tf.device('/cpu:0'):
        train_images = tf.placeholder(
            dtype=dtype,
            name='train_images',
            shape=[config.batch_size] + config.img_shape)
        train_neural = tf.placeholder(
            dtype=dtype,
            name='train_neural',
            shape=[config.batch_size] + [num_neurons])
        val_images = tf.placeholder(
            dtype=dtype,
            name='val_images',
            shape=[config.batch_size] + config.img_shape)
        val_neural = tf.placeholder(
            dtype=dtype,
            name='val_neural',
            shape=[config.batch_size] + [num_neurons])

    # 5. Prepare model on GPU
    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn') as scope:
            vgg = vgg16.Vgg16(
                vgg16_npy_path=config.vgg16_weight_path)
            train_mode = tf.get_variable(name='training', initializer=False)
            vgg.build(
                train_images,
                output_shape=1000,  # hardcode
                train_mode=train_mode,
                final_layer=layer_name)

            # Select a layer
            activities = vgg[layer_name]

            # Feature reduce with a 1x1 conv
            if config.reduce_features is not None:
                vgg, activities, reduce_weights = ff.pool_ff_interpreter(
                    self=vgg,
                    it_neuron_op='1x1conv',
                    act=activities,
                    it_name='feature_reduce',
                    out_channels=config.reduce_features,
                    aux=None)
            else:
                reduce_weights = None
            

            # Add con-model if requested
            if cm_type is not None or cm_type != 'none':
                norms = normalizations.normalizations()
                activities, cm_weights, _ = norms[cm_type](
                    x=activities,
                    r_in=rfs['r_in'],
                    j_in=rfs['j_in'],
                    timesteps=timesteps,
                    lesions=config.lesions)
            else:
                cm_weights = None

            # Create output layer for N-recording channels
            activities = tf.nn.dropout(activities, 0.5)
            vgg, output_activities, output_weights = ff.pool_ff_interpreter(
                self=vgg,
                it_neuron_op=output_type,
                act=activities,
                it_name='output',
                out_channels=num_neurons,
                aux=output_aux)

            # Prepare the loss function
            loss, _ = loss_utils.loss_interpreter(
                logits=output_activities,
                labels=train_neural,
                loss_type=config.loss_type)

            # Add contextual model WD
            if config.reduce_features is not None and reduce_weights is not None:
                loss += loss_utils.add_wd(
                    weights=reduce_weights,
                    wd_dict=config.wd_types)

            # Add contextual model WD
            if config.cm_wd_types is not None and cm_weights is not None:
                loss += loss_utils.add_wd(
                    weights=cm_weights,
                    wd_dict=config.cm_wd_types)

            # Add WD to output layer
            if config.wd_types is not None:
                loss += loss_utils.add_wd(
                    weights=output_weights,
                    wd_dict=config.wd_types)

            # Finetune the learning rates
            train_op = loss_utils.optimizer_interpreter(
                loss=loss,
                lr=config.lr,
                optimizer=config.optimizer)

            # Calculate metrics
            train_accuracy = eval_metrics.metric_interpreter(
                metric=config.metric,
                pred=output_activities,
                labels=train_neural)

            # Add summaries for debugging
            tf.summary.image('train images', train_images)
            tf.summary.image('validation images', val_images)
            tf.summary.scalar("loss", loss)
            tf.summary.scalar("training accuracy", train_accuracy)

            # Setup validation op
            scope.reuse_variables()

            # Validation graph is the same as training except no batchnorm
            val_vgg = vgg16.Vgg16(
                vgg16_npy_path=config.vgg16_weight_path)
            val_vgg.build(
                val_images,
                output_shape=1000,
                final_layer=layer_name)

            # Select a layer
            val_activities = val_vgg[layer_name]

            # Add feature reduction if requested
            if config.reduce_features is not None:
                val_vgg, val_activities, _ = ff.pool_ff_interpreter(
                    self=val_vgg,
                    it_neuron_op='1x1conv',
                    act=val_activities,
                    it_name='feature_reduce',
                    out_channels=config.reduce_features,
                    aux=None)
            else:
                reduce_weights = None

            # Add con-model if requested
            if cm_type is not None or cm_type != 'none':
                val_activities, _, _ = norms[cm_type](
                    x=val_activities,
                    r_in=rfs['r_in'],
                    j_in=rfs['j_in'],
                    timesteps=timesteps,
                    lesions=config.lesions)

            # Create output layer for N-recording channels
            val_vgg, val_output_activities, _ = ff.pool_ff_interpreter(
                self=val_vgg,
                it_neuron_op=output_type,
                act=val_activities,
                it_name='output',
                out_channels=num_neurons,
                aux=output_aux)

            # Prepare the loss function
            val_loss, _ = loss_utils.loss_interpreter(
                logits=val_output_activities,
                labels=val_neural,
                loss_type=config.loss_type)

            # Calculate metrics
            val_accuracy = eval_metrics.metric_interpreter(
                metric=config.metric,
                pred=val_output_activities,
                labels=val_neural)
            tf.summary.scalar('validation loss', val_loss)
            tf.summary.scalar('validation accuracy', val_accuracy)

    # Set up summaries and saver
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    # Initialize the graph
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # Need to initialize both of these if supplying num_epochs to inputs
    sess.run(tf.global_variables_initializer())
    summary_dir = os.path.join(
        config.summaries,
        dt_stamp)
    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

    # Start training loop
    train_vars = {
        'images': train_images,
        'neural_data': train_neural,
        'loss': loss,
        'score': train_accuracy,
        'train_op': train_op,
        'weights': cm_weights 
    }
    val_vars = {
        'images': val_images,
        'neural_data': val_neural,
        'loss': val_loss,
        'score': val_accuracy,
    }
    extra_params = {
        'cm_type': cm_type,
        'layer_name': layer_name,
        'output_type': output_type,
        'project_name': project_name,
        'model_type': model_type,
        'lesions': config.lesions,
        'timesteps': timesteps
    }
    np.savez(
        os.path.join(out_dir, 'training_config_file'),
        config=config,
        extra_params=extra_params)
    train_cv_out, val_cv_out, weights = training.training_loop(
        config=config,
        neural_data=neural_data,
        images=images,
        target_size=config.img_shape[:2],
        sess=sess,
        train_vars=train_vars,
        val_vars=val_vars,
        summary_op=summary_op,
        summary_writer=summary_writer,
        checkpoint_dir=checkpoint_dir,
        summary_dir=summary_dir,
        saver=saver)
    np.savez(
        os.path.join(out_dir, 'training_data'),
        config=config,
        extra_params=extra_params,
        train_cv_out=train_cv_out,
        val_cv_out=val_cv_out,
        weight=weights)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--cm_type',
        type=str,
        dest='cm_type')
    parser.add_argument(
        '--layer_name',
        type=str,
        dest='layer_name')
    parser.add_argument(
        '--output_type',
        type=str,
        dest='output_type')
    parser.add_argument(
        '--project_name',
        type=str,
        dest='project_name')
    parser.add_argument(
        '--model_type',
        type=str,
        dest='model_type',
        default='vgg16')
    parser.add_argument(
        '--timesteps',
        type=int,
        dest='timesteps')
    args = parser.parse_args()
    extract_vgg_features(**vars(args))
