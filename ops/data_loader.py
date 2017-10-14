import numpy as np
import tensorflow as tf


def repeat_elements(x, rep, axis):
    '''Repeats the elements of a tensor along an axis, like np.repeat
    If x has shape (s1, s2, s3) and axis=1, the output
    will have shape (s1, s2 * rep, s3)
    This function is taken from keras backend
    '''
    x_shape = x.get_shape().as_list()
    splits = tf.split(axis=axis, num_or_size_splits=x_shape[axis], value=x)
    x_rep = [s for s in splits for i in range(rep)]
    return tf.concat(axis=axis, values=x_rep)


def image_augmentations(
        image,
        data_augmentations,
        model_input_image_size):
    """Coordinating image augmentations for both image and heatmap."""
    im_size = [int(x) for x in image.get_shape()]
    im_size_check = any(np.less_equal(im_size[:2], model_input_image_size[:2]))
    if data_augmentations is not None:
        if 'random_crop' in data_augmentations and im_size_check:
            image = tf.random_crop(image, model_input_image_size)
        elif 'resize' in data_augmentations and im_size_check:
            if len(model_input_image_size) > 2:
                model_input_image_size = model_input_image_size[:2]
            image = tf.image.resize_images(
                tf.expand_dims(image, axis=0), model_input_image_size)
            image = tf.squeeze(image, axis=0)
        else:
            image = tf.image.resize_image_with_crop_or_pad(
                image, model_input_image_size[0], model_input_image_size[1])
        if 'left_right' in data_augmentations:
            image = tf.image.random_flip_left_right(image)
        if 'up_down' in data_augmentations:
            image = tf.image.random_flip_up_down(image)
        if 'random_contrast' in data_augmentations:
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        if 'random_brightness' in data_augmentations:
            image = tf.image.random_brightness(image, max_delta=63.)
    else:
        image = tf.image.resize_image_with_crop_or_pad(
            image, model_input_image_size[0], model_input_image_size[1])
    return image


def read_and_decode(
        filename_queue,
        model_input_image_size,
        tf_dict,
        tf_reader_settings,
        data_augmentations):
    """Read and decode tensors from tf_records and apply augmentations."""
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features=tf_dict)

    # Handle decoding of each element
    image = tf.decode_raw(
        features['image'],
        tf_reader_settings['image']['dtype'])
    label = tf.cast(
        features['label'],
        tf_reader_settings['label']['dtype'])

    # Reshape each element
    image = tf.reshape(image, tf_reader_settings['image']['reshape'])
    if tf_reader_settings['label']['reshape'] is not None:
        label = tf.reshape(label, tf_reader_settings['label']['reshape'])

    # Preprocess images and heatmaps
    image = image_augmentations(
        image=image,
        model_input_image_size=model_input_image_size,
        data_augmentations=data_augmentations)
    return image, label


def inputs(
        dataset,
        batch_size,
        model_input_image_size,
        tf_dict,
        data_augmentations,
        num_epochs,
        tf_reader_settings,
        shuffle):
    """Read tfrecords and prepare them for queueing."""
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 5 * batch_size
    num_threads = 2
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [dataset], num_epochs=num_epochs)
        # Even when reading in multiple threads, share the filename
        # queue.
        batch_data = read_and_decode(
            filename_queue=filename_queue,
            model_input_image_size=model_input_image_size,
            tf_dict=tf_dict,
            tf_reader_settings=tf_reader_settings,
            data_augmentations=data_augmentations)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        if shuffle:
            images, labels = tf.train.shuffle_batch(
                batch_data,
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=capacity,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=min_after_dequeue)
        else:
            images, labels = tf.train.batch(
                batch_data,
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=capacity)
        return images, labels
