"""Functions for handling feedforward and pooling operations."""
import tensorflow as tf
from ops import initialization


def pool_ff_interpreter(
        self,
        it_neuron_op,
        act,
        it_name,
        out_channels,
        aux=None):
    """Wrapper for FF and pooling functions."""
    assert out_channels is not None, 'How mny output units do you need?'
    if it_neuron_op == 'fc':
        self, act, weights = fc_layer(
            self=self,
            bottom=act,
            in_channels=int(act.get_shape()[-1]),
            out_channels=out_channels,
            name=it_name)
    elif it_neuron_op == '1x1conv':
        self, act, weights = conv_layer(
            self=self,
            bottom=act,
            in_channels=int(act.get_shape()[-1]),
            out_channels=out_channels,
            name=it_name,
            filter_size=1)
    elif it_neuron_op == 'sparse_pool':
        self, act, weights = sparse_pool_layer(
            self=self,
            bottom=act,
            in_channels=int(act.get_shape()[-1]),
            out_channels=out_channels,
            aux=aux,
            name=it_name)
    elif it_neuron_op == 'pass':
        pass
    else:
        raise NotImplementedError(it_neuron_op)
    return self, act, weights


def fc_layer(self, bottom, out_channels, name, in_channels=None):
    """Fully connected layer."""
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        self, weights, biases = get_fc_var(
            self=self,
            in_size=in_channels,
            out_size=out_channels,
            name=name)

        x = tf.reshape(bottom, [-1, in_channels])
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        return self, fc, {'weights': weights}


def conv_layer(
        self,
        bottom,
        in_channels,
        out_channels,
        name,
        filter_size=3,
        batchnorm=None):
    with tf.variable_scope(name):
        filt, conv_biases = self.get_conv_var(
            filter_size, in_channels, out_channels, name)

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, conv_biases)
        relu = tf.nn.relu(bias)

        if batchnorm is not None:
            if name in batchnorm:
                relu = self.batchnorm(relu)

        return self, relu, {'reduce_weights': filt}


def sparse_pool_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        aux=None):
    """Sparse pooling layer."""
    def create_gaussian_rf(xy, h, w):
        """Create a gaussian bump for initializing the spatial weights."""
        # TODO: implement this.
        pass

    with tf.variable_scope(name):
        bottom_shape = [int(x) for x in bottom.get_shape()]
        if in_channels is None:
            in_channels = bottom_shape[-1]

        # K channel weights
        channel_weights = tf.get_variable(
            name='%s_channel' % name,
            dtype=tf.float32,
            initializer=initialization.xavier_initializer(
                shape=[in_channels, out_channels],
                uniform=True,
                mask=None))

        # HxW spatial weights
        spatial_weights = tf.get_variable(
            name='%s_spatial' % name,
            dtype=tf.float32,
            initializer=initialization.xavier_initializer(
                shape=[1, bottom_shape[1], bottom_shape[2], 1],
                mask=None))

        # If supplied, initialize the spatial weights with RF info
        if aux is not None and 'xy' in aux.keys():
            gaussian_xy = aux['xy']
            if 'h' in aux.keys():
                gaussian_h = aux['h']
                gaussian_w = aux['w']
            else:
                gaussian_h, gaussian_w = None, None
            spatial_rf = create_gaussian_rf(
                xy=gaussian_xy,
                h=gaussian_h,
                w=gaussian_w)
            spatial_weights += spatial_rf
        spatial_sparse = tf.reduce_mean(
            bottom * spatial_weights, reduction_indices=[1, 2])
        output = tf.matmul(spatial_sparse, channel_weights)
        return self, output, {
            'spatial_weights': spatial_weights,
            'channel_weights': channel_weights}


def get_conv_var(
        self,
        filter_size,
        in_channels,
        out_channels,
        name,
        init_type='xavier'):
    """Prepare convolutional kernel weights."""
    if init_type == 'xavier':
        weight_init = [
            [filter_size, filter_size, in_channels, out_channels],
            tf.contrib.layers.xavier_initializer_conv2d(uniform=False)]
    else:
        weight_init = tf.truncated_normal(
            [filter_size, filter_size, in_channels, out_channels],
            0.0, 0.001)
    bias_init = tf.truncated_normal([out_channels], .0, .001)
    self, filters = get_var(
        self=self,
        initial_value=weight_init,
        name=name,
        idx=0,
        var_name=name + "_filters")
    self, biases = get_var(
        self=self,
        initial_value=bias_init,
        name=name,
        idx=1,
        var_name=name + "_biases")
    return self, filters, biases


def get_fc_var(
        self,
        in_size,
        out_size,
        name,
        init_type='xavier'):
    """Prepare fully connected weights."""
    if init_type == 'xavier':
        weight_init = [
            [in_size, out_size],
            tf.contrib.layers.xavier_initializer(uniform=False)]
    else:
        weight_init = tf.truncated_normal(
            [in_size, out_size], 0.0, 0.001)
    bias_init = tf.truncated_normal([out_size], .0, .001)
    self, weights = get_var(
        self=self,
        initial_value=weight_init,
        name=name,
        idx=0,
        var_name=name + "_weights")
    self, biases = get_var(
        self=self,
        initial_value=bias_init,
        name=name,
        idx=1,
        var_name=name + "_biases")
    return self, weights, biases


def get_var(
        self,
        initial_value,
        name,
        idx,
        var_name,
        in_size=None,
        out_size=None):
    """Handle variable loading if necessary."""
    if self.data_dict is not None and name in self.data_dict:
        value = self.data_dict[name][idx]
    else:
        value = initial_value

    if self.training:
        # get_variable, change the boolian to numpy
        if type(value) is list:
            var = tf.get_variable(
                name=var_name,
                shape=value[0],
                initializer=value[1],
                trainable=True)
        else:
            var = tf.get_variable(
                name=var_name,
                initializer=value,
                trainable=True)
    else:
        var = tf.constant(
            value,
            dtype=tf.float32,
            name=var_name)
    self.var_dict[(name, idx)] = var
    return self, var
