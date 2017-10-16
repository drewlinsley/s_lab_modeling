"""Config for s-lab project."""
import os


class Config(object):
    """S-lab project config file."""

    def __init__(self, **kwargs):
        """Declare config attributes."""
        self.data_directory = '/media/data_cifs/image_datasets/'
        self.results = '/media/data_cifs/contextual_circuit/results'
        self.summaries = '/media/data_cifs/contextual_circuit/summaries'
        self.vgg16_weight_path = '/media/data_cifs/clicktionary/pretrained_weights/vgg16.npy'
        self.projects = {
            'sheinberg_data': os.path.join(
                self.data_directory,
                'sheinberg_data'),
            'sheinberg_data_noise_subtracted': os.path.join(
                self.data_directory,
                'sheinberg_data_noise_subtracted'),
        }

        # Model parameters
        self.img_shape = [224, 224, 3]
        self.lesions = [None]
        self.reduce_features = 100
        self.num_epochs = 30
        self.batch_size = 10
        self.loss_type = 'pearson'
        self.lr = 1e-3
        self.optimizer = 'adam'
        self.metric = 'pearson'
        # self.cv = {
        #     'k_fold': 1  # 8
        # }
        self.cv = {
            'hold_out': 0.9  # 8
        }

        # Contextual model weight decay
        self.cm_wd_types = {
            'q_t': {
                'type': 'l1',
                'strength': 0.1
            },
            'p_t': {
                'type': 'l1',
                'strength': 0.1
            },
            't_t': {
                'type': 'l1',
                'strength': 0.01
            }
        }

        # Output layer connectivity
        self.wd_types = {
            'weights': {
                'type': 'l2',
                'strength': 1.
            },
            'spatial_weights': {
                'type': 'l2',  # 'laplace_l2',
                'strength': 1.
            },
            'channel_weights': {
                'type': 'l2',
                'strength': 1.
            },
            'reduce_weights': {
                'type': 'l1',
                'strength': 0.1
            }
        }

        # update attributes
        self.__dict__.update(kwargs)
