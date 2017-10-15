"""Config for s-lab project."""
import os


class Config(object):
    """S-lab project config file."""

    def __init__(self, **kwargs):
        """Declare config attributes."""
        self.data_directory = '/Users/drewlinsley/Desktop/s_lab_data'
        self.results = '/Users/drewlinsley/Desktop/s_lab_data/results'
        self.summaries = '/Users/drewlinsley/Desktop/s_lab_data/summaries'
        self.vgg16_weight_path = '/Users/drewlinsley/Downloads/vgg16.npy'
        self.projects = {
            'sheinberg_data': os.path.join(
                self.data_directory,
                'sheinberg_data'),
            'sheinberg_data_noise_subtracted': os.path.join(
                self.data_directory,
                'sheinberg_data_noise_subtracted'),
        }

        # Model parameters
        self.lesions = [None]
        self.batch_size = 10
        self.loss_type = 'l2'
        self.lr = 3e-4
        self.optimizer = 'adam'
        self.metric = 'pearson'
        self.cv = {
            'k_fold': 1  # 8
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
            }
        }

        # update attributes
        self.__dict__.update(kwargs)
