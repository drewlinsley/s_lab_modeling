"""Config for s-lab project."""
import os


class Config(object):
    """S-lab project config file."""

    def __init__(self, **kwargs):
        """Declare config attributes."""
        self.data_directory = '/Users/drewlinsley/Desktop/s_lab_data'
        self.projects = {
            'sheinberg_data': os.path.join(
                self.data_directory,
                'sheinberg_data'),
            'sheinberg_data_noise_subtracted': os.path.join(
                self.data_directory,
                'sheinberg_data_noise_subtracted'),
        }
        self.lesions = [None]

        # update attributes
        self.__dict__.update(kwargs)
