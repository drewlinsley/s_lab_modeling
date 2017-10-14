"""Wrapper for normalization functions."""
from utils import py_utils
from layers.normalization_functions import contextual_vector
from layers.normalization_functions import contextual_vector_separable


class normalizations(object):
    """Wrapper class for activation functions."""

    def __getitem__(self, name):
        """Get attribute from class."""
        return getattr(self, name)

    def __contains__(self, name):
        """Check if class contains attribute."""
        return hasattr(self, name)

    def __init__(self, kwargs=None):
        """Global variables for normalization functions."""
        self.timesteps = 3
        self.scale_CRF = True
        self.bias_CRF = True
        self.lesions = [None]
        self.training = None
        self.strides = [1, 1, 1, 1]
        self.padding = 'SAME'
        self.update_params(kwargs)

    def update_params(self, kwargs):
        """Update the class attributes with kwargs."""
        if kwargs is not None:
            for k, v in kwargs.iteritems():
                setattr(self, k, v)

    def calculate_kernel_from_RF(self, j_in, r_in, fix_r_out):
        """Calculate kernel size for a specified RF."""
        return (fix_r_out / j_in) + 1 - r_in

    def set_RFs(
            self,
            r_in=None,
            j_in=None,
            V1_CRF=0.26,
            V1_neCRF=0.54,
            V1_feCRF=1.41,
            default_stride=1,
            padding=1):
        """Set RF sizes for the normalizations.

        Based on calculation of an effective RF (i.e. not
        simply kernel size of the current layer, but instead
        w.r.t. input).

        Angelucci & Shushruth 2013 V1 RFs:
        https://www.shadlenlab.columbia.edu/people/Shushruth/Lab_Page/Home_files/GalleyProof.pdf
        CRF = 0.26 degrees (1x)
        eCRF near = 0.54 degrees (2x)
        eCRF far = 1.41 degrees (5.5x)

        Implementation is to calculate the RF of a computational unit in
        an activity tensor. Then, near and far eCRFs are derived relative
        to the CRF size. This means that the *filter size* for the CRF is 1
        tensor pixel. And the eRFs are deduced as the appropriate filter size
        for their calculated RFs.

        For instance, units in pool_2 of VGG16 have RFs of 16x16 pixels of
        the input image. Setting the CRF filter size to 1, this means that the
        near eCRF filter size must capture an RF of ~ 32x32 pixels, and the
        far eCRF filter size must capture an RF of ~ 80x80 pixels. The eRF
        calculator can deduce these filter sizes.
        """
        assert r_in is not None, 'You must pass an RF input size.'
        assert j_in is not None, 'You must pass an input jump.'
        self.SRF = 1  # See explanation above.
        self.CRF_excitation = 1
        self.CRF_inhibition = 1
        SSN_eRF = py_utils.iceil(r_in * (V1_neCRF / V1_CRF))
        self.SSN = self.calculate_kernel_from_RF(
            j_in=j_in,
            r_in=r_in,
            fix_r_out=SSN_eRF)
        SSF_eRF = py_utils.iceil(r_in * (V1_feCRF / V1_CRF))
        self.SSF = self.calculate_kernel_from_RF(
            j_in=j_in,
            r_in=r_in,
            fix_r_out=SSF_eRF)

    def contextual_vector(self, x, r_in, j_in, timesteps, lesions):
        """Contextual model."""
        self.set_RFs(r_in=r_in, j_in=j_in)
        contextual_layer = contextual_vector.ContextualCircuit(
            X=x,
            timesteps=timesteps,
            lesions=lesions,
            SRF=self.SRF,
            SSN=self.SSN,
            SSF=self.SSF,
            strides=self.strides,
            padding=self.padding)
        return contextual_layer.build()

    def contextual_vector_separable(self, x, r_in, j_in, timesteps, lesions):
        """Contextual model with separable convolutions."""
        self.set_RFs(r_in=r_in, j_in=j_in)
        contextual_layer = contextual_vector_separable.ContextualCircuit(
            X=x,
            timesteps=timesteps,
            lesions=lesions,
            SRF=self.SRF,
            SSN=self.SSN,
            SSF=self.SSF,
            strides=self.strides,
            padding=self.padding)
        return contextual_layer.build()

