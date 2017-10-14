"""VGG16 RF sizes."""


def get_eRFs(model_name):
    """Interpret and return eRF sizes of model layers."""
    if model_name == 'vgg16':
        return {
            'conv1_1': {
                'r_in': 3,
                'j_in': 1
            },
            'conv1_2': {
                'r_in': 5,
                'j_in': 1
            },
            'pool_1': {
                'r_in': 6,
                'j_in': 2
            },
            'conv1_1': {
                'r_in': 10,
                'j_in': 2
            },
            'conv2_2': {
                'r_in': 14,
                'j_in': 2
            },
            'pool_2': {
                'r_in': 16,
                'j_in': 4
            },
            'conv3_1': {
                'r_in': 24,
                'j_in': 4
            },
            'conv3_2': {
                'r_in': 32,
                'j_in': 4
            },
            'conv3_3': {
                'r_in': 40,
                'j_in': 4
            },
            'pool_3': {
                'r_in': 44,
                'j_in': 8
            },
            'conv4_1': {
                'r_in': 60,
                'j_in': 8
            },
            'conv4_2': {
                'r_in': 76,
                'j_in': 8
            },
            'conv4_3': {
                'r_in': 92,
                'j_in': 8
            },
            'pool_4': {
                'r_in': 100,
                'j_in': 16
            },
            'conv5_1': {
                'r_in': 132,
                'j_in': 16
            },
            'conv5_2': {
                'r_in': 164,
                'j_in': 16
            },
            'conv5_3': {
                'r_in': 196,
                'j_in': 16
            },
            'pool_5': {
                'r_in': 212,
                'j_in': 32
            },
            'fc6': {
                'r_in': 244,
                'j_in': 32
            },
            'fc7': {
                'r_in': 276,
                'j_in': 32
            },
        }
    else:
        raise NotImplementedError
