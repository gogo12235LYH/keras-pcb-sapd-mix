import numpy as np
import tensorflow.keras.backend as k


class AnchorParameters:
    def __init__(self,
                 sizes,
                 strides,
                 ratios,
                 scales,
                 interest_sizes
                 ):
        self.sizes = sizes
        self.strides = strides
        self.ratios = ratios
        self.scales = scales
        self.interest_sizes = interest_sizes

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


AnchorParameters.default = AnchorParameters(
    sizes=[32, 64, 128, 256, 512],
    strides=[8, 16, 32, 64, 128],
    ratios=np.array([0.5, 1, 2], k.floatx()),
    scales=np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], k.floatx()),
    # (5, 2)
    interest_sizes=[
        [-1, 64],
        [64, 128],
        [128, 256],
        [256, 512],
        [512, 1e8]
    ]
)


def guess_shapes(
        image_shape,
        pyramid_levels=(3, 4, 5, 6, 7)
):
    image_shape = np.array(image_shape[:2])
    # feature_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    feature_shapes = np.zeros((5, 2))

    for x in range(3, 7+1):
        feature_shapes[x - 3, 0] = (image_shape[0] + 2 ** x - 1) // (2 ** x)
        feature_shapes[x - 3, 1] = (image_shape[1] + 2 ** x - 1) // (2 ** x)
    return feature_shapes


def compute_locations_per_level(height, width, stride):
    shift_x = np.arange(0, width * stride, step=stride, dtype=np.float32)
    shift_y = np.arange(0, height * stride, step=stride, dtype=np.float32)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return np.stack([shift_x, shift_y], axis=1) + stride // 2


def compute_locations(feature_shapes, anchor_params=None):
    if anchor_params is None:
        anchor_params = AnchorParameters.default

    fpn_strides = anchor_params.strides
    locations = []
    for level, (feature_shape, fpn_stride) in enumerate(zip(feature_shapes, fpn_strides)):
        h, w = feature_shape
        loc_per_lv = compute_locations_per_level(h, w, fpn_stride)
        locations.append(loc_per_lv)

    return locations


def compute_interest_sizes(num_loc_each_lv, anchor_params=None):
    if anchor_params is None:
        anchor_params = AnchorParameters.default

    interest_sizes = anchor_params.interest_sizes
    assert len(num_loc_each_lv) == len(interest_sizes)

    tiled_interest_sizes = []
    for num_loc, interest_size in zip(num_loc_each_lv, interest_sizes):
        # (2,)
        interest_size = np.array(interest_size)
        # (1, 2)
        interest_size = np.expand_dims(interest_size, axis=0)
        # (num_loc, 2)
        interest_size = np.tile(interest_size, (num_loc, 1))

        tiled_interest_sizes.append(interest_size)

    # if image_shape = (640, 640, 3)
    # return shape = (8525, 2)
    return np.concatenate(tiled_interest_sizes, axis=0)
